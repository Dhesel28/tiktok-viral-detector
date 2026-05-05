"""
Training Script
===============
Trains two NLP models from the lyrics catalog and saves them to models/.

Model 1 — Viral Line Classifier
  Algorithm : Gradient Boosting Classifier
  Input     : 11 hand-crafted features per lyric line
  Label     : 1 if the line lives in a [Chorus/Hook/Bridge] section
               OR repeats 3+ times in the song  (weak supervision)
  Output    : viral probability (0-1) for every line

Model 2 — Song Similarity Model
  Algorithm : TF-IDF vectorizer  +  cosine NearestNeighbors
  Input     : full plain-text lyrics of a song
  Output    : top-k most similar songs from the catalog

Usage:
  python train.py              # train both models
  python train.py --eval       # train + print evaluation metrics
"""

from __future__ import annotations

import re
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))
from config import CATALOG_CSV, DATA_DIR

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

VIRAL_MODEL_PATH  = MODELS_DIR / "viral_line_model.pkl"
SIM_MODEL_PATH    = MODELS_DIR / "similarity_model.pkl"

# ── Word sets ──────────────────────────────────────────────────────────────────

ENERGY_WORDS = {
    "fire","lit","hype","wild","crazy","turnt","vibe","bounce","drop","bang",
    "banger","loud","scream","jump","move","go","fast","hard","rage","beast",
    "king","queen","boss","flex","drip","swag","pop","rock","hit","rise",
    "shine","burn","blaze","fly","rush","push","break","unstoppable","epic",
}
POSITIVE_WORDS = {
    "love","happy","joy","smile","beautiful","wonderful","amazing","great",
    "good","best","perfect","bright","warm","sweet","dream","hope","win",
    "glory","bless","grace","light","gold","free","peace","together",
    "forever","always","paradise","heaven",
}
NEGATIVE_WORDS = {
    "hate","pain","cry","tears","hurt","break","lost","alone","dark","cold",
    "dead","fall","fail","lie","cheat","betray","gone","miss","sad","empty",
    "numb","sorrow","grief","regret","fear","wrong","bad","worst","hell",
    "die","kill","blood",
}

HOOK_SECTIONS  = {"chorus","hook","refrain","bridge","pre-chorus","prechorus"}
_SECTION_RE    = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)
_REPEAT_THRESH = 3   # lines repeating >= this are labeled positive


# ═══════════════════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _last3(line: str) -> str:
    words = line.strip().split()
    return words[-1].lower()[-3:] if words else ""

def _rhymes(a: str, b: str) -> bool:
    ea, eb = _last3(a), _last3(b)
    return len(ea) >= 2 and ea == eb

def _word_set(line: str) -> set:
    return set(re.sub(r"[^\w\s]", " ", line.lower()).split())


def extract_line_features(
    text: str,
    prev_text: str,
    next_text: str,
    repetition_score: float,
    is_repeated: int,
    position_ratio: float,
) -> list[float]:
    """
    Return 11 features for a single lyric line.

    Features:
      0  repetition_score   — count / max_count in song
      1  energy_density     — energy words / word count
      2  rhyme_prev         — rhymes with previous line
      3  rhyme_next         — rhymes with next line
      4  brevity            — optimal if 4-9 words
      5  positive_density   — positive words / word count
      6  negative_density   — negative words / word count
      7  position_ratio     — position in song (0=start, 1=end)
      8  word_count_norm    — word count / 20, capped at 1
      9  is_repeated        — 1 if this line repeats anywhere
      10 strong_start       — first word is energy/emotional
    """
    ws  = _word_set(text)
    wc  = max(len(text.split()), 1)

    energy   = min(len(ws & ENERGY_WORDS)   / max(wc, 1) * 5, 1.0)
    pos_sent = min(len(ws & POSITIVE_WORDS) / max(wc, 1) * 5, 1.0)
    neg_sent = min(len(ws & NEGATIVE_WORDS) / max(wc, 1) * 5, 1.0)

    n = len(text.split())
    if 4 <= n <= 9:
        brevity = 1.0
    elif n < 4:
        brevity = 0.4 + n * 0.1
    else:
        brevity = max(0.0, 1.0 - (n - 9) * 0.07)

    first_word = text.split()[0].lower().strip(".,!?") if text.split() else ""
    strong_start = 1.0 if first_word in ENERGY_WORDS | POSITIVE_WORDS | NEGATIVE_WORDS else 0.0

    return [
        float(repetition_score),          # 0
        float(energy),                    # 1
        1.0 if _rhymes(text, prev_text) else 0.0,  # 2
        1.0 if _rhymes(text, next_text) else 0.0,  # 3
        float(brevity),                   # 4
        float(pos_sent),                  # 5
        float(neg_sent),                  # 6
        float(position_ratio),            # 7
        min(wc / 20.0, 1.0),              # 8
        float(is_repeated),               # 9
        float(strong_start),              # 10
    ]

FEATURE_NAMES = [
    "repetition_score", "energy_density", "rhyme_prev", "rhyme_next",
    "brevity", "positive_density", "negative_density", "position_ratio",
    "word_count_norm", "is_repeated", "strong_start",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Build training data
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y) from catalog DataFrame.
    Returns X of shape (n_lines, 11) and y of shape (n_lines,).
    """
    X_rows, y_rows = [], []

    for _, row in df.iterrows():
        lyrics = str(row.get("lyrics", "")).strip()
        if not lyrics:
            continue

        # Parse lines with section labels
        current_sec = "intro"
        raw_entries: list[tuple[str, str]] = []   # (section, text)
        for raw in lyrics.split("\n"):
            s = raw.strip()
            if not s:
                continue
            m = _SECTION_RE.match(s)
            if m:
                current_sec = m.group(1).lower().strip()
                current_sec = re.sub(r'\s*\d+$', '', current_sec).strip()
            else:
                raw_entries.append((current_sec, s))

        if not raw_entries:
            continue

        texts       = [t for _, t in raw_entries]
        sections    = [s for s, _ in raw_entries]
        total_lines = len(texts)

        # Repetition counts
        line_counts = Counter(t.lower() for t in texts)
        max_count   = max(line_counts.values(), default=1)

        for i, (sec, text) in enumerate(zip(sections, texts)):
            cnt = line_counts[text.lower()]
            rep_score = (cnt - 1) / max(max_count - 1, 1) if max_count > 1 else 0.0

            feat = extract_line_features(
                text       = text,
                prev_text  = texts[i - 1] if i > 0 else "",
                next_text  = texts[i + 1] if i < total_lines - 1 else "",
                repetition_score = rep_score,
                is_repeated      = 1 if cnt >= 2 else 0,
                position_ratio   = i / max(total_lines - 1, 1),
            )

            # Weak label: 1 if in hook section OR repeats >= threshold
            in_hook = any(k in sec for k in HOOK_SECTIONS)
            label   = 1 if (in_hook or cnt >= _REPEAT_THRESH) else 0

            X_rows.append(feat)
            y_rows.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# Model 1 — Viral Line Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def train_viral_line_model(
    X: np.ndarray,
    y: np.ndarray,
    evaluate: bool = False,
) -> tuple[GradientBoostingClassifier, StandardScaler]:

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators   = 300,
        max_depth      = 4,
        learning_rate  = 0.08,
        subsample      = 0.85,
        min_samples_leaf = 20,
        random_state   = 42,
    )
    clf.fit(X_sc, y)

    if evaluate:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="f1")
        print(f"  5-fold CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

        y_pred = clf.predict(X_sc)
        print(classification_report(y, y_pred,
              target_names=["Non-Viral", "Viral"], digits=3))

        # Feature importance
        fi = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True)
        print("\nFeature importances:")
        for name, imp in fi:
            bar = "█" * int(imp * 40)
            print(f"  {name:<22} {imp:.4f}  {bar}")

    return clf, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# Model 2 — Song Similarity
# ═══════════════════════════════════════════════════════════════════════════════

def _plain_text(lyrics: str) -> str:
    return _SECTION_RE.sub("", lyrics).strip()


def train_similarity_model(df: pd.DataFrame) -> dict:
    """
    Returns a dict with: vectorizer, nn_model, catalog_df, tfidf_matrix
    """
    has_lyr = df[
        df["lyrics"].notna()
        & (df["lyrics"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    plain_texts = has_lyr["lyrics"].apply(lambda x: _plain_text(str(x))).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range  = (1, 2),
        min_df       = 1,           # keep all terms so shared_vocab is populated
        max_df       = 0.90,
        sublinear_tf = True,
        max_features = 25_000,
        strip_accents = "unicode",
    )
    tfidf_mat = vectorizer.fit_transform(plain_texts)

    # NearestNeighbors with cosine metric — works on sparse matrices
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(tfidf_mat)

    # Store per-song top UNIGRAM TF-IDF terms for explanation (bigrams too specific)
    feature_names = vectorizer.get_feature_names_out()
    top_terms: list[list[str]] = []
    for i in range(tfidf_mat.shape[0]):
        row   = tfidf_mat[i].toarray().flatten()
        order = np.argsort(row)[::-1]
        terms = []
        for j in order:
            if row[j] <= 0:
                break
            t = feature_names[j]
            if " " not in t and len(t) > 2:   # unigrams only
                terms.append(t)
            if len(terms) >= 12:
                break
        top_terms.append(terms)

    # Build raw meaningful token sets per song (for shared-vocab display)
    _STOP = {
        "i","me","my","we","our","you","your","he","she","they","it","a","an","the",
        "and","or","but","in","on","at","to","for","of","with","by","from","up",
        "is","are","was","be","have","has","do","did","will","not","so","this","that",
        "yeah","oh","ooh","na","la","da","hey","uh","mm","ya","gonna","gotta","wanna",
    }
    def _tokens(text: str) -> set:
        import re as _re
        words = _re.sub(r"[^\w\s]", " ", text.lower()).split()
        return {w for w in words if len(w) > 2 and w not in _STOP}

    token_sets = [_tokens(t) for t in plain_texts]

    has_lyr = has_lyr.copy()
    has_lyr["top_terms"] = top_terms

    return {
        "vectorizer":   vectorizer,
        "nn_model":     nn,
        "catalog_df":   has_lyr,
        "tfidf_matrix": tfidf_mat,
        "token_sets":   token_sets,   # list[set] for shared vocab display
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Save / load
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(
    clf:        GradientBoostingClassifier,
    scaler:     StandardScaler,
    sim_bundle: dict,
) -> None:
    with open(VIRAL_MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f, protocol=5)
    print(f"  Viral line model → {VIRAL_MODEL_PATH}")

    # Don't pickle sparse tfidf_matrix inside catalog_df — store separately
    save_bundle = {
        "vectorizer":   sim_bundle["vectorizer"],
        "nn_model":     sim_bundle["nn_model"],
        "catalog_df":   sim_bundle["catalog_df"].drop(columns=["lyrics"], errors="ignore"),
        "tfidf_matrix": sim_bundle["tfidf_matrix"],
        "token_sets":   sim_bundle["token_sets"],
    }
    with open(SIM_MODEL_PATH, "wb") as f:
        pickle.dump(save_bundle, f, protocol=5)
    print(f"  Similarity model  → {SIM_MODEL_PATH}")


def models_exist() -> bool:
    return VIRAL_MODEL_PATH.exists() and SIM_MODEL_PATH.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(evaluate: bool = False) -> None:
    if not CATALOG_CSV.exists():
        print(f"[!] No catalog at {CATALOG_CSV}. Run: python pipeline.py --build")
        return

    print("[Train] Loading catalog …")
    df = pd.read_csv(CATALOG_CSV)
    has_lyr = df[df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")]
    print(f"  Songs with lyrics: {len(has_lyr)}")

    # ── Model 1 ─────────────────────────────────────────────────────────────
    print("\n[Train] Building line-level training data …")
    X, y = build_training_data(has_lyr)
    print(f"  Lines: {len(X)}  |  Viral: {y.sum()} ({round(y.mean()*100)}%)  "
          f"|  Non-viral: {(y==0).sum()} ({round((y==0).mean()*100)}%)")

    print("[Train] Training Viral Line Classifier (GradientBoosting) …")
    clf, scaler = train_viral_line_model(X, y, evaluate=evaluate)
    print("  Done.")

    # ── Model 2 ─────────────────────────────────────────────────────────────
    print("\n[Train] Training Song Similarity Model (TF-IDF + NearestNeighbors) …")
    sim_bundle = train_similarity_model(has_lyr)
    print(f"  Catalog: {sim_bundle['tfidf_matrix'].shape[0]} songs  "
          f"|  Vocab: {sim_bundle['tfidf_matrix'].shape[1]:,} terms")

    # ── Save Models 1 & 2 ────────────────────────────────────────────────────
    print("\n[Train] Saving models …")
    save_models(clf, scaler, sim_bundle)

    # ── Model 3 — Song-Level TikTok Virality Classifier ──────────────────────
    print("\n[Train] Training Song-Level Virality Classifier (Model 3) …")
    from song_viral_model import (
        build_song_training_data, train_song_model, save_song_model,
    )
    from nlp_models import ViralLineDetector

    # Requires Model 1 to be saved (provides line-level probs as features)
    detector = ViralLineDetector()
    X_song, y_song = build_song_training_data(has_lyr, detector)
    print(f"  Songs: {len(X_song)}  |  Viral: {y_song.sum()} ({round(y_song.mean()*100)}%)"
          f"  |  Non-viral: {(y_song==0).sum()} ({round((y_song==0).mean()*100)}%)")

    clf_song, scaler_song = train_song_model(X_song, y_song, evaluate=evaluate)
    save_song_model(clf_song, scaler_song)
    print("\n[Train] All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Print cross-validation and feature importance")
    args = parser.parse_args()
    main(evaluate=args.eval)
