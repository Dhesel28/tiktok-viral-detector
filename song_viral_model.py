"""
Song-Level TikTok Virality Classifier  (Model 3)
==================================================
Predicts whether a full song's lyrics have TikTok viral potential.

Training labels (from catalog_with_lyrics.csv tier column):
  Tier -1  TikTok trending songs          → viral = 1
  Tier  3  Major hits                     → viral = 1
  Tier  4  Super hits                     → viral = 1
  Tier  0  Non-hits                       → viral = 0
  Tier  1  Minor hits                     → viral = 0
  Tier  2  (ambiguous mid-tier, skipped)

Features — 13 song-level aggregates:
  0  hook_line_ratio       % of all lines in chorus/hook/bridge sections
  1  repetition_rate       fraction of lines that repeat (1 - unique/total)
  2  max_repeats_norm      most-repeated line count / 10, capped at 1
  3  avg_energy            mean energy-word density across lines
  4  avg_positive          mean positive-sentiment density
  5  avg_negative          mean negative-sentiment density
  6  avg_brevity           mean brevity score (optimal = 4-9 words)
  7  avg_viral_prob        mean line-level viral probability from Model 1
  8  top_viral_prob        single highest line-level viral probability
  9  strong_hook_density   fraction of lines with viral_prob ≥ 0.55
  10 viral_line_density    fraction of lines with viral_prob ≥ 0.75
  11 chorus_repeat_ratio   fraction of section types that are hook sections
  12 line_count_norm       total lyric lines / 80, capped at 1
"""

from __future__ import annotations

import re
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).parent

SONG_MODEL_PATH = _HERE / "models" / "song_viral_model.pkl"

SONG_FEATURE_NAMES = [
    "hook_line_ratio",
    "repetition_rate",
    "max_repeats_norm",
    "avg_energy",
    "avg_positive",
    "avg_negative",
    "avg_brevity",
    "avg_viral_prob",
    "top_viral_prob",
    "strong_hook_density",
    "viral_line_density",
    "chorus_repeat_ratio",
    "line_count_norm",
]

# Mirrors the word sets used in train.py and nlp_models.py
_ENERGY_WORDS = {
    "fire","lit","hype","wild","crazy","turnt","vibe","bounce","drop","bang",
    "banger","loud","scream","jump","move","go","fast","hard","rage","beast",
    "king","queen","boss","flex","drip","swag","pop","rock","hit","rise",
    "shine","burn","blaze","fly","rush","push","break","unstoppable","epic",
}
_POSITIVE_WORDS = {
    "love","happy","joy","smile","beautiful","wonderful","amazing","great",
    "good","best","perfect","bright","warm","sweet","dream","hope","win",
    "glory","bless","grace","light","gold","free","peace","together",
    "forever","always","paradise","heaven",
}
_NEGATIVE_WORDS = {
    "hate","pain","cry","tears","hurt","break","lost","alone","dark","cold",
    "dead","fall","fail","lie","cheat","betray","gone","miss","sad","empty",
    "numb","sorrow","grief","regret","fear","wrong","bad","worst","hell",
    "die","kill","blood",
}
_HOOK_SECS = {"chorus","hook","refrain","bridge","pre-chorus","prechorus","pre chorus"}


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_song_features(
    all_lines: list[dict],
) -> np.ndarray:
    """
    Build the 13-dim song feature vector from the output of ViralLineDetector.predict().

    Parameters
    ----------
    all_lines : list of dicts as returned by ViralLineDetector.predict()["all_lines"]
                Each dict has keys: text, section, viral_prob
    """
    if not all_lines:
        return np.zeros(len(SONG_FEATURE_NAMES), dtype=np.float32)

    n        = len(all_lines)
    texts    = [l["text"]       for l in all_lines]
    sections = [l["section"]    for l in all_lines]
    probs    = [l["viral_prob"] for l in all_lines]

    # ── Hook / structure ─────────────────────────────────────────────────────
    hook_count       = sum(1 for s in sections if any(h in s for h in _HOOK_SECS))
    hook_line_ratio  = hook_count / n

    section_types    = set(sections)
    hook_sec_types   = sum(1 for s in section_types if any(h in s for h in _HOOK_SECS))
    chorus_repeat_ratio = hook_sec_types / max(len(section_types), 1)

    # ── Repetition ───────────────────────────────────────────────────────────
    text_counts      = Counter(t.lower() for t in texts)
    unique_count     = len(text_counts)
    repetition_rate  = 1.0 - unique_count / n
    max_repeats_norm = min(max(text_counts.values(), default=1) / 10.0, 1.0)

    # ── Word-level sentiment / energy ────────────────────────────────────────
    all_words = []
    brevities = []
    for t in texts:
        words = re.sub(r"[^\w\s]", " ", t.lower()).split()
        all_words.extend(words)
        nw = len(t.split())
        if 4 <= nw <= 9:
            brevities.append(1.0)
        elif nw < 4:
            brevities.append(0.4 + nw * 0.1)
        else:
            brevities.append(max(0.0, 1.0 - (nw - 9) * 0.07))

    wc           = max(len(all_words), 1)
    avg_energy   = min(sum(1 for w in all_words if w in _ENERGY_WORDS)   / wc * 5, 1.0)
    avg_positive = min(sum(1 for w in all_words if w in _POSITIVE_WORDS) / wc * 5, 1.0)
    avg_negative = min(sum(1 for w in all_words if w in _NEGATIVE_WORDS) / wc * 5, 1.0)
    avg_brevity  = sum(brevities) / len(brevities) if brevities else 0.5

    # ── Viral probability aggregates ─────────────────────────────────────────
    avg_viral_prob      = sum(probs) / n
    top_viral_prob      = max(probs)
    strong_hook_density = sum(1 for p in probs if p >= 0.55) / n
    viral_line_density  = sum(1 for p in probs if p >= 0.75) / n
    line_count_norm     = min(n / 80.0, 1.0)

    return np.array([
        hook_line_ratio,        # 0
        repetition_rate,        # 1
        max_repeats_norm,       # 2
        avg_energy,             # 3
        avg_positive,           # 4
        avg_negative,           # 5
        avg_brevity,            # 6
        avg_viral_prob,         # 7
        top_viral_prob,         # 8
        strong_hook_density,    # 9
        viral_line_density,     # 10
        chorus_repeat_ratio,    # 11
        line_count_norm,        # 12
    ], dtype=np.float32)


# ── Training data builder ──────────────────────────────────────────────────────

def build_song_training_data(df, detector) -> tuple:
    """
    Build (X, y) from catalog DataFrame.
    Requires a trained ViralLineDetector instance to compute line-level probs.

    Labels:
      tier in (-1, 3, 4)  → y = 1 (viral)
      tier in (0, 1)      → y = 0 (not viral)
      tier == 2           → skipped
    """
    X_rows, y_rows = [], []

    VIRAL_TIERS    = {-1, 3, 4}
    NONVIRAL_TIERS = {0, 1}

    for _, row in df.iterrows():
        tier   = int(row.get("tier", 99))
        lyrics = str(row.get("lyrics", "")).strip()

        if tier not in VIRAL_TIERS and tier not in NONVIRAL_TIERS:
            continue
        if not lyrics:
            continue

        label  = 1 if tier in VIRAL_TIERS else 0

        result = detector.predict(lyrics)
        if not result["all_lines"]:
            continue

        feat = extract_song_features(result["all_lines"])
        X_rows.append(feat)
        y_rows.append(label)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.int8),
    )


# ── Model training ─────────────────────────────────────────────────────────────

def train_song_model(
    X: np.ndarray,
    y: np.ndarray,
    evaluate: bool = False,
) -> tuple:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf = GradientBoostingClassifier(
        n_estimators    = 200,
        max_depth       = 3,
        learning_rate   = 0.08,
        subsample       = 0.8,
        min_samples_leaf= 5,
        random_state    = 42,
    )
    clf.fit(X_sc, y)

    if evaluate:
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="f1")
        print(f"  5-fold CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
        y_pred = clf.predict(X_sc)
        print(classification_report(y, y_pred, target_names=["Non-Viral", "Viral"], digits=3))

        fi = sorted(zip(SONG_FEATURE_NAMES, clf.feature_importances_),
                    key=lambda x: x[1], reverse=True)
        print("\nFeature importances:")
        for name, imp in fi:
            bar = "█" * int(imp * 40)
            print(f"  {name:<25} {imp:.4f}  {bar}")

    return clf, scaler


# ── Save / load ────────────────────────────────────────────────────────────────

def save_song_model(clf, scaler) -> None:
    SONG_MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(SONG_MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f, protocol=5)
    print(f"  Song virality model → {SONG_MODEL_PATH}")


def load_song_model() -> tuple:
    if not SONG_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Song model not found at {SONG_MODEL_PATH}. Run: python train.py"
        )
    with open(SONG_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["clf"], bundle["scaler"]


def model_exists() -> bool:
    return SONG_MODEL_PATH.exists()


# ── Reasoning helper ───────────────────────────────────────────────────────────

def build_reasoning(feat: np.ndarray, prob: float, viral_line: Optional[dict]) -> str:
    """Human-readable explanation of the song-level prediction."""
    pct    = round(prob * 100)
    label  = ("🔥 High TikTok Potential"  if prob >= 0.65 else
               "⚡ Moderate Potential"     if prob >= 0.45 else
               "📉 Low TikTok Potential")

    parts = []
    if feat[0] >= 0.35:
        parts.append(f"**{round(feat[0]*100)}% of lines** are in chorus/hook sections — "
                     "TikTok clips need a repeating hook.")
    if feat[1] >= 0.35:
        parts.append(f"**High repetition** ({round(feat[1]*100)}% of lines repeat) — "
                     "builds earworm stickiness.")
    if feat[7] >= 0.55:
        parts.append(f"**Strong line-level scores** (avg {round(feat[7]*100)}%) — "
                     "multiple lines have hook-worthy quality.")
    if feat[8] >= 0.75:
        parts.append(f"**Peak viral line at {round(feat[8]*100)}%** — "
                     "at least one line is very likely to catch on TikTok.")
    if feat[3] >= 0.15:
        parts.append("**High energy language** — excites and demands attention.")
    if feat[4] >= 0.15:
        parts.append("**Positive emotional tone** — uplifting content drives shares.")
    if feat[5] >= 0.15:
        parts.append("**Emotional depth** — pain/longing lines trigger reaction videos.")

    if not parts:
        if prob >= 0.5:
            parts.append("Moderate structural and lyrical signals — solid hook potential.")
        else:
            parts.append("Limited hook repetition and energy signals — harder to clip.")

    lines = [f"### {label}  —  {pct}% TikTok Viral Score", ""]
    lines += [f"- {p}" for p in parts]

    if viral_line:
        lines += [
            "",
            f"**Best line to clip:**  *\"{viral_line['text']}\"*  "
            f"({round(viral_line['viral_prob']*100)}% viral probability)",
        ]
    return "\n".join(lines)
