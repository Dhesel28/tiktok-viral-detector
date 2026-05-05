"""
NLP Model Inference
===================
Two classes that load the trained models and run inference.

ViralLineDetector
  .predict(lyrics) -> dict
     - every line scored with viral probability
     - the single highest-probability "viral line"
     - a 5-line TikTok clip centred on that line
     - human-readable reasoning

SongSimilarityFinder
  .find_similar(lyrics, top_k=5) -> list[dict]
     - top-k catalog songs by cosine similarity
     - explanation: shared vocabulary, structural notes
"""

from __future__ import annotations

import re
import pickle
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

_HERE = Path(__file__).parent
VIRAL_MODEL_PATH = _HERE / "models" / "viral_line_model.pkl"
SIM_MODEL_PATH   = _HERE / "models" / "similarity_model.pkl"

_SECTION_RE = re.compile(r'\[([^\]]+)\]', re.IGNORECASE)

# ── shared feature constants (mirrors train.py) ────────────────────────────────
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


# ── text helpers ───────────────────────────────────────────────────────────────

def _plain_text(lyrics: str) -> str:
    return _SECTION_RE.sub("", lyrics).strip()

def _last3(line: str) -> str:
    words = line.strip().split()
    return words[-1].lower()[-3:] if words else ""

def _rhymes(a: str, b: str) -> bool:
    ea, eb = _last3(a), _last3(b)
    return len(ea) >= 2 and ea == eb

def _word_set(line: str) -> set:
    return set(re.sub(r"[^\w\s]", " ", line.lower()).split())


# ── section header detection ──────────────────────────────────────────────────

_SECTION_KEYWORDS = frozenset({
    "verse", "chorus", "hook", "bridge", "intro", "outro", "refrain",
    "pre-chorus", "prechorus", "interlude", "break", "spoken", "drop",
    "coda", "tag", "transition", "pre chorus",
})

def _has_section_keyword(text: str) -> bool:
    """True if text contains a known section keyword."""
    t = text.lower().replace("-", " ").replace("_", " ")
    return any(kw.replace("-", " ") in t for kw in _SECTION_KEYWORDS)


def _detect_section_header(text: str) -> Optional[str]:
    """
    Detect non-bracketed section headers like:
      (Chorus), VERSE 1, Chorus:, PRE-CHORUS, BRIDGE, Verse 2
    Returns canonical lowercased label or None if it's a regular lyric line.
    """
    s = text.strip()

    # Parentheses: (Chorus), (Verse 1)
    m = re.match(r'^\(([^)]{1,40})\)\s*$', s)
    if m:
        inner = m.group(1).strip()
        if _has_section_keyword(inner):
            return re.sub(r'\s*\d+$', '', inner.lower()).strip()

    # Trailing colon: Verse 1:, Chorus:, PRE-CHORUS:
    m = re.match(r'^(.{1,40}?)\s*:\s*$', s)
    if m:
        inner = m.group(1).strip()
        if 1 <= len(inner.split()) <= 4 and _has_section_keyword(inner):
            return re.sub(r'\s*\d+$', '', inner.lower()).strip()

    # Plain bare header: "CHORUS", "Verse 2", "Pre-Chorus", "BRIDGE"
    if 1 <= len(s.split()) <= 4 and _has_section_keyword(s):
        return re.sub(r'\s*\d+$', '', s.lower()).strip()

    return None


# ── feature extraction (same as train.py) ─────────────────────────────────────

def _extract_line_features(
    text: str,
    prev_text: str,
    next_text: str,
    repetition_score: float,
    is_repeated: int,
    position_ratio: float,
) -> list[float]:
    ws = _word_set(text)
    wc = max(len(text.split()), 1)

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

    first_word   = text.split()[0].lower().strip(".,!?") if text.split() else ""
    strong_start = 1.0 if first_word in ENERGY_WORDS | POSITIVE_WORDS | NEGATIVE_WORDS else 0.0

    return [
        float(repetition_score),
        float(energy),
        1.0 if _rhymes(text, prev_text) else 0.0,
        1.0 if _rhymes(text, next_text) else 0.0,
        float(brevity),
        float(pos_sent),
        float(neg_sent),
        float(position_ratio),
        min(wc / 20.0, 1.0),
        float(is_repeated),
        float(strong_start),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Viral Line Detector
# ══════════════════════════════════════════════════════════════════════════════

class ViralLineDetector:
    """
    Load trained GradientBoostingClassifier and score every line
    in a song for TikTok viral probability.
    """

    _instance: Optional["ViralLineDetector"] = None

    def __init__(self):
        if not VIRAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Viral line model not found at {VIRAL_MODEL_PATH}. "
                "Run: python train.py"
            )
        with open(VIRAL_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        self._clf    = bundle["clf"]
        self._scaler = bundle["scaler"]

    @classmethod
    def get(cls) -> "ViralLineDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── internal ──────────────────────────────────────────────────────────────

    def _parse_song(self, lyrics: str) -> list[dict]:
        """Return list of dicts: section, text, for every non-header line."""
        current_sec = "intro"
        entries: list[dict] = []
        for raw in lyrics.split("\n"):
            s = raw.strip()
            if not s:
                continue
            m = _SECTION_RE.match(s)
            if m:
                sec_label   = m.group(1).lower().strip()
                current_sec = re.sub(r'\s*\d+$', '', sec_label).strip()
            else:
                # Detect non-bracketed headers: CHORUS, Verse 1:, (Bridge), etc.
                alt = _detect_section_header(s)
                if alt is not None:
                    current_sec = alt
                else:
                    entries.append({"section": current_sec, "text": s})
        return entries

    def _build_feature_matrix(self, entries: list[dict]) -> np.ndarray:
        texts       = [e["text"] for e in entries]
        total       = len(texts)
        line_counts = Counter(t.lower() for t in texts)
        max_count   = max(line_counts.values(), default=1)
        rows = []
        for i, text in enumerate(texts):
            cnt       = line_counts[text.lower()]
            rep_score = (cnt - 1) / max(max_count - 1, 1) if max_count > 1 else 0.0
            feat = _extract_line_features(
                text             = text,
                prev_text        = texts[i - 1] if i > 0 else "",
                next_text        = texts[i + 1] if i < total - 1 else "",
                repetition_score = rep_score,
                is_repeated      = 1 if cnt >= 2 else 0,
                position_ratio   = i / max(total - 1, 1),
            )
            rows.append(feat)
        return np.array(rows, dtype=np.float32)

    # ── public ────────────────────────────────────────────────────────────────

    def predict(self, lyrics: str, clip_size: int = 5) -> dict:
        """
        Score every line and return:
          all_lines   — list of {text, section, viral_prob, label, why, repeats}
          viral_line  — highest-probability line
          runner_up   — second-highest
          viral_clip  — clip_size lines centred on viral_line
          reasoning   — markdown explanation string
        """
        entries = self._parse_song(lyrics)
        if not entries:
            return self._empty()

        X_raw   = self._build_feature_matrix(entries)
        X_sc    = self._scaler.transform(X_raw)
        probs   = self._clf.predict_proba(X_sc)[:, 1]   # P(viral)

        texts       = [e["text"] for e in entries]
        line_counts = Counter(t.lower() for t in texts)

        scored: list[dict] = []
        for i, (entry, prob) in enumerate(zip(entries, probs)):
            cnt = line_counts[entry["text"].lower()]
            scored.append({
                "idx":        i,
                "section":    entry["section"],
                "text":       entry["text"],
                "viral_prob": round(float(prob), 4),
                "repeats":    cnt,
                "label":      self._label(prob),
                "why":        self._why(X_raw[i], cnt, prob),
            })

        sorted_lines = sorted(scored, key=lambda x: x["viral_prob"], reverse=True)
        viral_line   = sorted_lines[0]
        runner_up    = sorted_lines[1] if len(sorted_lines) > 1 else None

        # Clip: clip_size lines centred on viral line
        half  = clip_size // 2
        start = max(0, viral_line["idx"] - half)
        end   = min(len(scored), start + clip_size)
        start = max(0, end - clip_size)
        clip  = scored[start:end]

        return {
            "all_lines":  scored,
            "viral_line": viral_line,
            "runner_up":  runner_up,
            "viral_clip": clip,
            "reasoning":  self._reasoning(viral_line, runner_up, X_raw[viral_line["idx"]]),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _label(prob: float) -> str:
        if prob >= 0.75:   return "🔥 Viral"
        if prob >= 0.55:   return "⚡ Strong Hook"
        if prob >= 0.35:   return "✓ Solid"
        return "–"

    @staticmethod
    def _why(feat: np.ndarray, repeats: int, prob: float) -> str:
        parts = []
        if feat[0] >= 0.4:    parts.append(f"repeats {repeats}×")
        if feat[1] >= 0.25:   parts.append("high energy")
        if feat[2] or feat[3]: parts.append("rhymes")
        if feat[4] >= 0.8:    parts.append("perfect length")
        if feat[5] >= 0.25:   parts.append("uplifting")
        if feat[6] >= 0.25:   parts.append("emotional")
        if prob >= 0.75:      parts.append("in chorus/hook section")
        return ", ".join(parts) if parts else "moderate signals"

    @staticmethod
    def _reasoning(viral: dict, runner_up: Optional[dict], feat: np.ndarray) -> str:
        pct = round(viral["viral_prob"] * 100)
        lines = [
            f"### 🔥 Viral Probability: {pct}%",
            f"> *\"{viral['text']}\"*",
            f"**Section:** [{viral['section'].title()}]  |  "
            f"**Repeats** {viral['repeats']}× in song",
            "",
            "**Why the model flagged this line:**",
        ]

        if feat[0] >= 0.4:
            lines.append(f"- **High repetition** ({viral['repeats']}×) — "
                         "the brain locks onto lines it hears repeatedly.")
        if feat[1] >= 0.25:
            lines.append("- **Energy language** — words that excite, energise, and demand attention.")
        if feat[2] or feat[3]:
            lines.append("- **End-rhyme** with adjacent line — natural flow, "
                         "satisfying for lip-sync and duets.")
        if feat[4] >= 0.8:
            wc = int(feat[8] * 20)
            lines.append(f"- **Ideal length** (~{wc} words) — "
                         "short enough for TikTok text overlays.")
        if feat[5] >= 0.25:
            lines.append("- **Positive emotion** — uplifting lines invite shares and duets.")
        if feat[6] >= 0.25:
            lines.append("- **Emotional punch** — strong feeling triggers reaction videos.")
        if pct >= 75:
            lines.append("- **Lives in a Chorus/Hook/Bridge** — "
                         "the highest-replayed section of any song.")

        if runner_up:
            lines.append(
                f"\n**Runner-up** ({round(runner_up['viral_prob']*100)}%): "
                f"*\"{runner_up['text']}\"* — [{runner_up['section'].title()}]"
            )

        lines.append(
            "\n**TikTok Clip Tip:** Use the 5-line clip shown below. "
            "Start filming at the beginning of that section for maximum impact."
        )
        return "\n".join(lines)

    @staticmethod
    def _empty() -> dict:
        return {
            "all_lines": [], "viral_line": None, "runner_up": None,
            "viral_clip": [], "reasoning": "No lyric lines found.",
        }


# ══════════════════════════════════════════════════════════════════════════════
# Song Similarity Finder
# ══════════════════════════════════════════════════════════════════════════════

class SongSimilarityFinder:
    """
    TF-IDF + cosine NearestNeighbors similarity over the lyrics catalog.
    Returns top-k songs with vocabulary overlap and structural explanation.
    """

    _instance: Optional["SongSimilarityFinder"] = None

    def __init__(self):
        if not SIM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Similarity model not found at {SIM_MODEL_PATH}. "
                "Run: python train.py"
            )
        with open(SIM_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        self._vec        = bundle["vectorizer"]
        self._nn         = bundle["nn_model"]
        self._catalog    = bundle["catalog_df"]
        self._tfidf_mat  = bundle["tfidf_matrix"]
        self._feat_names = self._vec.get_feature_names_out()
        self._token_sets = bundle.get("token_sets", [set()] * len(self._catalog))

    @classmethod
    def get(cls) -> "SongSimilarityFinder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── public ────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        query_lyrics: str,
        query_artist: str = "",
        query_song:   str = "",
        top_k:        int = 5,
    ) -> list[dict]:
        """
        Return top_k most similar catalog songs to the query lyrics.

        Each result dict contains:
          rank, song, artist, similarity (0-1), shared_vocab,
          explanation (markdown string)
        """
        plain_q = _plain_text(query_lyrics)
        q_vec   = self._vec.transform([plain_q])

        # k+5 to handle possible self-match filtering
        n_query = min(top_k + 10, len(self._catalog))
        distances, indices = self._nn.kneighbors(q_vec, n_neighbors=n_query)

        # cosine distance → similarity
        similarities = 1.0 - distances[0]
        q_top_terms  = self._top_terms_for_vector(q_vec)

        # Raw token set for shared-vocab display
        _STOP = {
            "i","me","my","we","our","you","your","he","she","they","it","a","an","the",
            "and","or","but","in","on","at","to","for","of","with","by","from","up",
            "is","are","was","be","have","has","do","did","will","not","so","this","that",
            "yeah","oh","ooh","na","la","da","hey","uh","mm","ya","gonna","gotta","wanna",
        }
        q_tokens = {w for w in re.sub(r"[^\w\s]", " ", plain_q.lower()).split()
                    if len(w) > 2 and w not in _STOP}

        results = []
        for dist, idx in zip(similarities, indices[0]):
            row = self._catalog.iloc[idx]
            # Skip exact self-match
            if (query_artist and query_song
                    and str(row.get("artist","")).lower() == query_artist.lower()
                    and str(row.get("song","")).lower()   == query_song.lower()):
                continue

            # Shared vocab: raw token overlap (more reliable than TF-IDF terms)
            cat_tokens = self._token_sets[idx] if idx < len(self._token_sets) else set()
            shared     = sorted(q_tokens & cat_tokens,
                                key=lambda w: len(w), reverse=True)[:8]

            cat_top_terms = row.get("top_terms", [])
            if isinstance(cat_top_terms, str):
                import ast
                try:
                    cat_top_terms = ast.literal_eval(cat_top_terms)
                except Exception:
                    cat_top_terms = []

            results.append({
                "rank":        len(results) + 1,
                "song":        str(row.get("song",   "")),
                "artist":      str(row.get("artist", "")),
                "similarity":  round(float(dist), 4),
                "shared_vocab": shared,
                "explanation": self._explanation(
                    row, float(dist), shared, q_top_terms, cat_top_terms
                ),
            })

            if len(results) >= top_k:
                break

        return results

    # ── helpers ───────────────────────────────────────────────────────────────

    def _top_terms_for_vector(self, vec, top_n: int = 15) -> list[str]:
        """Return top unigrams only — bigrams are too song-specific for explanation."""
        arr   = vec.toarray().flatten()
        # Sort all terms by weight, keep unigrams only
        order = np.argsort(arr)[::-1]
        terms = []
        for i in order:
            if arr[i] <= 0:
                break
            term = self._feat_names[i]
            if " " not in term and len(term) > 2:   # unigrams, skip very short words
                terms.append(term)
            if len(terms) >= top_n:
                break
        return terms

    @staticmethod
    def _explanation(
        row: pd.Series,
        sim: float,
        shared: list[str],
        q_terms: list[str],
        c_terms: list[str],
    ) -> str:
        pct = round(sim * 100)
        lines = [f"**{pct}% lyric similarity** with *{row.get('song','')}* "
                 f"by {row.get('artist','')}"]
        lines.append("")

        if sim >= 0.35:
            lines.append("- **Strong vocabulary overlap**: extensive shared word patterns.")
        elif sim >= 0.15:
            lines.append("- **Moderate vocabulary overlap**: several recurring words and phrases in common.")
        else:
            lines.append("- **Structural match**: songs share writing style more than specific words.")

        if shared:
            phrase_str = ", ".join(f"`{t}`" for t in shared[:6])
            lines.append(f"- **Shared lyric vocabulary**: {phrase_str}")

        # Unique to each
        q_unique = [t for t in q_terms[:6] if t not in c_terms]
        c_unique = [t for t in c_terms[:6] if t not in q_terms]
        if q_unique:
            lines.append(f"- **Your signature words**: {', '.join(q_unique[:4])}")
        if c_unique:
            lines.append(f"- **Their signature words**: {', '.join(c_unique[:4])}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Song-Level Virality Predictor  (Model 3)
# ══════════════════════════════════════════════════════════════════════════════

class SongViralityPredictor:
    """
    Predicts whether a full song will go viral on TikTok.
    Depends on a trained ViralLineDetector to compute line-level features first.

    Usage:
        detector = ViralLineDetector()
        predictor = SongViralityPredictor()
        result = predictor.predict(lyrics, detector)
        # result keys: viral_prob, label, reasoning, feat
    """

    _instance: Optional["SongViralityPredictor"] = None

    def __init__(self):
        from song_viral_model import load_song_model, SONG_MODEL_PATH
        if not SONG_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Song virality model not found at {SONG_MODEL_PATH}. "
                "Run: python train.py"
            )
        self._clf, self._scaler = load_song_model()

    @classmethod
    def get(cls) -> "SongViralityPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(self, lyrics: str, line_result: Optional[dict] = None) -> dict:
        """
        Predict song-level TikTok virality.

        Parameters
        ----------
        lyrics      : raw lyrics text
        line_result : optional pre-computed ViralLineDetector.predict() output.
                      If None, a ViralLineDetector will be instantiated internally.

        Returns
        -------
        dict with keys:
          viral_prob  float [0,1]
          label       str  "🔥 High Potential" / "⚡ Moderate" / "📉 Low"
          reasoning   str  markdown explanation
          feat        np.ndarray  raw feature vector
        """
        if line_result is None:
            detector    = ViralLineDetector.get()
            line_result = detector.predict(lyrics)

        from song_viral_model import extract_song_features, build_reasoning
        feat    = extract_song_features(line_result["all_lines"])
        feat_sc = self._scaler.transform(feat.reshape(1, -1))
        prob    = float(self._clf.predict_proba(feat_sc)[0, 1])

        label = ("🔥 High TikTok Potential"  if prob >= 0.65 else
                 "⚡ Moderate Potential"     if prob >= 0.45 else
                 "📉 Low TikTok Potential")

        return {
            "viral_prob": round(prob, 4),
            "label":      label,
            "reasoning":  build_reasoning(feat, prob, line_result.get("viral_line")),
            "feat":       feat,
        }
