"""
Similarity Engine  —  Multi-Model NLP
======================================
Four independent similarity axes combined into one final score:

  1. TF-IDF content      (35%) — bigram cosine similarity on full lyrics
  2. LDA topic model     (20%) — latent topic distribution similarity
  3. Lyric feature vec   (30%) — 11-dimensional hand-crafted feature vector
       vocab richness, rhyme density, hook strength, energy score,
       positive/negative sentiment, repetition rate, chorus density,
       avg line length, verse density, compression ratio
  4. Theme clusters      (15%) — overlap across 7 emotional word clusters

Explanation: tells the user exactly WHICH features matched and why.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# ── Word lists ─────────────────────────────────────────────────────────────────

ENERGY_WORDS = {
    "fire", "lit", "hype", "wild", "crazy", "turnt", "vibe", "bounce",
    "drop", "bang", "banger", "loud", "scream", "jump", "move", "go",
    "run", "fast", "hard", "rage", "beast", "king", "queen", "boss",
    "unstoppable", "flex", "drip", "swag", "pop", "rock", "roll", "hit",
}

POSITIVE_WORDS = {
    "love", "happy", "joy", "smile", "beautiful", "wonderful", "amazing",
    "great", "good", "best", "perfect", "shine", "bright", "warm", "sweet",
    "dream", "hope", "rise", "win", "glory", "bless", "grace", "light",
    "gold", "free", "peace", "together", "forever", "always", "paradise",
}

NEGATIVE_WORDS = {
    "hate", "pain", "cry", "tears", "hurt", "break", "lost", "alone",
    "dark", "cold", "dead", "fall", "fail", "lie", "cheat", "betray",
    "gone", "miss", "sad", "empty", "numb", "sorrow", "grief", "regret",
    "fear", "wrong", "bad", "worst", "hell", "die", "kill", "blood",
}

THEME_CLUSTERS: dict[str, list[str]] = {
    "love & romance":    ["love", "heart", "kiss", "baby", "darling", "forever",
                          "together", "romance", "lover", "babe", "crush", "hold"],
    "heartbreak":        ["break", "cry", "tears", "goodbye", "miss", "lost",
                          "alone", "hurt", "pain", "leave", "gone", "end", "numb"],
    "confidence":        ["boss", "king", "queen", "top", "crown", "flex",
                          "drip", "shine", "win", "best", "power", "rise", "real"],
    "party & energy":    ["party", "dance", "night", "vibe", "fire", "lit",
                          "move", "bounce", "jump", "turn", "beat", "feel", "loud"],
    "nostalgia":         ["remember", "childhood", "memory", "back", "used",
                          "yesterday", "old", "time", "past", "dream", "days"],
    "struggle & grind":  ["grind", "hustle", "work", "fight", "rise", "push",
                          "through", "overcome", "hard", "real", "street", "way"],
    "spiritual":         ["god", "pray", "faith", "bless", "soul", "grace",
                          "heaven", "holy", "spirit", "lord", "light", "amen"],
}

STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "it", "this", "that", "a", "an", "the", "and", "or", "but", "if",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "is", "are", "was", "were", "be", "been", "have", "has", "had",
    "do", "did", "will", "would", "could", "should", "not", "no", "so",
    "yeah", "oh", "ooh", "ah", "na", "la", "da", "hey", "uh", "mm",
    "em", "ya", "gonna", "gotta", "wanna", "cause", "cuz", "got", "get",
}

# ── Text utilities ─────────────────────────────────────────────────────────────

_SECTION_RE = re.compile(r'\[([^\]]+)\]')
_PUNCT      = re.compile(r"[^\w\s]")
_LDA_TOPICS = 15


def _tokenize(text: str) -> list[str]:
    text = _PUNCT.sub(" ", text.lower())
    return [w for w in text.split() if w and w not in STOPWORDS and len(w) > 1]


def _plain_text(lyrics: str) -> str:
    return _SECTION_RE.sub("", lyrics).strip()


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ── Lyric feature extraction ───────────────────────────────────────────────────

def _rhyme_density(lines: list[str]) -> float:
    """Fraction of adjacent line-pairs whose last words share the final 3 chars."""
    if len(lines) < 2:
        return 0.0
    pairs = 0
    rhymes = 0
    for a, b in zip(lines, lines[1:]):
        wa = a.split()[-1].lower() if a.split() else ""
        wb = b.split()[-1].lower() if b.split() else ""
        if len(wa) >= 3 and len(wb) >= 3:
            pairs += 1
            if wa[-3:] == wb[-3:]:
                rhymes += 1
    return round(rhymes / max(pairs, 1), 4)


def extract_lyric_features(lyrics: str) -> dict:
    """
    Return an 11-dim normalized feature dict for one song.
    All values are in [0, 1].
    """
    lines_raw = [l.strip() for l in lyrics.split("\n") if l.strip()]
    section_labels = _SECTION_RE.findall(lyrics)
    sections_norm  = [re.sub(r'\s*\d+$', '', s).strip().lower()
                      for s in section_labels]

    lyric_lines = [l for l in lines_raw if not _SECTION_RE.match(l)]
    tokens      = _tokenize(_plain_text(lyrics))

    total_lines  = max(len(lyric_lines), 1)
    total_tokens = max(len(tokens), 1)
    unique_tok   = len(set(tokens))

    # 1. vocab richness  (type-token ratio)
    vocab_richness = min(unique_tok / total_tokens, 1.0)

    # 2. rhyme density
    rhyme_den = _rhyme_density(lyric_lines)

    # 3. hook strength  (most-repeated single line as fraction of total)
    lc = Counter(l.lower() for l in lyric_lines)
    hook_strength = min(lc.most_common(1)[0][1] / total_lines, 1.0) if lc else 0.0

    # 4. energy score
    token_set = set(tokens)
    energy_score = min(len(token_set & ENERGY_WORDS) / 10, 1.0)

    # 5-6. sentiment
    pos = len(token_set & POSITIVE_WORDS) / max(unique_tok, 1)
    neg = len(token_set & NEGATIVE_WORDS) / max(unique_tok, 1)
    positive_sent = min(pos * 5, 1.0)
    negative_sent = min(neg * 5, 1.0)

    # 7. repetition rate  (lines that repeat)
    repeated   = sum(1 for c in lc.values() if c > 1)
    rep_rate   = repeated / total_lines

    # 8. chorus density
    n_sections  = max(len(sections_norm), 1)
    chorus_cnt  = sum(1 for s in sections_norm
                      if any(k in s for k in ("chorus", "hook", "refrain")))
    chorus_den  = chorus_cnt / n_sections

    # 9. avg line length (normalised: 0=very short, 1=very long)
    avg_len     = np.mean([len(l.split()) for l in lyric_lines]) if lyric_lines else 0
    avg_len_n   = min(avg_len / 20, 1.0)

    # 10. verse density
    verse_cnt   = sum(1 for s in sections_norm if "verse" in s)
    verse_den   = min(verse_cnt / max(n_sections, 1), 1.0)

    # 11. compression ratio  (unique / total lines)
    compression = len(lc) / total_lines

    return {
        "vocab_richness":    round(vocab_richness, 4),
        "rhyme_density":     round(rhyme_den, 4),
        "hook_strength":     round(hook_strength, 4),
        "energy_score":      round(energy_score, 4),
        "positive_sent":     round(positive_sent, 4),
        "negative_sent":     round(negative_sent, 4),
        "repetition_rate":   round(rep_rate, 4),
        "chorus_density":    round(chorus_den, 4),
        "avg_line_length":   round(float(avg_len_n), 4),
        "verse_density":     round(verse_den, 4),
        "compression_ratio": round(compression, 4),
    }


FEATURE_ORDER = [
    "vocab_richness", "rhyme_density", "hook_strength", "energy_score",
    "positive_sent", "negative_sent", "repetition_rate", "chorus_density",
    "avg_line_length", "verse_density", "compression_ratio",
]

FEATURE_LABELS = {
    "vocab_richness":    "Vocabulary Richness",
    "rhyme_density":     "Rhyme Density",
    "hook_strength":     "Hook Strength",
    "energy_score":      "Energy Score",
    "positive_sent":     "Positive Sentiment",
    "negative_sent":     "Negative Sentiment",
    "repetition_rate":   "Repetition Rate",
    "chorus_density":    "Chorus Density",
    "avg_line_length":   "Avg Line Length",
    "verse_density":     "Verse Density",
    "compression_ratio": "Compression Ratio",
}


def _feat_vec(feat_dict: dict) -> np.ndarray:
    return np.array([feat_dict[k] for k in FEATURE_ORDER], dtype=float)


def _feat_cosine(a: dict, b: dict) -> float:
    va = _feat_vec(a).reshape(1, -1)
    vb = _feat_vec(b).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0, 0])


# ── Theme detection ────────────────────────────────────────────────────────────

def _theme_scores(tokens: list[str]) -> dict[str, float]:
    ts = set(tokens)
    return {
        th: round(len(ts & set(kw)) / len(kw), 4)
        for th, kw in THEME_CLUSTERS.items()
    }


def _shared_themes(t1: dict, t2: dict, threshold: float = 0.12) -> list[str]:
    return [th for th in t1 if t1[th] >= threshold and t2[th] >= threshold]


# ── Shared n-gram extraction ───────────────────────────────────────────────────

def _top_shared_ngrams(tok1: list[str], tok2: list[str],
                       n: int = 2, top_k: int = 5) -> list[str]:
    set2 = set(_ngrams(tok2, n))
    cnt  = Counter(g for g in _ngrams(tok1, n) if g in set2)
    return [g for g, _ in cnt.most_common(top_k)]


# ══════════════════════════════════════════════════════════════════════════════
# Main engine
# ══════════════════════════════════════════════════════════════════════════════

class SimilarityEngine:
    """
    Multi-model similarity over a lyric catalog.
    Axes: TF-IDF content | LDA topics | Lyric feature vector | Theme clusters
    """

    WEIGHTS = {
        "tfidf":    0.50,   # most discriminating on small corpus
        "lda":      0.08,   # LDA less useful with < 500 songs
        "features": 0.32,
        "theme":    0.10,
    }

    def __init__(self, df: pd.DataFrame):
        self._df = df[
            df["lyrics"].notna()
            & (df["lyrics"].astype(str).str.strip() != "")
        ].reset_index(drop=True)

        if self._df.empty:
            raise ValueError("Catalog has no lyrics. Run the pipeline first.")

        n = len(self._df)
        print(f"[Engine] Indexing {n} songs ...")

        self._plain = self._df["lyrics"].apply(lambda x: _plain_text(str(x))).tolist()

        # ── TF-IDF ──────────────────────────────────────────────────────────
        self._tfidf_vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.90,
            sublinear_tf=True,
            max_features=20_000,
        )
        self._tfidf_mat = self._tfidf_vec.fit_transform(self._plain)

        # ── LDA topic model ─────────────────────────────────────────────────
        self._count_vec = CountVectorizer(
            ngram_range=(1, 1),
            min_df=2,
            max_df=0.90,
            max_features=8_000,
        )
        count_mat = self._count_vec.fit_transform(self._plain)
        n_topics  = min(_LDA_TOPICS, max(n // 5, 3))
        self._lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method="batch",
            max_iter=20,
        )
        self._lda_mat = self._lda.fit_transform(count_mat)  # (n, topics)
        # Normalize rows to unit sum for cosine
        self._lda_mat_norm = normalize(self._lda_mat, norm="l1")

        # ── Lyric features ───────────────────────────────────────────────────
        self._features = [extract_lyric_features(str(lyr)) for lyr in self._df["lyrics"]]
        self._feat_mat = np.vstack([_feat_vec(f) for f in self._features])

        # ── Themes & tokens ──────────────────────────────────────────────────
        self._tokens = [_tokenize(p) for p in self._plain]
        self._themes = [_theme_scores(t) for t in self._tokens]

        print(f"[Engine] Ready — {n_topics} LDA topics, {len(FEATURE_ORDER)} lyric features.")

    # ── Internal transforms ────────────────────────────────────────────────────

    def _transform_query(self, plain: str):
        q_tfidf = self._tfidf_vec.transform([plain])
        q_count = self._count_vec.transform([plain])
        q_lda   = normalize(self._lda.transform(q_count), norm="l1")
        return q_tfidf, q_lda

    # ── Public API ─────────────────────────────────────────────────────────────

    def find_similar(
        self,
        query_lyrics: str,
        query_artist: str = "",
        query_song:   str = "",
        top_k:        int = 5,
        weights:      Optional[dict] = None,
    ) -> list[dict]:
        """
        Return top_k most similar songs with multi-axis scores and explanations.
        """
        W = weights or self.WEIGHTS
        plain_q   = _plain_text(query_lyrics)
        tokens_q  = _tokenize(plain_q)
        feats_q   = extract_lyric_features(query_lyrics)
        themes_q  = _theme_scores(tokens_q)
        feat_vec_q = _feat_vec(feats_q).reshape(1, -1)

        q_tfidf, q_lda = self._transform_query(plain_q)

        tfidf_scores = cosine_similarity(q_tfidf, self._tfidf_mat)[0]
        lda_scores   = cosine_similarity(q_lda,   self._lda_mat_norm)[0]
        feat_scores  = cosine_similarity(feat_vec_q, self._feat_mat)[0]

        results = []
        for i in range(len(self._df)):
            row = self._df.iloc[i]
            # Skip exact self-match
            if (query_artist and query_song
                    and str(row.get("artist", "")).lower() == query_artist.lower()
                    and str(row.get("song",   "")).lower() == query_song.lower()):
                continue

            shared_th = _shared_themes(themes_q, self._themes[i])
            theme_sc  = min(len(shared_th) / max(len(THEME_CLUSTERS), 1) * 3, 1.0)

            final = (
                W["tfidf"]    * float(tfidf_scores[i])
                + W["lda"]    * float(lda_scores[i])
                + W["features"] * float(feat_scores[i])
                + W["theme"]  * theme_sc
            )

            # Dominant LDA topics (top 2)
            q_topics = q_lda[0]
            c_topics = self._lda_mat_norm[i]
            shared_topic_idx = np.argsort(np.minimum(q_topics, c_topics))[-3:][::-1]
            shared_topics_str = [f"Topic {t+1}" for t in shared_topic_idx
                                 if q_topics[t] > 0.05 and c_topics[t] > 0.05]

            phrases_bi  = _top_shared_ngrams(tokens_q, self._tokens[i], n=2, top_k=4)
            phrases_tri = _top_shared_ngrams(tokens_q, self._tokens[i], n=3, top_k=2)
            shared_phrases = phrases_tri + phrases_bi

            results.append({
                "rank":          0,
                "artist":        str(row.get("artist", "")),
                "song":          str(row.get("song",   "")),
                "tier":          int(row.get("tier",   -1)),
                "source":        str(row.get("source", "")),
                "tfidf_score":   round(float(tfidf_scores[i]), 4),
                "lda_score":     round(float(lda_scores[i]),   4),
                "feature_score": round(float(feat_scores[i]),  4),
                "theme_score":   round(theme_sc,               4),
                "final_score":   round(final,                  4),
                "shared_themes":  shared_th,
                "shared_topics":  shared_topics_str,
                "shared_phrases": shared_phrases,
                "query_feats":    feats_q,
                "match_feats":    self._features[i],
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        top = results[:top_k]

        for rank, r in enumerate(top, 1):
            r["rank"]        = rank
            r["explanation"] = _build_explanation(r)

        return top

    def query_by_title(self, artist: str, song: str, top_k: int = 5) -> list[dict]:
        mask = (
            self._df["artist"].str.lower() == artist.lower()
        ) & (self._df["song"].str.lower() == song.lower())
        hits = self._df[mask]
        if hits.empty:
            raise KeyError(f"'{song}' by {artist} not found in catalog.")
        lyr = str(hits.iloc[0]["lyrics"])
        return self.find_similar(lyr, query_artist=artist, query_song=song, top_k=top_k)

    def get_song_features(self, artist: str, song: str) -> Optional[dict]:
        """Return lyric feature dict for a catalog song."""
        mask = (
            self._df["artist"].str.lower() == artist.lower()
        ) & (self._df["song"].str.lower() == song.lower())
        hits = self._df[mask]
        if hits.empty:
            return None
        idx = hits.index[0]
        return self._features[self._df.index.get_loc(idx)]


# ── Explanation builder ────────────────────────────────────────────────────────

def _build_explanation(r: dict) -> str:
    lines: list[str] = []
    score_pct = round(r["final_score"] * 100)
    lines.append(f"**{score_pct}% overall match** with *{r['song']}* by {r['artist']}")
    lines.append("")

    # TF-IDF content
    tf = r["tfidf_score"]
    if tf >= 0.30:
        lines.append("- **High lyric content overlap**: extensive shared vocabulary and phrase patterns.")
    elif tf >= 0.12:
        lines.append("- **Moderate lyric overlap**: several shared words and recurring phrases.")
    else:
        lines.append("- **Loose word-level similarity**: different vocabulary, but other axes align.")

    # LDA topics
    if r["shared_topics"]:
        lines.append(f"- **Shared lyrical themes** (topic model): {', '.join(r['shared_topics'])}")

    # Shared phrases
    if r["shared_phrases"]:
        phrase_str = ", ".join(f'`{p}`' for p in r["shared_phrases"][:3])
        lines.append(f"- **Recurring phrases/hooks**: {phrase_str}")

    # Thematic clusters
    if r["shared_themes"]:
        lines.append(f"- **Emotional territory**: {', '.join(f'*{t}*' for t in r['shared_themes'])}")

    # Feature vector — highlight the top 3 most-similar features
    qf = r["query_feats"]
    mf = r["match_feats"]
    feat_diffs = {k: abs(qf[k] - mf[k]) for k in FEATURE_ORDER}
    closest = sorted(feat_diffs, key=lambda k: feat_diffs[k])[:3]
    feat_summaries = []
    for k in closest:
        val = round((qf[k] + mf[k]) / 2 * 100)
        feat_summaries.append(f"{FEATURE_LABELS[k]} ({val}%)")
    lines.append(f"- **Structural alignment**: {', '.join(feat_summaries)}")

    # Hook comparison
    qh = qf["hook_strength"]
    mh = mf["hook_strength"]
    if abs(qh - mh) < 0.08:
        level = "strong" if qh > 0.15 else "moderate"
        lines.append(f"- **Hook pattern**: both songs share a {level} repeating hook.")

    # Energy comparison
    qe = qf["energy_score"]
    me = mf["energy_score"]
    if abs(qe - me) < 0.15:
        label = "high" if qe > 0.4 else ("medium" if qe > 0.2 else "low")
        lines.append(f"- **Energy level**: both tracks carry *{label}* energy.")

    # Tier context
    from config import TIER_LABELS, TIER_DESCRIPTIONS
    tier = r["tier"]
    if tier in TIER_LABELS:
        lines.append(f"- **Chart tier**: {TIER_LABELS[tier]} — {TIER_DESCRIPTIONS[tier]}")

    return "\n".join(lines)


# ── Convenience loader ─────────────────────────────────────────────────────────

def load_engine() -> SimilarityEngine:
    from config import CATALOG_CSV
    if not CATALOG_CSV.exists():
        raise FileNotFoundError(
            f"No catalog at {CATALOG_CSV}. Run: python pipeline.py --build"
        )
    df = pd.read_csv(CATALOG_CSV)
    return SimilarityEngine(df)
