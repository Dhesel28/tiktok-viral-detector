"""
Trending Finder
===============
Given a song's lyrics, extract its top NLP keywords and search
TikTok in real-time for currently trending songs that match.

Returns ranked results with play counts — these are songs that
are ACTUALLY trending right now and share lyrical DNA with the input.
"""

from __future__ import annotations

import re
import time
import requests
from collections import Counter
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import RAPIDAPI_KEY, TIKTOK_API_HOST

HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": TIKTOK_API_HOST,
}
BASE_URL = f"https://{TIKTOK_API_HOST}"

_SECTION_RE = re.compile(r'\[([^\]]+)\]')
_PUNCT      = re.compile(r"[^\w\s]")
_STOPWORDS  = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "it", "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "is", "are", "was", "be", "have", "do", "will", "not",
    "yeah", "oh", "ooh", "na", "la", "hey", "uh", "gonna", "gotta", "wanna",
}
_ORIGINAL_SOUND = re.compile(r"original\s*sound", re.I)


# ── Keyword extraction ─────────────────────────────────────────────────────────

def extract_keywords(lyrics: str, top_n: int = 8) -> list[str]:
    """
    Use TF-IDF on the lyrics to extract the most distinctive keywords.
    Falls back to raw frequency count if TF-IDF gives no results.
    """
    plain = _SECTION_RE.sub("", lyrics)
    plain = _PUNCT.sub(" ", plain.lower())
    tokens = [w for w in plain.split() if w not in _STOPWORDS and len(w) > 2]

    if not tokens:
        return []

    # TF-IDF on single document (use sublinear_tf for better discrimination)
    try:
        vec = TfidfVectorizer(ngram_range=(1, 1), max_features=200, sublinear_tf=True)
        vec.fit([" ".join(tokens)])
        scores = dict(zip(vec.get_feature_names_out(), vec.idf_))
        # Sort by IDF inverse (lower IDF = less common globally = more distinctive)
        sorted_words = sorted(scores, key=lambda w: scores[w])
        # Use the highest-frequency distinctive words
        freq = Counter(tokens)
        combined = sorted(
            (w for w in freq if w in scores),
            key=lambda w: freq[w] * (1 / max(scores[w], 0.1)),
            reverse=True,
        )
        return combined[:top_n]
    except Exception:
        # Fallback: raw frequency
        return [w for w, _ in Counter(tokens).most_common(top_n)]


def _build_search_queries(keywords: list[str], artist: str = "", song: str = "") -> list[str]:
    """Build TikTok search queries from keywords + optional artist/song."""
    queries = []
    if song and artist:
        queries.append(f"songs like {song}")
        queries.append(f"similar to {artist}")
    # Keyword combos
    if len(keywords) >= 2:
        queries.append(f"{keywords[0]} {keywords[1]} song")
    if len(keywords) >= 4:
        queries.append(f"{keywords[2]} {keywords[3]} trending song")
    # Individual strong keywords
    queries.extend([f"{kw} song trending" for kw in keywords[:3]])
    return queries[:6]


# ── TikTok search ──────────────────────────────────────────────────────────────

def _search_tiktok(query: str, count: int = 15) -> list[dict]:
    try:
        resp = requests.get(
            f"{BASE_URL}/api/search/general",
            headers=HEADERS,
            params={"keyword": query, "count": count, "cursor": 0},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("item_list", data.get("data", []))
    except Exception:
        return []


def _parse_item(item: dict) -> Optional[dict]:
    try:
        v      = item.get("item", item)
        music  = v.get("music", {})
        stats  = v.get("stats", {})
        title  = str(music.get("title",      "")).strip()
        author = str(music.get("authorName", "")).strip()

        if not title or len(title) <= 3:
            return None
        if _ORIGINAL_SOUND.search(title):
            return None
        if len(author) <= 1:
            return None

        return {
            "song":        title,
            "artist":      author,
            "play_count":  int(stats.get("playCount",  0)),
            "like_count":  int(stats.get("diggCount",  0)),
            "share_count": int(stats.get("shareCount", 0)),
            "music_id":    str(music.get("id", "")),
        }
    except Exception:
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def find_trending_similar(
    query_lyrics: str,
    query_artist: str = "",
    query_song:   str = "",
    top_k:        int = 10,
) -> list[dict]:
    """
    Search TikTok in real-time for trending songs similar to the input lyrics.

    Steps:
      1. Extract top NLP keywords from lyrics
      2. Build TikTok search queries
      3. Fetch + deduplicate results
      4. Rank by play count
      5. Annotate with matched keywords

    Returns a list of dicts with: song, artist, play_count, like_count,
    keyword_matches, trend_score
    """
    if not RAPIDAPI_KEY:
        return []

    keywords = extract_keywords(query_lyrics, top_n=8)
    queries  = _build_search_queries(keywords, query_artist, query_song)

    print(f"[TrendingFinder] Keywords: {keywords[:5]}")
    print(f"[TrendingFinder] Queries:  {queries[:3]}")

    all_items: list[dict] = []
    seen_ids:  set[str]   = set()

    for q in queries:
        raw = _search_tiktok(q, count=20)
        for item in raw:
            parsed = _parse_item(item)
            if parsed and parsed["music_id"] not in seen_ids:
                seen_ids.add(parsed["music_id"])
                all_items.append(parsed)
        time.sleep(0.3)

    if not all_items:
        return []

    df = pd.DataFrame(all_items)

    # Filter out the query song itself
    if query_song:
        df = df[df["song"].str.lower() != query_song.lower()]

    # Aggregate duplicates (same song+artist from different searches)
    df = (
        df.groupby(["song", "artist"])
        .agg(
            play_count=("play_count",  "sum"),
            like_count=("like_count",  "sum"),
            share_count=("share_count","sum"),
            music_id=("music_id",       "first"),
        )
        .reset_index()
    )

    # Keyword overlap score (how many query keywords appear in the song title or artist)
    kw_set = set(k.lower() for k in keywords)
    def kw_match(row):
        combined = (row["song"] + " " + row["artist"]).lower()
        return sum(1 for k in kw_set if k in combined)

    df["keyword_matches"] = df.apply(kw_match, axis=1)

    # Trend score: log(play_count) normalised + keyword bonus
    import numpy as np
    log_plays = np.log1p(df["play_count"])
    max_log   = log_plays.max() if log_plays.max() > 0 else 1.0
    df["trend_score"] = (
        0.80 * (log_plays / max_log)
        + 0.20 * (df["keyword_matches"] / max(len(kw_set), 1))
    ).round(4)

    df = df.sort_values("trend_score", ascending=False).head(top_k)
    df["keywords_used"] = ", ".join(keywords[:5])

    # ── Why match + matched keyword tags ─────────────────────────────────────
    def _build_why(row: pd.Series) -> str:
        combined     = (row["song"] + " " + row["artist"]).lower()
        matched_kws  = [k for k in keywords if k in combined]
        plays        = int(row.get("play_count", 0))
        parts: list[str] = []

        if matched_kws:
            parts.append("Shares keywords: " + ", ".join(matched_kws[:3]))
        else:
            parts.append("Found via lyric keyword search: " + ", ".join(keywords[:3]))

        if plays >= 10_000_000:
            parts.append(f"{plays // 1_000_000}M+ plays")
        elif plays >= 1_000_000:
            parts.append(f"{plays // 1_000_000}M plays")
        elif plays >= 100_000:
            parts.append(f"{plays // 1_000}K plays")

        return "  ·  ".join(parts)

    def _matched_kws(row: pd.Series) -> list:
        combined = (row["song"] + " " + row["artist"]).lower()
        return [k for k in keywords if k in combined]

    df["why_match"]        = df.apply(_build_why, axis=1)
    df["matched_keywords"] = df.apply(_matched_kws, axis=1)

    return df.to_dict(orient="records")
