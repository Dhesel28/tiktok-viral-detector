"""
Fetch trending TikTok songs via RapidAPI (tiktok-api23).
Uses broad keyword sweep to maximise unique songs, then enriches
each with Genius lyrics and a virality snippet analysis.
"""

from __future__ import annotations

import re
import time
import requests
import pandas as pd
from config import RAPIDAPI_KEY, TIKTOK_API_HOST, TIKTOK_TRENDING_CSV, DATA_DIR

HEADERS = {
    "x-rapidapi-key":  RAPIDAPI_KEY,
    "x-rapidapi-host": TIKTOK_API_HOST,
}
BASE_URL = f"https://{TIKTOK_API_HOST}"

_ORIGINAL_SOUND = re.compile(
    r"original\s*sound|оригинальный\s*звук|suara\s*asli|som\s*original", re.I
)

# Broad keyword sweep — covers different genres and time windows
_SEARCH_KEYWORDS = [
    "trending song 2025",
    "viral song 2025",
    "trending music tiktok",
    "tiktok hit song",
    "popular song 2025",
    "top hits 2025",
    "viral pop song",
    "viral rap song",
    "viral country song",
    "viral rnb song",
    "viral latin song",
    "tiktok dance song",
    "tiktok sad song",
    "tiktok love song",
    "tiktok hype song",
    "viral hook song",
    "new music trending",
    "chart topping 2025",
    "number one song 2025",
    "breakout hit 2025",
]


# ── Raw fetch ──────────────────────────────────────────────────────────────────

def _fetch_raw_videos(target: int = 300) -> list[dict]:
    """Sweep all keywords until we have `target` raw items."""
    all_items: list[dict] = []
    seen_ids: set[str] = set()

    for kw in _SEARCH_KEYWORDS:
        if len(all_items) >= target:
            break
        try:
            resp = requests.get(
                f"{BASE_URL}/api/search/general",
                headers=HEADERS,
                params={"keyword": kw, "count": 20, "cursor": 0},
                timeout=15,
            )
            resp.raise_for_status()
            data  = resp.json()
            items = data.get("item_list", data.get("data", []))
            for item in items:
                vid = item.get("id", item.get("item", {}).get("id", ""))
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    all_items.append(item)
        except Exception as e:
            print(f"  [!] TikTok error for '{kw}': {e}")
        time.sleep(0.35)

    return all_items


# ── Parse ──────────────────────────────────────────────────────────────────────

def _parse_videos(raw_items: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in raw_items:
        try:
            v      = entry.get("item", entry)
            music  = v.get("music", {})
            stats  = v.get("stats", {})
            title  = str(music.get("title",      "")).strip()
            author = str(music.get("authorName", "")).strip()

            if not title or len(title) <= 3:
                continue
            if _ORIGINAL_SOUND.search(title):
                continue
            if len(author) <= 1:
                continue

            rows.append({
                "song":          title,
                "artist":        author,
                "music_id":      str(music.get("id", "")),
                "play_count":    int(stats.get("playCount",   0)),
                "like_count":    int(stats.get("diggCount",   0)),
                "comment_count": int(stats.get("commentCount",0)),
                "share_count":   int(stats.get("shareCount",  0)),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Aggregate by (song, artist)
    agg = (
        df.groupby(["song", "artist"])
        .agg(
            total_plays=("play_count",    "sum"),
            total_likes=("like_count",    "sum"),
            total_comments=("comment_count","sum"),
            total_shares=("share_count",  "sum"),
            video_count=("music_id",       "count"),
            music_id=("music_id",          "first"),
        )
        .reset_index()
        .sort_values("total_plays", ascending=False)
    )
    return agg


# ── Lyrics enrichment ──────────────────────────────────────────────────────────

def _enrich_lyrics(df: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    """Fetch Genius lyrics for each TikTok trending song."""
    try:
        from genius_fetcher import fetch_lyrics, _make_client
        genius = _make_client()
    except Exception as e:
        print(f"  [!] Could not init Genius client: {e}")
        df["lyrics"] = None
        return df

    if "lyrics" not in df.columns:
        df["lyrics"] = None

    total = len(df)
    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics","")).strip():
            continue
        artist = str(row["artist"]).strip()
        song   = str(row["song"]).strip()
        print(f"  [Genius] [{idx+1}/{total}] {song} — {artist}")
        lyr = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyr or ""
        time.sleep(delay)

    found = (df["lyrics"].notna() & (df["lyrics"].astype(str).str.strip() != "")).sum()
    print(f"  [Genius] Lyrics fetched for {found}/{total} TikTok songs")
    return df


# ── Virality + snippet enrichment ─────────────────────────────────────────────

def _enrich_virality(df: pd.DataFrame) -> pd.DataFrame:
    """Add virality score and best snippet for each song with lyrics."""
    try:
        from snippet_detector import virality_analysis
    except Exception:
        df["virality_score"]  = None
        df["virality_label"]  = None
        df["best_snippet"]    = None
        df["hook_line"]       = None
        return df

    scores, labels, snippets, hooks = [], [], [], []
    for _, row in df.iterrows():
        lyr = str(row.get("lyrics","")).strip()
        if lyr:
            try:
                va = virality_analysis(lyr)
                scores.append(va["overall_virality"])
                labels.append(va["virality_label"])
                snippets.append("\n".join(va["snippet"]["lines"][:4]))
                hooks.append(va["snippet"]["hook_line"])
            except Exception:
                scores.append(None); labels.append(None)
                snippets.append(None); hooks.append(None)
        else:
            scores.append(None); labels.append(None)
            snippets.append(None); hooks.append(None)

    df["virality_score"] = scores
    df["virality_label"] = labels
    df["best_snippet"]   = snippets
    df["hook_line"]      = hooks
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def get_trending_songs(count: int = 300, fetch_lyrics: bool = True) -> pd.DataFrame:
    """
    Fetch trending TikTok songs, enrich with lyrics and virality analysis.
    Saves to data/tiktok_trending.csv.
    """
    print(f"[TikTok] Fetching up to {count} raw trending videos ...")
    raw = _fetch_raw_videos(target=count)
    print(f"  Raw items collected: {len(raw)}")

    df = _parse_videos(raw)
    if df.empty:
        print("  [!] No valid songs — check RapidAPI key/quota")
        return df

    df["source"] = "tiktok_trending"
    df["tier"]   = -1

    # Carry over existing lyrics to avoid re-fetching
    if TIKTOK_TRENDING_CSV.exists():
        existing = pd.read_csv(TIKTOK_TRENDING_CSV)
        if "lyrics" in existing.columns:
            lyr_map = existing.dropna(subset=["lyrics"]).set_index(
                ["song", "artist"]
            )["lyrics"].to_dict()
            df["lyrics"] = df.apply(
                lambda r: lyr_map.get((r["song"], r["artist"])), axis=1
            )
        else:
            df["lyrics"] = None
    else:
        df["lyrics"] = None

    if fetch_lyrics:
        print(f"[TikTok] Fetching lyrics for {df['lyrics'].isna().sum()} songs ...")
        df = _enrich_lyrics(df)

    print("[TikTok] Running virality analysis ...")
    df = _enrich_virality(df)

    df.to_csv(TIKTOK_TRENDING_CSV, index=False)
    print(f"[TikTok] Saved {len(df)} trending songs → {TIKTOK_TRENDING_CSV}")
    return df


if __name__ == "__main__":
    df = get_trending_songs()
    print(df[["song","artist","total_plays","virality_label"]].head(20).to_string(index=False))
