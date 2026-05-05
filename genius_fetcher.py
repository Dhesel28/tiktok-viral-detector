"""
Fetch lyrics from Genius API for a list of (artist, song) pairs.
Adds a 'lyrics' column to the input DataFrame and saves to catalog_with_lyrics.csv.
"""

from __future__ import annotations

import re
import time
import lyricsgenius
import pandas as pd
from typing import Optional
from config import GENIUS_TOKEN, CATALOG_CSV


# ── Genius client ─────────────────────────────────────────────────────────────

def _make_client() -> lyricsgenius.Genius:
    if not GENIUS_TOKEN:
        raise ValueError("GENIUS_ACCESS_TOKEN not set — check your .env file")
    genius = lyricsgenius.Genius(
        GENIUS_TOKEN,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Acoustic)"],
        verbose=False,
        timeout=10,
    )
    genius.remove_section_headers = False  # keep [Chorus], [Verse] labels
    return genius


# ── Lyric cleaning ─────────────────────────────────────────────────────────────

_HEADER_NOISE = re.compile(
    r"^.*?(?=\[)", re.DOTALL
)  # strip everything before the first [Section]

_EMBED_TRAILER = re.compile(
    r"\d+\s*Embed\s*$", re.MULTILINE
)

_YOU_MIGHT = re.compile(
    r"You might also like.*", re.DOTALL | re.IGNORECASE
)


def _clean_lyrics(raw: str) -> str:
    """Remove Genius page junk from raw lyrics text."""
    text = _YOU_MIGHT.sub("", raw)
    text = _EMBED_TRAILER.sub("", text)
    # Strip leading header before first [Section] label
    match = re.search(r"\[", text)
    if match:
        text = text[match.start():]
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_lyrics(artist: str, song: str, genius: Optional[lyricsgenius.Genius] = None) -> Optional[str]:
    """Return cleaned lyrics string, or None if not found."""
    if genius is None:
        genius = _make_client()
    try:
        result = genius.search_song(song, artist)
        if result and result.lyrics:
            return _clean_lyrics(result.lyrics)
    except Exception as e:
        print(f"    [Genius] Error for '{song}' by {artist}: {e}")
    return None


def enrich_with_lyrics(df: pd.DataFrame, delay: float = 0.5) -> pd.DataFrame:
    """
    Add a 'lyrics' column to df (must have 'artist' and 'song' columns).
    Skips rows that already have non-empty lyrics.
    Saves incrementally to CATALOG_CSV.
    """
    genius = _make_client()

    if "lyrics" not in df.columns:
        df["lyrics"] = None

    total = len(df)
    for idx, row in df.iterrows():
        if pd.notna(row.get("lyrics")) and str(row.get("lyrics", "")).strip():
            continue  # already fetched

        artist = str(row["artist"]).strip()
        song = str(row["song"]).strip()
        print(f"  [{idx+1}/{total}] Fetching: {song} — {artist}")

        lyrics = fetch_lyrics(artist, song, genius)
        df.at[idx, "lyrics"] = lyrics if lyrics else ""

        time.sleep(delay)

        # Save progress every 10 songs
        if (idx + 1) % 10 == 0:
            df.to_csv(CATALOG_CSV, index=False)
            print(f"    → Progress saved ({idx+1}/{total})")

    df.to_csv(CATALOG_CSV, index=False)
    fetched = df["lyrics"].notna() & (df["lyrics"] != "")
    print(f"[Genius] Done. Lyrics found for {fetched.sum()}/{total} songs.")
    return df


if __name__ == "__main__":
    from dataset import load_balanced_tiers
    df = load_balanced_tiers()
    df = enrich_with_lyrics(df)
    print(df[["artist", "song", "tier"]].head())
