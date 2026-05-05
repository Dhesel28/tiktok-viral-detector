"""
Tatiana Song Analyzer
=====================
Loads Tatiana_cyanite.csv, fetches lyrics via Genius,
runs the ViralLineDetector on every song, and saves
results to data/tatiana_results.csv.
"""

from __future__ import annotations

import re
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Optional, Callable

_HERE = Path(__file__).parent
_ROOT = _HERE.parent                         # SUPA_Spring root

TATIANA_CSV  = _ROOT / "Tatiana_cyanite.csv"
RESULTS_CSV  = _HERE / "data" / "tatiana_results.csv"


# ── Song title cleaning ────────────────────────────────────────────────────────

def _clean_title(title: str) -> str:
    """Strip surrounding quotes and junk suffixes from a raw song title."""
    t = str(title).strip().strip("'\"")
    # Remove "Official Audio / Video / Music Video" suffixes
    t = re.sub(
        r"\s+('?)(?:Official\s+)?(?:Audio|Video|Lyric\s+Video|Music\s+Video)('?)",
        "", t, flags=re.I,
    ).strip().strip("'\"").strip()
    return t


# ── Load source data ───────────────────────────────────────────────────────────

def load_tatiana() -> pd.DataFrame:
    df = pd.read_csv(TATIANA_CSV)
    df["song_clean"]   = df["song_title"].apply(_clean_title)
    df["artist_clean"] = df["artist"].apply(lambda x: str(x).strip())
    return df


# ── Lyrics fetching ────────────────────────────────────────────────────────────

def fetch_all_lyrics(
    df: pd.DataFrame,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Fetch Genius lyrics for every row. Calls progress_cb(i, total, song_name)."""
    sys.path.insert(0, str(_HERE))
    from genius_fetcher import fetch_lyrics, _make_client

    genius      = _make_client()
    lyrics_list = []
    total       = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_cb:
            progress_cb(i, total, row["song_clean"])
        lyr = fetch_lyrics(row["artist_clean"], row["song_clean"], genius)
        lyrics_list.append(lyr or "")
        time.sleep(0.6)

    df = df.copy()
    df["lyrics"] = lyrics_list
    return df


# ── Viral detection ────────────────────────────────────────────────────────────

def run_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Run ViralLineDetector on every row. Returns one-row-per-song results DF."""
    sys.path.insert(0, str(_HERE))
    from nlp_models import ViralLineDetector
    detector = ViralLineDetector()

    rows = []
    for _, row in df.iterrows():
        lyr = str(row.get("lyrics", "")).strip()

        if not lyr:
            rows.append({
                "song_title":     row["song_title"],
                "song_clean":     row["song_clean"],
                "artist":         row["artist_clean"],
                "lyrics_found":   False,
                "viral_prob":     None,
                "viral_line":     None,
                "viral_section":  None,
                "viral_repeats":  None,
                "viral_why":      None,
                "runner_up_line": None,
                "runner_up_prob": None,
                "lyrics":         "",
            })
            continue

        result = detector.predict(lyr)
        vl     = result["viral_line"]
        ru     = result["runner_up"]

        rows.append({
            "song_title":     row["song_title"],
            "song_clean":     row["song_clean"],
            "artist":         row["artist_clean"],
            "lyrics_found":   True,
            "viral_prob":     vl["viral_prob"]  if vl else None,
            "viral_line":     vl["text"]        if vl else None,
            "viral_section":  vl["section"]     if vl else None,
            "viral_repeats":  vl["repeats"]     if vl else None,
            "viral_why":      vl["why"]         if vl else None,
            "runner_up_line": ru["text"]        if ru else None,
            "runner_up_prob": ru["viral_prob"]  if ru else None,
            "lyrics":         lyr,
        })

    return pd.DataFrame(rows)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_results(
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Load → fetch lyrics → detect → save to RESULTS_CSV."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df  = load_tatiana()
    df  = fetch_all_lyrics(df, progress_cb)
    res = run_detector(df)
    res.to_csv(RESULTS_CSV, index=False)
    return res


# ── Load cached results ────────────────────────────────────────────────────────

def load_results() -> Optional[pd.DataFrame]:
    """Return cached results DataFrame or None if not yet built."""
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return None
