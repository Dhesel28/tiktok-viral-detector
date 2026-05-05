"""
Pipeline — orchestrates all steps:
  1. Build balanced_tiers (50 songs/tier from Summer Dataset)
  2. Fetch TikTok trending songs
  3. Enrich everything with Genius lyrics
  4. Merge into catalog_with_lyrics.csv

Usage:
  python pipeline.py --build          # full build from scratch
  python pipeline.py --tiktok-only    # only refresh TikTok trending list
  python pipeline.py --lyrics-only    # only fill missing lyrics
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from config import CATALOG_CSV, BALANCED_TIERS_CSV, TIKTOK_TRENDING_CSV


def build_catalog(skip_tiktok: bool = False, lyrics_only: bool = False) -> pd.DataFrame:
    """Full pipeline: sample → TikTok → lyrics → merge."""

    # ── Step 1: Balanced tiers ─────────────────────────────────────────────
    if not lyrics_only:
        print("\n=== Step 1: Building balanced_tiers (50/tier) ===")
        from dataset import build_balanced_tiers
        balanced = build_balanced_tiers()
        balanced["source"] = "balanced_tiers"
    else:
        from dataset import load_balanced_tiers
        balanced = load_balanced_tiers()
        balanced["source"] = "balanced_tiers"

    # ── Step 2: TikTok trending ────────────────────────────────────────────
    if not skip_tiktok and not lyrics_only:
        print("\n=== Step 2: Fetching TikTok trending songs ===")
        from tiktok_fetcher import get_trending_songs
        tiktok_df = get_trending_songs(count=80)
    elif TIKTOK_TRENDING_CSV.exists():
        tiktok_df = pd.read_csv(TIKTOK_TRENDING_CSV)
    else:
        tiktok_df = pd.DataFrame()

    # ── Step 3: Merge balanced + TikTok ───────────────────────────────────
    print("\n=== Step 3: Merging catalog ===")

    # Ensure both have same base columns
    base_cols = ["artist", "song", "tier", "source"]

    balanced_base = balanced[["artist", "song", "tier", "source"]].copy()

    if not tiktok_df.empty:
        tiktok_base = tiktok_df[["artist", "song", "tier", "source"]].copy() if "source" in tiktok_df.columns else tiktok_df[["artist", "song", "tier"]].assign(source="tiktok_trending")
        catalog = pd.concat([balanced_base, tiktok_base], ignore_index=True)
    else:
        catalog = balanced_base.copy()
        print("  [!] No TikTok songs — catalog is balanced_tiers only")

    # Carry over existing lyrics if catalog already exists
    if CATALOG_CSV.exists():
        existing = pd.read_csv(CATALOG_CSV)
        if "lyrics" in existing.columns:
            key = existing.set_index(["artist", "song"])["lyrics"].to_dict()
            catalog["lyrics"] = catalog.apply(
                lambda r: key.get((r["artist"], r["song"]), None), axis=1
            )
            already = catalog["lyrics"].notna() & (catalog["lyrics"].astype(str).str.strip() != "")
            print(f"  Carried over {already.sum()} existing lyric records")
        else:
            catalog["lyrics"] = None
    else:
        catalog["lyrics"] = None

    # Deduplicate
    before = len(catalog)
    catalog = catalog.drop_duplicates(subset=["artist", "song"]).reset_index(drop=True)
    print(f"  Catalog size: {len(catalog)} songs ({before - len(catalog)} duplicates removed)")

    # ── Step 4: Fetch missing lyrics ──────────────────────────────────────
    missing = catalog["lyrics"].isna() | (catalog["lyrics"].astype(str).str.strip() == "")
    print(f"\n=== Step 4: Fetching lyrics for {missing.sum()} songs (Genius) ===")

    if missing.sum() > 0:
        from genius_fetcher import enrich_with_lyrics
        catalog = enrich_with_lyrics(catalog, delay=0.6)
    else:
        print("  All lyrics already fetched — nothing to do.")
        catalog.to_csv(CATALOG_CSV, index=False)

    print(f"\n[Pipeline] Done. Catalog at: {CATALOG_CSV}")
    return catalog


def main():
    parser = argparse.ArgumentParser(description="TikTok Trend Lyrics Pipeline")
    parser.add_argument("--build", action="store_true", help="Full build from scratch")
    parser.add_argument("--tiktok-only", action="store_true", help="Refresh TikTok trending only")
    parser.add_argument("--lyrics-only", action="store_true", help="Only fill missing lyrics")
    args = parser.parse_args()

    if args.tiktok_only:
        from tiktok_fetcher import get_trending_songs
        get_trending_songs(count=80)
        print("TikTok trending list refreshed. Run --build or --lyrics-only to fetch their lyrics.")
        return

    if args.lyrics_only:
        build_catalog(skip_tiktok=True, lyrics_only=True)
        return

    # Default: full build
    build_catalog()


if __name__ == "__main__":
    main()
