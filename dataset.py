"""
Load Summer Dataset.csv and build balanced_tiers.csv
by random-sampling SAMPLE_PER_TIER songs from each tier.
"""

from __future__ import annotations

import pandas as pd
from config import SUMMER_DATASET, BALANCED_TIERS_CSV, SAMPLE_PER_TIER


def build_balanced_tiers(random_state: int = 42) -> pd.DataFrame:
    """
    Read Summer Dataset.csv, sample SAMPLE_PER_TIER rows per tier,
    save to balanced_tiers.csv, and return the DataFrame.
    """
    print(f"[Dataset] Loading {SUMMER_DATASET.name} ...")
    df = pd.read_csv(SUMMER_DATASET)

    required = {"song_id", "artist", "song", "tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summer Dataset missing columns: {missing}")

    df["tier"] = df["tier"].astype(int)
    tiers = sorted(df["tier"].unique())
    print(f"  Tiers found: {tiers}")
    print(f"  Tier counts: {df['tier'].value_counts().sort_index().to_dict()}")

    sampled_frames = []
    for tier in tiers:
        tier_df = df[df["tier"] == tier]
        n = min(SAMPLE_PER_TIER, len(tier_df))
        sampled = tier_df.sample(n=n, random_state=random_state)
        sampled_frames.append(sampled)
        print(f"  Tier {tier}: sampled {n}/{len(tier_df)}")

    balanced = pd.concat(sampled_frames, ignore_index=True)
    balanced = balanced[["song_id", "artist", "song", "tier"]].copy()
    balanced.to_csv(BALANCED_TIERS_CSV, index=False)
    print(f"[Dataset] Saved {len(balanced)} rows → {BALANCED_TIERS_CSV}")
    return balanced


def load_balanced_tiers() -> pd.DataFrame:
    """Load existing balanced_tiers.csv, or build it if missing."""
    if BALANCED_TIERS_CSV.exists():
        return pd.read_csv(BALANCED_TIERS_CSV)
    return build_balanced_tiers()


if __name__ == "__main__":
    df = build_balanced_tiers()
    print(df.head(10).to_string(index=False))
