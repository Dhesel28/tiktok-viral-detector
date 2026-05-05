"""Load API keys and shared config from .env"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Reuse the .env from the sibling viral_lyrics_analyzer project
_env_path = Path(__file__).parent.parent / "viral_lyrics_analyzer" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent / ".env")

GENIUS_TOKEN    = os.getenv("GENIUS_ACCESS_TOKEN", "")
RAPIDAPI_KEY    = os.getenv("RAPIDAPI_KEY", "")
TIKTOK_API_HOST = "tiktok-api23.p.rapidapi.com"

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SUMMER_DATASET      = Path(__file__).parent.parent / "Summer Dataset.csv"
CATALOG_CSV         = DATA_DIR / "catalog_with_lyrics.csv"
TIKTOK_TRENDING_CSV = DATA_DIR / "tiktok_trending.csv"
BALANCED_TIERS_CSV  = DATA_DIR / "balanced_tiers.csv"

SAMPLE_PER_TIER = 50
