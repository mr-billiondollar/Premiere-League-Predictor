"""
update_current_season.py
Downloads the CURRENT Premier League season's live-updating results CSV
from football-data.co.uk (same source/schema as our historical data,
just being added to throughout the season) and saves it into data/raw/
so it flows through the exact same load_all_seasons() pipeline.

Run this before predict_upcoming.py or check_results.py so both scripts
work off freshly played matches.
"""

import pandas as pd
import requests
from pathlib import Path
from datetime import date

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def current_season_code(today: date = None) -> str:
    """
    The Premier League season runs Aug -> May. football-data.co.uk names
    files by two-digit start/end year, e.g. 2025-26 season -> "2526".
    If we're before August, we're still in the season that started last
    calendar year.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    end_year = start_year + 1
    return f"{str(start_year)[2:]}{str(end_year)[2:]}"


def download_current_season(season_code: str = None) -> Path:
    season_code = season_code or current_season_code()
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    out_path = RAW_DIR / f"season-{season_code}.csv"

    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))

    # football-data.co.uk's live in-season files use dd/mm/yy (usually) --
    # NOT the ISO format our historical files use. Normalize here so
    # data_loader.py doesn't need to know the difference between sources.
    try:
        parsed = pd.to_datetime(df["Date"], format="%d/%m/%y")
    except ValueError:
        parsed = pd.to_datetime(df["Date"], dayfirst=True)  # fallback, e.g. dd/mm/yyyy
    df["Date"] = parsed.dt.strftime("%Y-%m-%d")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} matches so far this season -> {out_path}")
    return out_path


if __name__ == "__main__":
    download_current_season()
