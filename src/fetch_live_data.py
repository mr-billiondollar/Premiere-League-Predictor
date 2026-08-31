"""
fetch_live_data.py
Two different free data sources, used for what each is actually good at:

1. football-data.org API -> the UPCOMING FIXTURE LIST (who plays who, when).
   Free tier does NOT include shots/possession stats -- just fixtures and
   (delayed) scores. That's fine, we only need the schedule from here.

2. football-data.co.uk current-season CSV -> RECENT RESULTS WITH SHOTS.
   Same schema as our training data (data/raw/season-XXXX.csv), updated
   throughout the season. This is what lets us compute each team's
   current rolling form using the exact same features the model was
   trained on.

Set your football-data.org API key as an environment variable before
running anything that calls fetch_upcoming_fixtures():
    export FOOTBALL_DATA_API_KEY="your_key_here"      (Mac/Linux)
    setx FOOTBALL_DATA_API_KEY "your_key_here"          (Windows, new terminal after)
"""

import os
import requests
import pandas as pd
from datetime import date, timedelta
from io import StringIO

from team_mapping import normalize_team_name, print_unmapped_teams

FOOTBALL_DATA_ORG_BASE = "https://api.football-data.org/v4"
CURRENT_SEASON_CODE = "2627"  # 2026-27 season -- bump this next August
CURRENT_SEASON_CSV_URL = f"https://www.football-data.co.uk/mmz4281/{CURRENT_SEASON_CODE}/E0.csv"


def fetch_upcoming_fixtures(days_ahead: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame of scheduled Premier League matches in the next
    `days_ahead` days, with team names already converted to our training
    data's naming convention.
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FOOTBALL_DATA_API_KEY environment variable is not set. "
            "Get a free key at https://www.football-data.org/client/register "
            "and set it before running this script."
        )

    today = date.today()
    params = {
        "status": "SCHEDULED",
        "dateFrom": today.isoformat(),
        "dateTo": (today + timedelta(days=days_ahead)).isoformat(),
    }
    headers = {"X-Auth-Token": api_key}
    url = f"{FOOTBALL_DATA_ORG_BASE}/competitions/PL/matches"

    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    matches = data.get("matches", [])
    if not matches:
        print(f"No scheduled PL matches found in the next {days_ahead} days. "
              f"Try increasing days_ahead, or check that the season is active.")
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])

    rows = []
    api_names_seen = set()
    for m in matches:
        home_api_name = m["homeTeam"]["name"]
        away_api_name = m["awayTeam"]["name"]
        api_names_seen.add(home_api_name)
        api_names_seen.add(away_api_name)
        rows.append({
            "Date": m["utcDate"][:10],
            "HomeTeam": normalize_team_name(home_api_name),
            "AwayTeam": normalize_team_name(away_api_name),
        })

    print_unmapped_teams(api_names_seen)

    fixtures = pd.DataFrame(rows)
    fixtures["Date"] = pd.to_datetime(fixtures["Date"])
    print(f"Fetched {len(fixtures)} upcoming fixture(s) from football-data.org")
    return fixtures


def fetch_current_season_results() -> pd.DataFrame:
    """
    Downloads the current season's played matches (with shots data) from
    football-data.co.uk. Same column schema as our historical training CSVs.
    """
    response = requests.get(CURRENT_SEASON_CSV_URL, timeout=15)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    keep_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
                 "HS", "AS", "HST", "AST", "HC", "AC"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"])  # keep only played matches
    df["Season"] = "2026-27"

    print(f"Fetched {len(df)} played match(es) so far this season from football-data.co.uk")
    return df


if __name__ == "__main__":
    print("Testing fixture fetch...")
    try:
        fixtures = fetch_upcoming_fixtures()
        print(fixtures)
    except EnvironmentError as e:
        print(f"Skipped (expected if you haven't set your API key yet): {e}")

    print("\nTesting current season results fetch...")
    results = fetch_current_season_results()
    print(results.tail())
