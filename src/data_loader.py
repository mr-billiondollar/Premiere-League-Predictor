"""
data_loader.py
Loads every Premier League season CSV in data/raw/, standardizes them,
and returns one clean, chronologically-sorted DataFrame ready for
feature engineering.

Source data: football-data.co.uk match stats (2000-01 season onward,
the first season with shots/shots-on-target recorded).
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The columns we actually need for this project. football-data.co.uk
# files sometimes carry extra betting-odds columns we don't want.
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # full-time goals home/away, result
    "HS", "AS", "HST", "AST",       # shots, shots on target
    "HC", "AC",                     # corners (handy extra signal later)
]


def load_season_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # All season files here already use ISO format (YYYY-MM-DD) -- no
    # dayfirst ambiguity, so we parse it explicitly and directly.
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    # season label like "2000-01" from the filename, e.g. season-0001.csv
    season_code = path.stem.replace("season-", "")
    df["Season"] = f"20{season_code[:2]}-{season_code[2:]}"
    return df


def load_all_seasons(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("season-*.csv"))
    if not files:
        raise FileNotFoundError(f"No season-*.csv files found in {raw_dir}")

    frames = [load_season_file(f) for f in files]
    data = pd.concat(frames, ignore_index=True)

    # Drop any match missing core fields (a handful of postponed/void rows
    # sometimes sneak in with blank stats).
    before = len(data)
    data = data.dropna(subset=["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"])
    dropped = before - len(data)

    data = data.sort_values("Date").reset_index(drop=True)

    print(f"Loaded {len(files)} season files")
    print(f"Total matches: {len(data)}  (dropped {dropped} incomplete rows)")
    print(f"Date range: {data['Date'].min().date()} -> {data['Date'].max().date()}")

    return data


if __name__ == "__main__":
    df = load_all_seasons()
    print("\nColumns:", list(df.columns))
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print("\nResult distribution (FTR: H=home win, D=draw, A=away win):")
    print(df["FTR"].value_counts())
    print("\nAs %:")
    print((df["FTR"].value_counts(normalize=True) * 100).round(1))