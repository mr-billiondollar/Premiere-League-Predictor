"""
feature_engineering.py
Turns raw match rows into model-ready features by computing each team's
rolling recent form BEFORE each match (no data leakage - a match's
features never include that match's own result).

Output: data/processed/model_ready.csv
One row per fixture, with home_* and away_* rolling features plus the
target column FTR (H/D/A).
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_all_seasons

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ROLLING_WINDOW = 5  # matches of history used for every rolling feature


def build_team_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape match-level rows into team-level rows: one row per team per
    match, from that team's point of view. This lets us compute rolling
    stats per team regardless of whether they were home or away.
    """
    matches = matches.reset_index().rename(columns={"index": "match_id"})

    home = matches[[
        "match_id", "Date", "Season", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "HS", "AS", "HST", "AST",
    ]].copy()
    home.columns = [
        "match_id", "Date", "Season", "Team", "Opponent",
        "GF", "GA", "Shots", "ShotsAgainst", "SOT", "SOTAgainst",
    ]
    home["Venue"] = "H"

    away = matches[[
        "match_id", "Date", "Season", "AwayTeam", "HomeTeam",
        "FTAG", "FTHG", "AS", "HS", "AST", "HST",
    ]].copy()
    away.columns = [
        "match_id", "Date", "Season", "Team", "Opponent",
        "GF", "GA", "Shots", "ShotsAgainst", "SOT", "SOTAgainst",
    ]
    away["Venue"] = "A"

    log = pd.concat([home, away], ignore_index=True)

    # Points earned in that match, from this team's perspective
    log["Points"] = 0
    log.loc[log["GF"] > log["GA"], "Points"] = 3
    log.loc[log["GF"] == log["GA"], "Points"] = 1

    return log.sort_values(["Team", "Date"]).reset_index(drop=True)


def add_rolling_features(log: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    For each team, compute rolling averages over their last `window`
    matches -- shifted by 1 so today's match is never included in its
    own features.
    """
    log = log.sort_values(["Team", "Date"]).copy()
    grouped = log.groupby("Team", group_keys=False)

    def shifted_roll(col, agg="mean"):
        shifted = grouped[col].shift(1)
        roller = shifted.groupby(log["Team"]).rolling(window, min_periods=window)
        result = roller.agg(agg).reset_index(level=0, drop=True)
        return result

    log["avg_goals_scored"] = shifted_roll("GF")
    log["avg_goals_conceded"] = shifted_roll("GA")
    log["avg_shots"] = shifted_roll("Shots")
    log["avg_shots_on_target"] = shifted_roll("SOT")
    log["form_points_last5"] = shifted_roll("Points", agg="sum")

    return log


def build_fixture_features(matches: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Merge each team's pre-match rolling features back onto fixture rows."""
    feature_cols = [
        "avg_goals_scored", "avg_goals_conceded",
        "avg_shots", "avg_shots_on_target", "form_points_last5",
    ]

    matches = matches.reset_index().rename(columns={"index": "match_id"})

    home_feats = log[log["Venue"] == "H"][["match_id"] + feature_cols].copy()
    home_feats.columns = ["match_id"] + [f"home_{c}" for c in feature_cols]

    away_feats = log[log["Venue"] == "A"][["match_id"] + feature_cols].copy()
    away_feats.columns = ["match_id"] + [f"away_{c}" for c in feature_cols]

    out = matches.merge(home_feats, on="match_id").merge(away_feats, on="match_id")
    return out


def build_dataset() -> pd.DataFrame:
    matches = load_all_seasons()
    log = build_team_match_log(matches)
    log = add_rolling_features(log)
    dataset = build_fixture_features(matches, log)

    keep = [
        "match_id", "Date", "Season", "HomeTeam", "AwayTeam",
        "home_avg_goals_scored", "home_avg_goals_conceded",
        "home_avg_shots", "home_avg_shots_on_target", "home_form_points_last5",
        "away_avg_goals_scored", "away_avg_goals_conceded",
        "away_avg_shots", "away_avg_shots_on_target", "away_form_points_last5",
        "FTR",
    ]
    dataset = dataset[keep]

    before = len(dataset)
    dataset = dataset.dropna().reset_index(drop=True)
    dropped = before - len(dataset)

    print(f"Fixtures before dropping cold-start rows: {before}")
    print(f"Dropped {dropped} rows (teams without {ROLLING_WINDOW} prior matches "
          f"of history yet -- e.g. early-season / newly promoted sides)")
    print(f"Final model-ready dataset: {len(dataset)} fixtures")
    print(f"\nTarget distribution:\n{dataset['FTR'].value_counts(normalize=True).round(3) * 100}")

    return dataset


if __name__ == "__main__":
    dataset = build_dataset()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "model_ready.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\nSample rows:")
    print(dataset.head(3).to_string(index=False))
