"""
predict_upcoming.py
The payoff of the whole project: predicts real, not-yet-played Premier
League fixtures, and logs every prediction so score_predictions.py can
later check how many were actually correct.

Run this periodically (e.g. once a week) to build up a real track record.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

STALE_FORM_DAYS = 200  # if a team's most recent match is older than this, treat as cold-start

import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_all_seasons
from feature_engineering import compute_latest_team_form
from fetch_live_data import fetch_upcoming_fixtures, fetch_current_season_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_LOG = PROJECT_ROOT / "data" / "predictions" / "predictions_log.csv"

FEATURE_COLS = [
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "home_avg_shots", "home_avg_shots_on_target", "home_form_points_last5",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "away_avg_shots", "away_avg_shots_on_target", "away_form_points_last5",
]


def build_feature_row(home_team, away_team, team_form: pd.DataFrame, league_avg: pd.Series):
    """
    Look up each team's current rolling form. If a team has no history yet
    (e.g. just promoted, hasn't played enough matches this season), fall
    back to the league average rather than crashing or silently using
    garbage -- and say so, loudly, since it's a real accuracy risk.
    """
    row = {}
    cutoff = pd.Timestamp.now() - timedelta(days=STALE_FORM_DAYS)
    for side, team in [("home", home_team), ("away", away_team)]:
        has_recent_data = team in team_form.index and team_form.loc[team, "last_match_date"] >= cutoff
        if has_recent_data:
            stats = team_form.loc[team]
        else:
            reason = "no history at all" if team not in team_form.index else \
                f"last match was {team_form.loc[team, 'last_match_date'].date()} -- too stale to trust"
            print(f"NOTE: '{team}' has {reason}. Using league average as a "
                  f"fallback. Prediction for this fixture will be less reliable "
                  f"(this is expected for newly-promoted teams early in the season).")
            stats = league_avg
        row[f"{side}_avg_goals_scored"] = stats["avg_goals_scored"]
        row[f"{side}_avg_goals_conceded"] = stats["avg_goals_conceded"]
        row[f"{side}_avg_shots"] = stats["avg_shots"]
        row[f"{side}_avg_shots_on_target"] = stats["avg_shots_on_target"]
        row[f"{side}_form_points_last5"] = stats["form_points_last5"]
    return row


def main(days_ahead: int = 10):
    print("Loading historical seasons (2000-01 to 2025-26)...")
    historical = load_all_seasons()

    print("\nFetching this season's played matches so far...")
    try:
        current = fetch_current_season_results()
        combined = pd.concat([historical, current], ignore_index=True)
    except Exception as e:
        print(f"Could not fetch current-season results ({e}). "
              f"Falling back to historical data only -- form will be "
              f"slightly stale (based on last season's final matches).")
        combined = historical

    print("\nComputing each team's current rolling form...")
    team_form = compute_latest_team_form(combined)

    # Fallback average for brand-new/cold-start teams. IMPORTANT: must be
    # restricted to recently-active teams -- team_form contains every club
    # that's played in the PL since 2000, including long-relegated sides
    # (e.g. a team last seen in 2011). Averaging across all of them would
    # silently pull the "league average" toward a decade-stale blend.
    cutoff = pd.Timestamp.now() - timedelta(days=STALE_FORM_DAYS)
    active_teams = team_form[team_form["last_match_date"] >= cutoff]
    league_avg = active_teams.mean(numeric_only=True)

    print("\nFetching upcoming fixtures...")
    fixtures = fetch_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures.empty:
        print("No upcoming fixtures to predict. Exiting.")
        return

    print("\nLoading trained models...")
    xgb_unweighted = joblib.load(MODELS_DIR / "xgboost_unweighted.pkl")
    xgb_balanced = joblib.load(MODELS_DIR / "xgboost_balanced.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")

    results = []
    for _, fixture in fixtures.iterrows():
        feat_dict = build_feature_row(fixture["HomeTeam"], fixture["AwayTeam"], team_form, league_avg)
        X = pd.DataFrame([feat_dict])[FEATURE_COLS]

        pred_unweighted = label_encoder.inverse_transform(xgb_unweighted.predict(X))[0]
        pred_balanced = label_encoder.inverse_transform(xgb_balanced.predict(X))[0]
        proba = xgb_unweighted.predict_proba(X)[0]
        proba_dict = {cls: round(p, 3) for cls, p in zip(label_encoder.classes_, proba)}

        results.append({
            "prediction_made_on": datetime.now().date().isoformat(),
            "match_date": fixture["Date"].date().isoformat(),
            "home_team": fixture["HomeTeam"],
            "away_team": fixture["AwayTeam"],
            "predicted_unweighted": pred_unweighted,
            "predicted_balanced": pred_balanced,
            "prob_home_win": proba_dict.get("H"),
            "prob_draw": proba_dict.get("D"),
            "prob_away_win": proba_dict.get("A"),
            "actual_result": "",   # filled in later by score_predictions.py
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'=' * 90}")
    print("PREDICTIONS")
    print(f"{'=' * 90}")
    outcome_word = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    for _, r in results_df.iterrows():
        print(f"{r['match_date']}  {r['home_team']:<16} vs {r['away_team']:<16}  "
              f"-> {outcome_word[r['predicted_unweighted']]:<10} "
              f"(H:{r['prob_home_win']:.0%} D:{r['prob_draw']:.0%} A:{r['prob_away_win']:.0%})  "
              f"[balanced model says: {outcome_word[r['predicted_balanced']]}]")

    # Append to the running predictions log
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_LOG.exists():
        existing = pd.read_csv(PREDICTIONS_LOG)
        combined_log = pd.concat([existing, results_df], ignore_index=True)
        combined_log = combined_log.drop_duplicates(
            subset=["match_date", "home_team", "away_team"], keep="last"
        )
    else:
        combined_log = results_df
    combined_log.to_csv(PREDICTIONS_LOG, index=False)
    print(f"\nLogged {len(results_df)} prediction(s) to {PREDICTIONS_LOG}")
    print("Run score_predictions.py after these matches are played to check accuracy.")


if __name__ == "__main__":
    main()
