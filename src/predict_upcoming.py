"""
predict_upcoming.py
The payoff of the whole project: predicts real, not-yet-played Premier
League fixtures, and logs every prediction so score_predictions.py can
later check how many were actually correct.

Run this periodically (e.g. once a week) to build up a real track record.

IMPORTANT:
XGBoost models are loaded using XGBoost's native JSON format instead of
joblib/pickle. This avoids XGBoost pickle corruption and version-related
loading problems.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import joblib
from xgboost import XGBClassifier


STALE_FORM_DAYS = 200


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_LOG = (
    PROJECT_ROOT / "data" / "predictions" / "predictions_log.csv"
)


# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_all_seasons
from feature_engineering import compute_latest_team_form
from fetch_live_data import (
    fetch_upcoming_fixtures,
    fetch_current_season_results,
)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "home_avg_shots",
    "home_avg_shots_on_target",
    "home_form_points_last5",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "away_avg_shots",
    "away_avg_shots_on_target",
    "away_form_points_last5",
]


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_feature_row(
    home_team,
    away_team,
    team_form: pd.DataFrame,
    league_avg: pd.Series,
):
    """
    Look up each team's current rolling form.

    If a team has no recent history, use the current active-league average
    instead of crashing or using stale historical data.
    """

    row = {}

    cutoff = pd.Timestamp.now() - timedelta(days=STALE_FORM_DAYS)

    for side, team in [
        ("home", home_team),
        ("away", away_team),
    ]:

        has_recent_data = (
            team in team_form.index
            and pd.notna(team_form.loc[team, "last_match_date"])
            and team_form.loc[team, "last_match_date"] >= cutoff
        )

        if has_recent_data:
            stats = team_form.loc[team]

        else:
            if team not in team_form.index:
                reason = "no history at all"
            else:
                last_match = team_form.loc[team, "last_match_date"]

                if pd.isna(last_match):
                    reason = "no valid last-match date"
                else:
                    reason = (
                        f"last match was {last_match.date()} "
                        f"-- too stale to trust"
                    )

            print(
                f"NOTE: '{team}' has {reason}. "
                f"Using league average as a fallback. "
                f"Prediction for this fixture will be less reliable."
            )

            stats = league_avg

        row[f"{side}_avg_goals_scored"] = stats[
            "avg_goals_scored"
        ]

        row[f"{side}_avg_goals_conceded"] = stats[
            "avg_goals_conceded"
        ]

        row[f"{side}_avg_shots"] = stats[
            "avg_shots"
        ]

        row[f"{side}_avg_shots_on_target"] = stats[
            "avg_shots_on_target"
        ]

        row[f"{side}_form_points_last5"] = stats[
            "form_points_last5"
        ]

    return row


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models():
    """
    Load all trained models.

    Random Forest models:
        joblib -> .pkl

    XGBoost models:
        native XGBoost JSON -> .json

    IMPORTANT:
    Do NOT use joblib.load() for XGBoost models.
    """

    print("\nLoading trained models...")

    # Random Forest models are still saved with joblib.
    rf_unweighted = joblib.load(
        MODELS_DIR / "random_forest_unweighted.pkl"
    )

    rf_balanced = joblib.load(
        MODELS_DIR / "random_forest_balanced.pkl"
    )

    # XGBoost models are now saved using model.save_model().
    xgb_unweighted = XGBClassifier()
    xgb_unweighted.load_model(
        MODELS_DIR / "xgboost_unweighted.json"
    )

    xgb_balanced = XGBClassifier()
    xgb_balanced.load_model(
        MODELS_DIR / "xgboost_balanced.json"
    )

    # Label encoder is shared by the XGBoost models.
    label_encoder = joblib.load(
        MODELS_DIR / "label_encoder.pkl"
    )

    return (
        rf_unweighted,
        rf_balanced,
        xgb_unweighted,
        xgb_balanced,
        label_encoder,
    )


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------

def main(days_ahead: int = 10):

    # -----------------------------------------------------------------------
    # Historical data
    # -----------------------------------------------------------------------

    print(
        "Loading historical seasons "
        "(2000-01 to 2025-26)..."
    )

    historical = load_all_seasons()

    # -----------------------------------------------------------------------
    # Current season results
    # -----------------------------------------------------------------------

    print("\nFetching this season's played matches so far...")

    try:
        current = fetch_current_season_results()

        combined = pd.concat(
            [historical, current],
            ignore_index=True,
        )

    except Exception as e:

        print(
            f"Could not fetch current-season results ({e}). "
            f"Falling back to historical data only -- form will be "
            f"slightly stale."
        )

        combined = historical

    # -----------------------------------------------------------------------
    # Compute current team form
    # -----------------------------------------------------------------------

    print("\nComputing each team's current rolling form...")

    team_form = compute_latest_team_form(combined)

    # -----------------------------------------------------------------------
    # League-average fallback
    # -----------------------------------------------------------------------

    cutoff = (
        pd.Timestamp.now()
        - timedelta(days=STALE_FORM_DAYS)
    )

    active_teams = team_form[
        team_form["last_match_date"] >= cutoff
    ]

    if active_teams.empty:
        print(
            "WARNING: No recently active teams found. "
            "Using all available team form data for fallback."
        )

        league_avg = team_form.mean(
            numeric_only=True
        )

    else:
        league_avg = active_teams.mean(
            numeric_only=True
        )

    # -----------------------------------------------------------------------
    # Upcoming fixtures
    # -----------------------------------------------------------------------

    print("\nFetching upcoming fixtures...")

    fixtures = fetch_upcoming_fixtures(
        days_ahead=days_ahead
    )

    if fixtures.empty:
        print(
            "No upcoming fixtures to predict. Exiting."
        )
        return

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------

    (
        rf_unweighted,
        rf_balanced,
        xgb_unweighted,
        xgb_balanced,
        label_encoder,
    ) = load_models()

    # -----------------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------------

    results = []

    for _, fixture in fixtures.iterrows():

        home_team = fixture["HomeTeam"]
        away_team = fixture["AwayTeam"]

        feat_dict = build_feature_row(
            home_team,
            away_team,
            team_form,
            league_avg,
        )

        X = pd.DataFrame(
            [feat_dict]
        )[FEATURE_COLS]

        # ---------------------------------------------------------------
        # XGBoost predictions
        # ---------------------------------------------------------------

        xgb_unweighted_encoded = xgb_unweighted.predict(X)

        xgb_balanced_encoded = xgb_balanced.predict(X)

        pred_unweighted = (
            label_encoder.inverse_transform(
                xgb_unweighted_encoded
            )[0]
        )

        pred_balanced = (
            label_encoder.inverse_transform(
                xgb_balanced_encoded
            )[0]
        )

        # ---------------------------------------------------------------
        # XGBoost probabilities
        # ---------------------------------------------------------------

        proba = xgb_unweighted.predict_proba(X)[0]

        proba_dict = {
            cls: round(float(prob), 3)
            for cls, prob in zip(
                label_encoder.classes_,
                proba,
            )
        }

        # ---------------------------------------------------------------
        # Store prediction
        # ---------------------------------------------------------------

        results.append(
            {
                "prediction_made_on": (
                    datetime.now()
                    .date()
                    .isoformat()
                ),

                "match_date": (
                    fixture["Date"]
                    .date()
                    .isoformat()
                ),

                "home_team": home_team,

                "away_team": away_team,

                "predicted_unweighted": pred_unweighted,

                "predicted_balanced": pred_balanced,

                "prob_home_win": proba_dict.get("H"),

                "prob_draw": proba_dict.get("D"),

                "prob_away_win": proba_dict.get("A"),

                "actual_result": "",
            }
        )

    results_df = pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # Print predictions
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 90}")
    print("PREDICTIONS")
    print(f"{'=' * 90}")

    outcome_word = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    for _, result in results_df.iterrows():

        print(
            f"{result['match_date']}  "
            f"{result['home_team']:<16} "
            f"vs "
            f"{result['away_team']:<16} "
            f"-> "
            f"{outcome_word[result['predicted_unweighted']]:<10} "
            f"(H:{result['prob_home_win']:.0%} "
            f"D:{result['prob_draw']:.0%} "
            f"A:{result['prob_away_win']:.0%}) "
            f"[balanced model says: "
            f"{outcome_word[result['predicted_balanced']]}]"
        )

    # -----------------------------------------------------------------------
    # Save prediction log
    # -----------------------------------------------------------------------

    PREDICTIONS_LOG.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if PREDICTIONS_LOG.exists():

        existing = pd.read_csv(
            PREDICTIONS_LOG
        )

        combined_log = pd.concat(
            [
                existing,
                results_df,
            ],
            ignore_index=True,
        )

        combined_log = combined_log.drop_duplicates(
            subset=[
                "match_date",
                "home_team",
                "away_team",
            ],
            keep="last",
        )

    else:

        combined_log = results_df

    combined_log.to_csv(
        PREDICTIONS_LOG,
        index=False,
    )

    # -----------------------------------------------------------------------
    # Finished
    # -----------------------------------------------------------------------

    print(
        f"\nLogged {len(results_df)} prediction(s) "
        f"to {PREDICTIONS_LOG}"
    )

    print(
        "Run score_predictions.py after these matches "
        "are played to check accuracy."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()