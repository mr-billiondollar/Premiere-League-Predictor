"""
score_predictions.py
This is the whole point of Phase 5: checks logged predictions against
real results once matches have been played, and reports running accuracy.

Run this periodically (e.g. a day or two after a predicted match date).
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from fetch_live_data import fetch_current_season_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_LOG = PROJECT_ROOT / "data" / "predictions" / "predictions_log.csv"


def main():
    if not PREDICTIONS_LOG.exists():
        print(f"No predictions logged yet at {PREDICTIONS_LOG}. Run predict_upcoming.py first.")
        return

    log = pd.read_csv(PREDICTIONS_LOG)
    log["match_date"] = pd.to_datetime(log["match_date"])

    print("Fetching latest results to check against predictions...")
    results = fetch_current_season_results()
    results = results.rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team", "FTR": "actual"})

    # Match each prediction to its real result by date + teams
    merged = log.merge(
        results[["Date", "home_team", "away_team", "actual"]],
        left_on=["match_date", "home_team", "away_team"],
        right_on=["Date", "home_team", "away_team"],
        how="left",
    )
    merged["actual_result"] = merged["actual"].combine_first(merged["actual_result"])
    merged = merged.drop(columns=["Date", "actual"])
    merged.to_csv(PREDICTIONS_LOG, index=False)  # save updated actuals back

    played = merged.dropna(subset=["actual_result"])
    played = played[played["actual_result"] != ""]

    if played.empty:
        print("No predicted matches have been played yet -- nothing to score.")
        return

    played["correct_unweighted"] = played["predicted_unweighted"] == played["actual_result"]
    played["correct_balanced"] = played["predicted_balanced"] == played["actual_result"]

    n = len(played)
    acc_unweighted = played["correct_unweighted"].mean()
    acc_balanced = played["correct_balanced"].mean()

    print(f"\n{'=' * 70}")
    print(f"LIVE TRACK RECORD -- {n} predicted matches have now been played")
    print(f"{'=' * 70}")
    print(f"XGBoost (unweighted): {played['correct_unweighted'].sum()}/{n} correct ({acc_unweighted:.1%})")
    print(f"XGBoost (balanced):   {played['correct_balanced'].sum()}/{n} correct ({acc_balanced:.1%})")

    print(f"\nMatch-by-match:")
    for _, r in played.iterrows():
        mark_u = "correct" if r["correct_unweighted"] else "wrong"
        print(f"  {r['match_date'].date()}  {r['home_team']} vs {r['away_team']}: "
              f"predicted {r['predicted_unweighted']}, actual {r['actual_result']} -- {mark_u}")

    print(f"\nFull log saved to {PREDICTIONS_LOG}")
    print("This running accuracy number -- calculated on real predictions made BEFORE the")
    print("matches happened -- is your strongest resume/interview talking point. It's the")
    print("difference between 'I backtested a model' and 'I ran it live and tracked results.'")


if __name__ == "__main__":
    main()
