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
from fetch_live_data import fetch_current_season_results, fetch_finished_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_LOG = PROJECT_ROOT / "data" / "predictions" / "predictions_log.csv"


def get_results_with_fallback():
    """
    Fetches from BOTH sources when possible and takes the UNION, rather
    than only falling back to football-data.org when football-data.co.uk
    throws an exception. This matters because football-data.co.uk has
    been observed to return a successful 200 response containing only
    the first few weeks of a season's results, weeks after more games
    were actually played -- a silent staleness bug, not a connection
    failure, so a plain try/except around exceptions never catches it.
    Cross-checking against an independent second source and merging
    catches this case too.
    """
    primary, fallback = None, None

    try:
        primary = fetch_current_season_results()
    except Exception as e:
        print(f"\nPrimary source (football-data.co.uk) unavailable: {e}")

    try:
        fallback = fetch_finished_results()
    except Exception as e:
        print(f"\nSecondary source (football-data.org) unavailable: {e}")

    if primary is None and fallback is None:
        print("\nBoth result sources are down right now -- this is likely temporary.")
        print("Your predictions log is unchanged. Try again in a few minutes.")
        return None
    if primary is None:
        return fallback
    if fallback is None:
        return primary

    combined = pd.concat([primary, fallback], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    if len(combined) > max(len(primary), len(fallback)):
        print(f"\nNote: football-data.co.uk alone returned {len(primary)} matches, "
              f"football-data.org alone returned {len(fallback)} -- combined to "
              f"{len(combined)} unique matches. (This gap is exactly the silent-staleness "
              f"issue this cross-check exists to catch.)")
    return combined


def main():
    if not PREDICTIONS_LOG.exists():
        print(f"No predictions logged yet at {PREDICTIONS_LOG}. Run predict_upcoming.py first.")
        return

    log = pd.read_csv(PREDICTIONS_LOG)
    log["match_date"] = pd.to_datetime(log["match_date"])

    print("Fetching latest results to check against predictions...")
    results = get_results_with_fallback()
    if results is None:
        return
    if results.empty:
        print("No finished matches found yet in the lookback window -- nothing to score.")
        return
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

    # Dixon-Coles was added partway through this project -- older logged
    # predictions won't have it, so score it only over the rows that do
    # rather than crashing or silently treating missing as wrong.
    has_dc = "predicted_dixon_coles" in played.columns and played["predicted_dixon_coles"].notna()
    if has_dc.any():
        dc_played = played[has_dc]
        dc_correct = (dc_played["predicted_dixon_coles"] == dc_played["actual_result"])
        n_dc = len(dc_played)
        print(f"Dixon-Coles:          {dc_correct.sum()}/{n_dc} correct ({dc_correct.mean():.1%})"
              + (f"  [only {n_dc}/{n} predictions have a Dixon-Coles pick logged]" if n_dc < n else ""))
        if n_dc >= 15:  # small samples make a calibration check misleading, not just noisy
            actual_draw_rate = (dc_played["actual_result"] == "D").mean()
            dc_avg_draw_prob = dc_played["dc_prob_draw"].mean()
            xgb_avg_draw_prob = dc_played["prob_draw"].mean()
            print(f"\nLive draw-probability calibration check (n={n_dc}):")
            print(f"  Actual draw rate so far:        {actual_draw_rate:.1%}")
            print(f"  Dixon-Coles' average estimate:  {dc_avg_draw_prob:.1%}")
            print(f"  XGBoost's average estimate:     {xgb_avg_draw_prob:.1%}")
            print("  (Whichever average sits closer to the actual rate is winning the "
                  "calibration bet live, not just in the original backtest.)")

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
