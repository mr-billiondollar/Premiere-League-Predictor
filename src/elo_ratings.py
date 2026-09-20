"""
elo_ratings.py
A continuously-updated Elo rating per team, unlike the fixed 5-match
rolling window used elsewhere in this project. Every match nudges both
teams' ratings based on how surprising the result was -- beating a much
stronger team moves your rating more than beating a much weaker one, and
the size of the win matters too (a 4-0 win says more than a 1-0 win).

Formula (matches the well-established World Football Elo Ratings
approach used by eloratings.net):
    expected_home = 1 / (1 + 10^(-(home_elo + HOME_ADV - away_elo) / 400))
    actual_home   = 1.0 win / 0.5 draw / 0.0 loss
    goal_diff_multiplier: bigger margins move ratings more (see below)
    delta = K * goal_diff_multiplier * (actual_home - expected_home)
    home_elo += delta
    away_elo -= delta   (zero-sum: home_adv only affects what's "expected",
                         not the stored rating, so ratings stay zero-sum)

Same leakage discipline as everywhere else in this project: a match's
Elo FEATURE is always each team's rating going INTO that match (i.e.
computed from every prior match only), never including that match's own
result. This falls out naturally from processing matches in chronological
order and reading the rating dict BEFORE updating it.
"""

import pandas as pd

INITIAL_RATING = 1500
HOME_ADVANTAGE = 100  # rating points, matches the commonly-cited ~100 pt home edge in football Elo systems
K_FACTOR = 20


def goal_diff_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more. Standard World Football Elo scaling."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8


def compute_elo_history(matches: pd.DataFrame, k: float = K_FACTOR,
                         home_advantage: float = HOME_ADVANTAGE,
                         initial_rating: float = INITIAL_RATING) -> pd.DataFrame:
    """
    Walks through matches in chronological order, returning a DataFrame
    with each match's PRE-match Elo for both teams (leakage-safe by
    construction -- read the rating, THEN update it).

    matches needs columns: Date, HomeTeam, AwayTeam, FTHG, FTAG.
    New teams (never seen before) start at initial_rating the moment
    they play their first match in this history.
    """
    matches = matches.sort_values("Date").reset_index(drop=True)
    ratings = {}
    home_elo_before, away_elo_before = [], []

    for _, row in matches.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        r_home = ratings.get(home, initial_rating)
        r_away = ratings.get(away, initial_rating)

        home_elo_before.append(r_home)
        away_elo_before.append(r_away)

        hg, ag = row["FTHG"], row["FTAG"]
        actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        expected_home = 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / 400))
        gd_mult = goal_diff_multiplier(hg - ag)

        delta = k * gd_mult * (actual_home - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    matches = matches.copy()
    matches["home_elo"] = home_elo_before
    matches["away_elo"] = away_elo_before
    matches["elo_diff"] = matches["home_elo"] - matches["away_elo"]
    return matches, ratings


def get_current_ratings(matches: pd.DataFrame, k: float = K_FACTOR,
                         home_advantage: float = HOME_ADVANTAGE,
                         initial_rating: float = INITIAL_RATING) -> dict:
    """For live prediction: every team's Elo as of right now (after the
    most recent match in `matches`), ready to use for a not-yet-played
    fixture. New/unseen teams simply aren't in the returned dict --
    callers should fall back to initial_rating (1500, an average team)."""
    _, ratings = compute_elo_history(matches, k, home_advantage, initial_rating)
    return ratings


def ratings_table(ratings: dict) -> pd.DataFrame:
    """Sorted ratings snapshot -- for a sanity check against known team quality."""
    return pd.DataFrame(
        [{"Team": t, "Elo": r} for t, r in ratings.items()]
    ).sort_values("Elo", ascending=False).reset_index(drop=True)
