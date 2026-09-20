"""
dixon_coles.py
Implements the Dixon-Coles (1997) extension of Maher's (1982) Poisson
goal-scoring model for football match prediction.

Core idea, different from the XGBoost classifier: rather than treating
Home/Draw/Away as three categories to classify, this models the number
of goals each team will score as Poisson-distributed random variables,
parameterized by each team's ATTACK strength, DEFENSE strength, and a
league-wide HOME ADVANTAGE constant. A draw isn't a separately-learned
class fighting for space against the other two -- it's just whatever
probability mass falls on home_goals == away_goals once you have a
proper joint model of the scoreline. This is why it tends to produce
better-calibrated draw probabilities than a direct classifier.

Model:
    home_goals ~ Poisson(lambda_home)
    away_goals ~ Poisson(lambda_away)
    lambda_home = exp(attack_home + defense_away + home_advantage)
    lambda_away = exp(attack_away + defense_home)

Dixon-Coles refinement: pure independent Poisson slightly underestimates
low-scoring draws (0-0, 1-1) in practice. A small correlation adjustment
(tau) corrects scores of 0-0, 1-0, 0-1, and 1-1 using one extra fitted
parameter, rho.

Time decay: more recent matches count more when fitting team strengths,
via exponential decay with a ~1-year half-life -- a team's current form
matters more than what they did 5 seasons ago.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score correlation adjustment."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class DixonColes:
    def __init__(self, half_life_days: float = 365):
        self.half_life_days = half_life_days
        self.teams = None
        self.params = None  # dict: team -> (attack, defense); plus 'home_adv', 'rho'

    def _decay_weights(self, dates: pd.Series, reference_date: pd.Timestamp) -> np.ndarray:
        days_ago = (reference_date - dates).dt.days.values
        xi = np.log(2) / self.half_life_days
        return np.exp(-xi * days_ago)

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp = None):
        """matches needs columns: Date, HomeTeam, AwayTeam, FTHG, FTAG"""
        if reference_date is None:
            reference_date = matches["Date"].max()

        self.teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
        n = len(self.teams)
        team_idx = {t: i for i, t in enumerate(self.teams)}

        weights = self._decay_weights(matches["Date"], reference_date)
        home_idx = matches["HomeTeam"].map(team_idx).values
        away_idx = matches["AwayTeam"].map(team_idx).values
        hg = matches["FTHG"].values
        ag = matches["FTAG"].values

        def unpack(params):
            attack = params[:n]
            defense = params[n:2 * n]
            home_adv = params[2 * n]
            rho = params[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(params):
            attack, defense, home_adv, rho = unpack(params)
            lam_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lam_away = np.exp(attack[away_idx] + defense[home_idx])

            ll = poisson.logpmf(hg, lam_home) + poisson.logpmf(ag, lam_away)
            # Apply the tau adjustment for low-scoring games
            tau_vals = np.array([
                tau(h, a, lh, la, rho)
                for h, a, lh, la in zip(hg, ag, lam_home, lam_away)
            ])
            tau_vals = np.clip(tau_vals, 1e-10, None)  # guard against invalid rho making tau <= 0
            ll = ll + np.log(tau_vals)
            return -np.sum(weights * ll)

        # Init: all attack/defense at 0 (average team), small positive home advantage, rho near 0
        x0 = np.zeros(2 * n + 2)
        x0[2 * n] = 0.25  # home_adv starting guess

        # Constraint: mean attack = 0, for identifiability (otherwise attack/defense
        # can drift by an arbitrary additive constant with no effect on lambda)
        constraints = [{"type": "eq", "fun": lambda p, n=n: np.mean(p[:n])}]

        result = minimize(
            neg_log_likelihood, x0, method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        attack, defense, home_adv, rho = unpack(result.x)
        self.params = {
            "attack": dict(zip(self.teams, attack)),
            "defense": dict(zip(self.teams, defense)),
            "home_adv": home_adv,
            "rho": rho,
        }
        self.fit_success = result.success
        self.fit_message = result.message
        return self

    def predict_probs(self, home_team: str, away_team: str, max_goals: int = 10):
        """Returns (P(Home win), P(Draw), P(Away win))."""
        a = self.params["attack"]
        d = self.params["defense"]
        home_adv = self.params["home_adv"]
        rho = self.params["rho"]

        # Teams with no fitted rating (e.g. brand-new to the league within
        # this training window) fall back to average strength (0, 0 under
        # our mean-zero identifiability constraint) -- same philosophy as
        # the XGBoost pipeline's league-average fallback for cold-start teams.
        a_home = a.get(home_team, 0.0)
        d_home = d.get(home_team, 0.0)
        a_away = a.get(away_team, 0.0)
        d_away = d.get(away_team, 0.0)

        lam_home = np.exp(a_home + d_away + home_adv)
        lam_away = np.exp(a_away + d_home)

        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
                p *= tau(i, j, lam_home, lam_away, rho)
                score_matrix[i, j] = p

        score_matrix /= score_matrix.sum()  # renormalize (truncation + tau can shift total slightly)

        p_home = np.sum(np.tril(score_matrix, -1))
        p_draw = np.sum(np.diag(score_matrix))
        p_away = np.sum(np.triu(score_matrix, 1))
        return p_home, p_draw, p_away

    def team_ratings_table(self) -> pd.DataFrame:
        """Attack/defense ratings, sorted by attack strength -- for a sanity check
        against known real team quality."""
        rows = [
            {"Team": t, "Attack": self.params["attack"][t], "Defense": self.params["defense"][t]}
            for t in self.teams
        ]
        return pd.DataFrame(rows).sort_values("Attack", ascending=False).reset_index(drop=True)
