"""
Dixon-Coles Poisson goal model for team-level soccer markets.

Reference: Dixon, M.J. and Coles, S.G. (1997), "Modelling Association
Football Scores and Inefficiencies in the Football Betting Market", Journal
of the Royal Statistical Society Series C. The model treats home and away
goals as Poisson variables driven by each team's attack/defense strength and
a home-advantage multiplier, with a small correlation correction (`rho`)
applied only to the four low-scoring cells (0-0, 1-0, 0-1, 1-1) where an
independent-Poisson model is empirically known to misprice real match data.

WHY THIS SITS IN ITS OWN MODULE, SEPARATE FROM ev_engine_core.py: fitting
team strength from historical RESULTS must never see a sportsbook price, and
deriving probabilities from the fitted model must never see one either --
this file has zero imports from anything odds-related. The only place model
output and market odds are allowed to meet is `evaluate_leg` in
ev_engine_core.py (see that module's architecture-contract docstring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

from historical_data import MatchResult

# ~1-year half-life at the default xi -- old results still count, but a
# team's form from 18 months ago barely moves today's rating. This is the
# standard order of magnitude used in published Dixon-Coles implementations;
# tune per-competition if a league's season structure warrants it.
DEFAULT_TIME_DECAY_XI = 0.0018

# rho is a small correlation nudge on 4 cells, not a free-ranging parameter;
# published fits for top-flight leagues sit within roughly +-0.15. The wider
# bound here just keeps the optimizer from wandering into a nonsensical
# region rather than trying to hand-tune a tight prior.
DEFAULT_RHO_BOUNDS = (-0.4, 0.4)

# Attack/defense live in log-space. Without hard bounds, a few blowouts
# against weak sides can push a team's log-attack past ~2 (lambda ~ e^2 ~
# 7+ against league-average defense), which then floods Over markets with
# absurd true probabilities. +-2.0 keeps every pairwise lambda under e^4 ~
# 55 even in the worst mutual extreme -- and MAX_EXPECTED_GOALS below still
# clamps the projection path; this bound is what keeps MLE from *fitting*
# those extremes in the first place.
DEFAULT_ATTACK_DEFENSE_BOUNDS = (-2.0, 2.0)

# Home advantage is typically ~0.2-0.4 in log space for top leagues; +-1.0
# is already a generous envelope (e^1 ~ 2.7x home boost) without letting it
# become a free escape hatch for unbounded scoring rates.
DEFAULT_HOME_ADVANTAGE_BOUNDS = (-1.0, 1.0)

# Ridge weight on attack + defense log-ratings inside the NLL. Penalizes
# the squared magnitude of every team's attack/defense so a handful of
# historical thrashings can't buy an exponentially inflated rating. Tuned
# small enough that relative ordering on synthetic data still recovers
# (see test_fit_dixon_coles_recovers_relative_team_strength_from_synthetic_data)
# but large enough to pull tournament-history outliers back toward zero.
DEFAULT_L2_REGULARIZATION = 0.25

# Hard ceiling on projected expected goals (lambda / mu) for ANY single
# team in a matchup, applied in expected_goals() / scoreline_matrix()
# BEFORE the Poisson PMF is evaluated. Real soccer almost never clears
# 3.5 xG for one side in a competitive match; anything above that is a
# model pathology (usually from unbounded MLE on sparse national-team
# history), not a bettable signal -- clamping here stops Over 2.5/3.5
# true probs from exploding even if a stale cache still holds wild
# ratings.
MAX_EXPECTED_GOALS = 3.5

# Truncating the scoreline grid at 10 goals per side: Poisson(lambda<=4)
# assigns effectively zero mass beyond this in real soccer scoring ranges,
# and the matrix is renormalized to sum to 1 after truncation regardless.
MAX_GOALS_GRID = 10


class SoccerModelDataError(ValueError):
    """
    Raised for an EXPECTED "this specific leg/matchup can't be graded"
    condition (missing team identifiers, an unsupported line shape, a team
    with zero fitted history, etc.) -- never for "here is our best guess
    anyway". Callers (see projection_model.SoccerProjectionModel) catch this
    specifically and turn it into a zero-probability, clearly-warned result
    instead of fabricating a number or crashing an entire board over one bad
    leg.
    """


@dataclass(frozen=True)
class TeamRatings:
    """
    Fitted Dixon-Coles parameters, in log space (see `fit_dixon_coles`).

    `attack`/`defense` are only meaningful as RELATIVE values between teams:
    the model is additively scale-invariant in log space (adding a constant
    to every attack rating and subtracting it from every defense rating
    leaves every predicted scoreline unchanged), so an individual team's
    number should never be read as an absolute "goals per game" figure on
    its own -- only `expected_goals()` output (which combines both teams'
    ratings) is directly interpretable.
    """

    attack: dict[str, float]
    defense: dict[str, float]
    home_advantage: float
    rho: float
    league_avg_home_goals: float
    league_avg_away_goals: float
    matches_played: dict[str, int] = field(default_factory=dict)
    fitted_at: str = ""

    def expected_goals(self, home_team_id: str, away_team_id: str) -> tuple[float, float]:
        """
        (lambda_home, lambda_away): Poisson means for this specific matchup.

        Unrated teams (no historical matches in the fitted dataset) fall
        back to a log-rating of 0.0, i.e. league-average attack/defense --
        a deliberate, documented "no information" prior, not a guess about
        that specific team's actual strength.

        Each side's lambda is hard-capped at MAX_EXPECTED_GOALS before it
        leaves this method: the Poisson grid downstream must never see an
        8+ xG "projection" that is really just an unbounded MLE artifact.
        """

        log_attack_home = self.attack.get(home_team_id, 0.0)
        log_defense_home = self.defense.get(home_team_id, 0.0)
        log_attack_away = self.attack.get(away_team_id, 0.0)
        log_defense_away = self.defense.get(away_team_id, 0.0)

        lambda_home = math.exp(log_attack_home + log_defense_away + self.home_advantage)
        lambda_away = math.exp(log_attack_away + log_defense_home)
        return (
            min(float(lambda_home), MAX_EXPECTED_GOALS),
            min(float(lambda_away), MAX_EXPECTED_GOALS),
        )


def fit_dixon_coles(
    matches: Iterable[MatchResult],
    *,
    time_decay_xi: float = DEFAULT_TIME_DECAY_XI,
    as_of: datetime | None = None,
    rho_bounds: tuple[float, float] = DEFAULT_RHO_BOUNDS,
    attack_defense_bounds: tuple[float, float] = DEFAULT_ATTACK_DEFENSE_BOUNDS,
    home_advantage_bounds: tuple[float, float] = DEFAULT_HOME_ADVANTAGE_BOUNDS,
    l2_regularization: float = DEFAULT_L2_REGULARIZATION,
) -> TeamRatings:
    """
    Maximum-likelihood fit of Dixon-Coles attack/defense/home-advantage/rho
    parameters against a list of historical MatchResult rows, with
    exponential time-decay weighting so recent form matters more than
    results from a year+ ago.

    Uses a log-linear parametrization (log(lambda_home) = attack_home +
    defense_away + home_advantage, log(lambda_away) = attack_away +
    defense_home) -- the standard Poisson-regression form of this model,
    chosen because it keeps lambda positive automatically (no positivity
    constraints needed in the optimizer) and makes the additive
    scale-invariance explicit.

    Three stabilizers keep tournament / sparse-national-team history from
    producing hallucinated 8+ xG projections:

    1. L2 (Ridge) penalty on every attack/defense log-rating inside the NLL
       -- extreme magnitudes cost the optimizer even when they fit a few
       historical blowouts well.
    2. Hard box constraints on attack/defense (`attack_defense_bounds`,
       default +-2.0) and home advantage (`home_advantage_bounds`) passed
       to scipy.optimize.minimize (L-BFGS-B).
    3. A separate MAX_EXPECTED_GOALS clamp on `TeamRatings.expected_goals`
       / `scoreline_matrix` so the Poisson grid never sees an uncapped
       lambda even if a caller bypasses a fresh fit.
    """

    match_list = list(matches)
    if len(match_list) < 10:
        raise SoccerModelDataError(
            f"Need at least 10 finished matches with scores to fit team ratings; got {len(match_list)}."
        )

    teams = sorted({m.home_team_id for m in match_list} | {m.away_team_id for m in match_list})
    team_index = {team_id: i for i, team_id in enumerate(teams)}
    n_teams = len(teams)

    as_of = as_of or _latest_match_datetime(match_list)
    weights = np.array([_time_decay_weight(m.utc_date, as_of, time_decay_xi) for m in match_list])

    home_idx = np.array([team_index[m.home_team_id] for m in match_list])
    away_idx = np.array([team_index[m.away_team_id] for m in match_list])
    home_goals = np.array([m.home_goals for m in match_list], dtype=float)
    away_goals = np.array([m.away_goals for m in match_list], dtype=float)

    league_avg_home_goals = float(np.average(home_goals, weights=weights))
    league_avg_away_goals = float(np.average(away_goals, weights=weights))

    # Parameter vector layout: [attack_0..attack_{n-1}, defense_0..defense_{n-1}, home_advantage, rho]
    x0 = np.zeros(2 * n_teams + 2)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        # Mean-center attack and defense every evaluation. Without this the
        # log-linear model is additively scale-invariant (add c to every
        # attack and subtract c from every defense -- same lambdas), so L2
        # "toward zero" is meaningless and the whole league's attack can
        # drift positive together, which is exactly how a World-Cup history
        # fit ended up projecting 3.5+ xG for mid-tier sides. Centering
        # removes that gauge freedom so Ridge actually shrinks *relative*
        # strength outliers and home_advantage carries the league scoring
        # baseline.
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        attack = attack - np.mean(attack)
        defense = defense - np.mean(defense)
        home_advantage = params[2 * n_teams]
        rho = params[2 * n_teams + 1]
        return attack, defense, home_advantage, rho

    def negative_log_likelihood(params: np.ndarray) -> float:
        attack, defense, home_advantage, rho = unpack(params)
        log_lambda_home = attack[home_idx] + defense[away_idx] + home_advantage
        log_lambda_away = attack[away_idx] + defense[home_idx]
        lambda_home = np.exp(log_lambda_home)
        lambda_away = np.exp(log_lambda_away)

        tau = _dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho)
        # A pathological rho during optimization can push tau <= 0 for an
        # observed low scoreline; clip to a tiny positive floor so the
        # optimizer sees a very large (but finite, differentiable-enough)
        # penalty instead of -inf/NaN derailing the search.
        tau_safe = np.clip(tau, 1e-10, None)

        log_pmf_home = home_goals * log_lambda_home - lambda_home - gammaln(home_goals + 1.0)
        log_pmf_away = away_goals * log_lambda_away - lambda_away - gammaln(away_goals + 1.0)

        log_likelihood = weights * (np.log(tau_safe) + log_pmf_home + log_pmf_away)
        nll = float(-np.sum(log_likelihood))

        # Ridge / L2: penalize squared attack + defense magnitudes so the
        # MLE cannot buy an arbitrarily large rating off a few thrashings
        # of weak opponents. Home advantage and rho are already box-
        # constrained and are NOT included here -- regularizing them would
        # bias the league-wide home-field baseline toward zero for no gain
        # on the blowout pathology this term is meant to fix.
        if l2_regularization > 0.0:
            nll += float(l2_regularization) * float(np.sum(attack * attack) + np.sum(defense * defense))
        return nll

    lo, hi = attack_defense_bounds
    ha_lo, ha_hi = home_advantage_bounds
    bounds = (
        [(lo, hi)] * n_teams  # attack
        + [(lo, hi)] * n_teams  # defense
        + [(ha_lo, ha_hi)]  # home advantage
        + [rho_bounds]
    )
    fit = minimize(negative_log_likelihood, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})

    attack, defense, home_advantage, rho = unpack(fit.x)

    matches_played: dict[str, int] = {team_id: 0 for team_id in teams}
    for m in match_list:
        matches_played[m.home_team_id] += 1
        matches_played[m.away_team_id] += 1

    return TeamRatings(
        attack={team_id: float(attack[team_index[team_id]]) for team_id in teams},
        defense={team_id: float(defense[team_index[team_id]]) for team_id in teams},
        home_advantage=float(home_advantage),
        rho=float(rho),
        league_avg_home_goals=league_avg_home_goals,
        league_avg_away_goals=league_avg_away_goals,
        matches_played=matches_played,
        fitted_at=as_of.isoformat(),
    )


def _dixon_coles_tau(
    x: np.ndarray, y: np.ndarray, lambda_home: np.ndarray, lambda_away: np.ndarray, rho: float
) -> np.ndarray:
    """
    Low-score correlation correction from Dixon & Coles (1997). An
    independent-Poisson model systematically misprices 0-0/1-0/0-1/1-1
    scorelines relative to real match data; this term corrects exactly
    those four cells and leaves every other scoreline's probability
    untouched (tau = 1 elsewhere).
    """

    tau = np.ones_like(x)
    tau = np.where((x == 0) & (y == 0), 1.0 - lambda_home * lambda_away * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lambda_home * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + lambda_away * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
    return tau


def _parse_utc_date(value: str) -> datetime:
    """TheStatsAPI timestamps look like '2024-01-15T15:00:00.000Z'."""

    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _latest_match_datetime(matches: list[MatchResult]) -> datetime:
    return max((_parse_utc_date(m.utc_date) for m in matches), default=datetime.now(timezone.utc))


def _time_decay_weight(utc_date: str, as_of: datetime, xi: float) -> float:
    match_dt = _parse_utc_date(utc_date)
    days_elapsed = max(0.0, (as_of - match_dt).total_seconds() / 86400.0)
    return math.exp(-xi * days_elapsed)


def scoreline_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    *,
    max_goals: int = MAX_GOALS_GRID,
) -> np.ndarray:
    """
    matrix[i, j] = P(home scores i, away scores j), normalized to sum to 1.

    Built once per matchup and reused for every market derived from it
    (Match Result, BTTS, Total Goals, Asian Handicap) -- this guarantees
    those markets stay internally consistent with each other. The model's
    1X2 probabilities and its Over/Under 2.5 probability come from literally
    the same underlying distribution, not independently calibrated
    sub-models that could silently contradict one another.

    Both lambdas are clamped to MAX_EXPECTED_GOALS before the Poisson PMF
    is evaluated -- same ceiling as TeamRatings.expected_goals -- so a
    direct caller that bypasses expected_goals still cannot feed an 8+
    xG hallucination into the scoreline grid.
    """

    lambda_home = min(float(lambda_home), MAX_EXPECTED_GOALS)
    lambda_away = min(float(lambda_away), MAX_EXPECTED_GOALS)

    goal_range = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goal_range, lambda_home)
    away_pmf = poisson.pmf(goal_range, lambda_away)
    matrix = np.outer(home_pmf, away_pmf)

    matrix[0, 0] *= 1.0 - lambda_home * lambda_away * rho
    matrix[0, 1] *= 1.0 + lambda_home * rho
    matrix[1, 0] *= 1.0 + lambda_away * rho
    matrix[1, 1] *= 1.0 - rho

    total = matrix.sum()
    if total <= 0:
        raise SoccerModelDataError("Degenerate scoreline matrix (non-positive total probability)")
    return matrix / total


def match_result_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """(home_win, draw, away_win) probabilities, summing to 1.0 by construction."""

    home_idx, away_idx = np.indices(matrix.shape)
    home_win = float(matrix[home_idx > away_idx].sum())
    draw = float(matrix[home_idx == away_idx].sum())
    away_win = float(matrix[home_idx < away_idx].sum())
    return home_win, draw, away_win


def btts_probability(matrix: np.ndarray) -> float:
    """P(both teams score at least one goal)."""

    home_idx, away_idx = np.indices(matrix.shape)
    return float(matrix[(home_idx >= 1) & (away_idx >= 1)].sum())


def total_goals_over_probability(matrix: np.ndarray, line: float) -> float:
    """
    P(home_goals + away_goals > line). `line` is expected to be a
    half-integer (2.5, 3.5, ...), matching every real total-goals line seen
    from TheStatsAPI, so there is no push to account for here.
    """

    home_idx, away_idx = np.indices(matrix.shape)
    total_goals = home_idx + away_idx
    return float(matrix[total_goals > line].sum())


@dataclass(frozen=True)
class HandicapOutcome:
    """win + loss + push always sums to 1.0 (push is 0.0 for any half-integer line)."""

    win_probability: float
    loss_probability: float
    push_probability: float


def asian_handicap_probabilities(matrix: np.ndarray, side: str, line: float) -> HandicapOutcome:
    """
    `side` is "home" or "away"; `line` is signed from that side's
    perspective (e.g. side="home", line=-0.5 means "Home -0.5").

    Quarter lines (e.g. -0.25, -0.75) are split into two adjacent
    half-stake bets on the nearest half/whole lines, matching how
    sportsbooks actually settle Asian handicap quarter lines -- this is
    exact arithmetic on the two underlying half-lines, not an approximation.
    """

    quarters = round(line * 4)
    if quarters % 2 != 0:
        lower_line = (quarters - 1) / 4.0
        upper_line = (quarters + 1) / 4.0
        lower = _asian_handicap_single_line(matrix, side, lower_line)
        upper = _asian_handicap_single_line(matrix, side, upper_line)
        return HandicapOutcome(
            win_probability=(lower.win_probability + upper.win_probability) / 2.0,
            loss_probability=(lower.loss_probability + upper.loss_probability) / 2.0,
            push_probability=(lower.push_probability + upper.push_probability) / 2.0,
        )
    return _asian_handicap_single_line(matrix, side, line)


def _asian_handicap_single_line(matrix: np.ndarray, side: str, line: float) -> HandicapOutcome:
    home_idx, away_idx = np.indices(matrix.shape)
    if side == "home":
        adjusted_diff = (home_idx + line) - away_idx
    elif side == "away":
        adjusted_diff = (away_idx + line) - home_idx
    else:
        raise SoccerModelDataError(f"Asian handicap side must be 'home' or 'away', got {side!r}")

    win = float(matrix[adjusted_diff > 0].sum())
    push = float(matrix[adjusted_diff == 0].sum())
    loss = float(matrix[adjusted_diff < 0].sum())
    return HandicapOutcome(win_probability=win, loss_probability=loss, push_probability=push)
