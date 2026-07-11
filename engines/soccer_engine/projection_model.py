"""
Concrete `ev_engine_core.TrueProbabilityProvider` implementation for soccer.

This is the ADAPTER layer the architecture contract at the top of
ev_engine_core.py describes: ev_engine_core.py defines the
`TrueProbabilityProvider` Protocol but deliberately never imports a
concrete model (the generic EV engine shouldn't depend on one specific
projection approach). This module is the other side of that boundary --
it's the only place that imports both the team-strength / player-props
models AND ev_engine_core's `MarketLeg` / `ProbabilityEstimate` types.

Usage:

    ratings = fit_dixon_coles(fetch_historical_matches(api_key, competition_id))
    profiles = {p.player_id: build_player_rate_profile(p) for p in season_stats}
    model = SoccerProjectionModel(ratings, profiles)
    model.add_lineup(fetch_match_lineup(api_key, match_id))  # once a team sheet is confirmed
    results = build_match_ev_board(api_key, match_id, model)  # from ev_engine_core
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ev_engine_core import MarketLeg, ProbabilityEstimate, clamp_probability
from lineups import NOT_IN_SQUAD, MatchLineup
from player_props_model import (
    STAT_FIELD_BY_MARKET_TYPE,
    PlayerRateProfile,
    opponent_adjustment_factor,
    player_prop_probability,
)
from team_strength import (
    MAX_GOALS_GRID,
    SoccerModelDataError,
    TeamRatings,
    asian_handicap_probabilities,
    btts_probability,
    match_result_probabilities,
    scoreline_matrix,
    total_goals_over_probability,
)

TEAM_MARKET_FAMILIES = frozenset({"moneyline", "btts", "total", "spread"})
PLAYER_MARKET_FAMILIES = frozenset({"goalscorer", "shots", "assists"})


def normalize_probability_scale(value: float) -> float:
    """
    Defensive decimal/percentage conversion helper.

    ev_engine_core.ProbabilityEstimate always expects a 0-1 decimal (see its
    module-level architecture contract), and every probability this module
    computes internally (Poisson-derived) is already natively 0-1 -- so this
    helper is not load-bearing for our own math. It exists as a guard at the
    model boundary in case a future data source, or a misconfigured upstream
    feed, ever hands this hook a 0-100 style percentage (e.g. 65.0) instead
    of a decimal (0.65) by mistake: values above 1.0 are assumed to be on a
    0-100 scale and are divided down; anything already <= 1.0 passes through
    untouched.
    """

    if value > 1.0:
        return value / 100.0
    return value


class SoccerProjectionModel:
    """
    Implements `ev_engine_core.TrueProbabilityProvider`. Pass an instance of
    this class directly as the `probability_provider` argument to
    `evaluate_leg`, `rank_ev_board`, or `build_match_ev_board`.

    Never raises out of `__call__`: any leg this model can't confidently
    grade (missing team/player identifiers, unsupported market, insufficient
    fitted history, ...) comes back as a hard `true_probability=0.0` with an
    explicit warning describing why, rather than crashing the whole EV board
    over one bad leg or silently fabricating a number.
    """

    def __init__(
        self,
        team_ratings: TeamRatings,
        player_profiles: Mapping[str, PlayerRateProfile] | None = None,
        *,
        lineups: Mapping[str, MatchLineup] | None = None,
        max_goals: int = MAX_GOALS_GRID,
    ) -> None:
        self._team_ratings = team_ratings
        self._player_profiles: dict[str, PlayerRateProfile] = dict(player_profiles or {})
        self._lineups: dict[str, MatchLineup] = dict(lineups or {})
        self._max_goals = max_goals
        self._matrix_cache: dict[tuple[str, str], np.ndarray] = {}

    def add_lineup(self, lineup: MatchLineup) -> None:
        """
        Register a confirmed match lineup (see lineups.fetch_match_lineup),
        keyed by its match_id. Player-prop legs for that match_id graded
        AFTER this call will size expected minutes off the confirmed
        starting_xi/substitutes instead of historical
        minutes-per-appearance -- see player_props_model.expected_minutes_factor.
        """

        self._lineups[lineup.match_id] = lineup

    def __call__(self, leg: MarketLeg) -> ProbabilityEstimate:
        try:
            family = leg.market_family
            if family in TEAM_MARKET_FAMILIES:
                return self._grade_team_market(leg, family)
            if family in PLAYER_MARKET_FAMILIES:
                return self._grade_player_market(leg)
            return ProbabilityEstimate(
                true_probability=0.0,
                warnings=[f"no projection model coverage for market family {family!r}"],
            )
        except SoccerModelDataError as exc:
            # An expected, NAMED "can't grade this leg" condition -- not a
            # guess at what the probability might be. Surfaced as a hard
            # zero with a visible warning so the leg is excluded from the
            # positive-EV board rather than silently vanishing or crashing
            # every other leg's grading in the same run.
            return ProbabilityEstimate(true_probability=0.0, warnings=[str(exc)])

    def _matrix_for(self, home_team_id: str, away_team_id: str) -> np.ndarray:
        key = (home_team_id, away_team_id)
        cached = self._matrix_cache.get(key)
        if cached is None:
            lambda_home, lambda_away = self._team_ratings.expected_goals(home_team_id, away_team_id)
            cached = scoreline_matrix(lambda_home, lambda_away, self._team_ratings.rho, max_goals=self._max_goals)
            self._matrix_cache[key] = cached
        return cached

    def _grade_team_market(self, leg: MarketLeg, family: str) -> ProbabilityEstimate:
        home_team_id = leg.metadata.get("home_team_id")
        away_team_id = leg.metadata.get("away_team_id")
        if not home_team_id or not away_team_id:
            raise SoccerModelDataError(
                f"MarketLeg for game_id={leg.game_id!r} selection={leg.selection!r} is missing "
                "home_team_id/away_team_id metadata required to grade a team-level market -- "
                "pass team ids through flatten_thestatsapi_match_odds(...) / build_match_ev_board(...)."
            )

        matrix = self._matrix_for(str(home_team_id), str(away_team_id))
        sample_size = min(
            self._team_ratings.matches_played.get(str(home_team_id), 0),
            self._team_ratings.matches_played.get(str(away_team_id), 0),
        )

        if family == "moneyline":
            home_win, draw, away_win = match_result_probabilities(matrix)
            by_side = {"home": home_win, "draw": draw, "away": away_win}
            if leg.side not in by_side:
                raise SoccerModelDataError(f"Unrecognized match_odds side: {leg.side!r}")
            win_prob = by_side[leg.side]
            return ProbabilityEstimate(
                true_probability=clamp_probability(normalize_probability_scale(win_prob)),
                sample_size=sample_size,
            )

        if family == "btts":
            yes_prob = btts_probability(matrix)
            win_prob = yes_prob if leg.side == "yes" else (1.0 - yes_prob)
            return ProbabilityEstimate(
                true_probability=clamp_probability(normalize_probability_scale(win_prob)),
                sample_size=sample_size,
            )

        if family == "total":
            if leg.line is None:
                raise SoccerModelDataError(f"Total goals leg {leg.selection!r} is missing a line")
            over_prob = total_goals_over_probability(matrix, leg.line)
            win_prob = over_prob if leg.side == "over" else (1.0 - over_prob)
            return ProbabilityEstimate(
                true_probability=clamp_probability(normalize_probability_scale(win_prob)),
                sample_size=sample_size,
            )

        if family == "spread":
            if leg.line is None or leg.side not in {"home", "away"}:
                raise SoccerModelDataError(
                    f"Asian handicap leg {leg.selection!r} requires a line and a home/away side"
                )
            outcome = asian_handicap_probabilities(matrix, leg.side, leg.line)
            warnings = (
                [f"push probability {outcome.push_probability:.4f} excluded from EV by design (stake refunded)"]
                if outcome.push_probability > 0
                else []
            )
            return ProbabilityEstimate(
                true_probability=clamp_probability(normalize_probability_scale(outcome.win_probability)),
                loss_probability=clamp_probability(normalize_probability_scale(outcome.loss_probability)),
                sample_size=sample_size,
                warnings=warnings,
            )

        raise SoccerModelDataError(f"Unhandled team market family: {family!r}")

    def _grade_player_market(self, leg: MarketLeg) -> ProbabilityEstimate:
        if not leg.entity_id:
            raise SoccerModelDataError(f"Player-prop leg {leg.selection!r} is missing entity_id (player id)")

        profile = self._player_profiles.get(leg.entity_id)
        if profile is None:
            raise SoccerModelDataError(
                f"No player rate profile available for entity_id={leg.entity_id!r} ({leg.entity_name!r})"
            )

        stat_field = STAT_FIELD_BY_MARKET_TYPE.get(leg.market_type)
        if stat_field is None:
            raise SoccerModelDataError(
                f"No supported stat mapping for player-prop market_type={leg.market_type!r} "
                "(e.g. first_goalscorer is intentionally unsupported -- see player_props_model.py)"
            )

        home_team_id = leg.metadata.get("home_team_id")
        away_team_id = leg.metadata.get("away_team_id")
        opponent_team_id = None
        if profile.team_id and home_team_id and away_team_id:
            if str(profile.team_id) == str(home_team_id):
                opponent_team_id = str(away_team_id)
            elif str(profile.team_id) == str(away_team_id):
                opponent_team_id = str(home_team_id)

        adjustment = opponent_adjustment_factor(self._team_ratings, opponent_team_id)

        lineup_status = None
        lineup = self._lineups.get(leg.game_id)
        # A lineup that exists but isn't confirmed (see MatchLineup.confirmed)
        # is treated the same as no lineup at all -- an unconfirmed team
        # sheet is not trustworthy enough to zero out a player's minutes.
        if lineup is not None and lineup.confirmed:
            lineup_status = lineup.status_for(leg.entity_id)

        win_prob = player_prop_probability(
            profile, stat_field, leg.line, opponent_adjustment=adjustment, lineup_status=lineup_status
        )

        warnings = []
        if profile.appearances < 5:
            warnings.append(f"rate estimated from only {profile.appearances} appearances")
        if opponent_team_id is None:
            warnings.append("opponent team could not be resolved; used neutral (1.0x) defensive adjustment")
        if lineup_status == NOT_IN_SQUAD:
            warnings.append("confirmed lineup excludes this player from the matchday squad; minutes forced to 0")
        elif lineup_status is None:
            warnings.append("no confirmed lineup yet; using historical minutes-per-appearance estimate")

        return ProbabilityEstimate(
            true_probability=clamp_probability(normalize_probability_scale(win_prob)),
            sample_size=profile.appearances,
            warnings=warnings,
        )
