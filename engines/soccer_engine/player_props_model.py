"""
Per-player Poisson rate model for soccer prop markets: anytime goalscorer,
shots, shots-on-target, and assists.

Deliberately simpler than the Dixon-Coles team model in team_strength.py: a
player's own per-90 rate for a counting stat, scaled by (a) how much of a
match they're typically involved in, and (b) the opponent's defensive
strength (borrowed from the Dixon-Coles `defense` rating as a documented,
explicit proxy -- teams that concede more goals also tend to allow more
shots/chances against, even though `defense` itself is calibrated on goals,
not shots. This is a modeling simplification, not a hidden assumption).

Expected playing time is sized by `expected_minutes_factor`, which prefers a
confirmed lineup (see lineups.py) over the player's own historical
minutes-per-appearance whenever one is available:
  - Confirmed STARTING: still uses the historical per-90 baseline -- a
    starter who's normally subbed at the 70th minute shouldn't suddenly be
    modeled as playing a full 90 just because they're starting.
  - Confirmed SUBSTITUTE (on the bench, not starting): the historical
    per-appearance average overstates a bench role, so expected involvement
    drops to a fixed substitute-cameo baseline instead.
  - Confirmed NOT_IN_SQUAD (omitted entirely -- injury, rest, not
    registered): expected minutes are exactly 0, a confirmed fact, not a
    guess.
  - No confirmed lineup yet (lineup_status=None): falls back to the
    player's historical minutes-per-appearance, same as before a team sheet
    exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import poisson

from historical_data import PlayerSeasonStats
from lineups import NOT_IN_SQUAD, SUBSTITUTE
from team_strength import TeamRatings

# A confirmed bench player's expected involvement if subbed on -- a
# documented, tunable stand-in for "some chance of a late cameo", not a
# per-player prediction. Deliberately conservative (a typical impact-sub
# window) rather than 0, since a bench player genuinely isn't a 0%
# probability of playing the way a NOT_IN_SQUAD player is.
SUBSTITUTE_BASELINE_MINUTES = 20.0

# Maps a TheStatsAPI player-prop market_type to the raw season counting-stat
# field it's measuring. "first_goalscorer" is intentionally absent: modeling
# WHO scores first requires a race-to-first-goal calculation across every
# player on both teams (conditioning on each team's own scoring-time
# distribution), not a simple per-player "scores at least once" Poisson --
# approximating it with the anytime-goalscorer math would silently misprice
# it, which the "never guess" rule for this engine does not allow.
STAT_FIELD_BY_MARKET_TYPE = {
    "anytime_goalscorer": "goals",
    "player_shots": "total_shots",
    "player_shots_on_target": "shots_on_target",
    "player_assists": "assists",
}


@dataclass(frozen=True)
class PlayerRateProfile:
    """Per-90 rates derived from one season of counting-stat totals."""

    player_id: str
    player_name: str
    team_id: str | None
    minutes_per_appearance: float
    goals_per_90: float
    assists_per_90: float
    shots_per_90: float
    shots_on_target_per_90: float
    appearances: int


def build_player_rate_profile(stats: PlayerSeasonStats) -> PlayerRateProfile | None:
    """
    Convert one season of raw counting stats into per-90 rates.

    Returns None (dropped, not guessed) if minutes_played is 0: a per-90
    rate is mathematically undefined without a positive minutes denominator,
    and a player with zero recorded minutes has no observed data to build a
    rate from in the first place.
    """

    if stats.minutes_played <= 0:
        return None

    per_90_factor = 90.0 / stats.minutes_played
    return PlayerRateProfile(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_id=stats.team_id,
        minutes_per_appearance=(stats.minutes_played / stats.appearances) if stats.appearances else 0.0,
        goals_per_90=stats.goals * per_90_factor,
        assists_per_90=stats.assists * per_90_factor,
        shots_per_90=stats.total_shots * per_90_factor,
        shots_on_target_per_90=stats.shots_on_target * per_90_factor,
        appearances=stats.appearances,
    )


def opponent_adjustment_factor(team_ratings: TeamRatings, opponent_team_id: str | None) -> float:
    """
    Scale factor for how much weaker/stronger than average the opponent's
    defense is, applied multiplicatively to a player's baseline rate.

    `team_ratings.defense` is log-scale (see team_strength.py); exponentiating
    the DIFFERENCE between the opponent's defense rating and the league
    average gives a multiplicative adjustment centered at 1.0 for an
    average defense, above 1.0 against a leakier-than-average defense, and
    below 1.0 against a stingier one. Returns 1.0 (neutral, no adjustment)
    if the opponent has no fitted rating or isn't identified.
    """

    if not opponent_team_id or not team_ratings.defense:
        return 1.0
    opponent_defense = team_ratings.defense.get(opponent_team_id)
    if opponent_defense is None:
        return 1.0
    league_avg_defense = sum(team_ratings.defense.values()) / len(team_ratings.defense)
    return math.exp(opponent_defense - league_avg_defense)


def expected_minutes_factor(profile: PlayerRateProfile, *, lineup_status: str | None = None) -> float:
    """
    Fraction of a full 90 minutes this player is expected to play.

    `lineup_status` is one of lineups.STARTING, lineups.SUBSTITUTE,
    lineups.NOT_IN_SQUAD, or None (no confirmed lineup yet). See the
    module-level docstring for what each value does; in short, only a
    confirmed NOT_IN_SQUAD or SUBSTITUTE status overrides the historical
    per-appearance heuristic -- STARTING and "unknown" both fall back to it.
    """

    if lineup_status == NOT_IN_SQUAD:
        return 0.0
    if lineup_status == SUBSTITUTE:
        return min(1.0, SUBSTITUTE_BASELINE_MINUTES / 90.0)

    # STARTING or unconfirmed (None): historical per-appearance heuristic.
    if profile.minutes_per_appearance <= 0:
        return 0.0
    return min(1.0, profile.minutes_per_appearance / 90.0)


def _lambda_for_stat(
    profile: PlayerRateProfile,
    stat_field: str,
    opponent_adjustment: float,
    *,
    lineup_status: str | None = None,
) -> float:
    per_90_by_field = {
        "goals": profile.goals_per_90,
        "assists": profile.assists_per_90,
        "total_shots": profile.shots_per_90,
        "shots_on_target": profile.shots_on_target_per_90,
    }
    if stat_field not in per_90_by_field:
        raise KeyError(f"Unknown player-prop stat field: {stat_field!r}")
    minutes_factor = expected_minutes_factor(profile, lineup_status=lineup_status)
    return per_90_by_field[stat_field] * minutes_factor * opponent_adjustment


def player_prop_probability(
    profile: PlayerRateProfile,
    stat_field: str,
    line: float | None,
    *,
    opponent_adjustment: float = 1.0,
    lineup_status: str | None = None,
) -> float:
    """
    P(stat count > line) for an "Over" line, or P(count >= 1) for a
    no-line "at least once" market (e.g. anytime goalscorer), modeling the
    count as Poisson(lambda) where lambda is the player's opponent- and
    minutes-adjusted per-match rate.

    Every real prop line seen from TheStatsAPI for these markets is a
    half-integer (0.5, 1.5, 2.5, ...), so there is no push to account for.
    """

    lam = _lambda_for_stat(profile, stat_field, opponent_adjustment, lineup_status=lineup_status)
    if lam <= 0:
        return 0.0
    if line is None:
        return float(1.0 - poisson.cdf(0, lam))
    floor_count = math.floor(line)
    return float(1.0 - poisson.cdf(floor_count, lam))
