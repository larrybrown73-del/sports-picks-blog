"""
Live starting-XI lineups for the player-prop projection model.

Sourced from TheStatsAPI's `/football/matches/{match_id}/lineups` endpoint
-- the same "what's actually confirmed to happen" category as
historical_data.py, just scoped to one upcoming match's team sheet rather
than season-level history. Kept in its own module because a lineup is
per-match and ephemeral (per TheStatsAPI's own docs, only available once
the official team sheet is announced, "approximately 1 hour before
kickoff"), unlike the season-aggregate data historical_data.py deals with.

This is what replaces player_props_model.py's historical
minutes-per-appearance placeholder once a real team sheet exists: a
confirmed starter still uses that historical per-90 baseline (see
`expected_minutes_factor`), but a confirmed non-starter's expected minutes
are overridden -- to a fixed substitute-cameo baseline if they're only on
the bench, or to exactly 0 if they're outside the matchday squad entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ev_engine_core import is_thestatsapi_not_found, thestatsapi_get

STARTING = "starting"
SUBSTITUTE = "substitute"
NOT_IN_SQUAD = "not_in_squad"


class LineupNotAvailable(Exception):
    """
    Raised when TheStatsAPI 404s because the official team sheet hasn't
    been announced yet. The correct response to "we don't know the lineup
    yet" is to say exactly that -- never to return an empty starting_xi,
    which would silently zero out every player's expected minutes as if
    they were all confirmed out.
    """


@dataclass(frozen=True)
class MatchLineup:
    """
    A single match's confirmed team sheet, reduced to just what the
    player-prop model needs: which player ids are starting, which are
    unused-bench-or-came-on substitutes, and which side each team was on.
    """

    match_id: str
    confirmed: bool
    starting_player_ids: frozenset[str]
    substitute_player_ids: frozenset[str]
    home_team_id: str | None
    away_team_id: str | None

    def status_for(self, player_id: str) -> str:
        """
        One of STARTING, SUBSTITUTE, or NOT_IN_SQUAD. NOT_IN_SQUAD means the
        player is confirmed absent from the matchday squad entirely
        (injury, rest, not registered) -- not "no data", a real, confirmed
        answer.
        """

        if player_id in self.starting_player_ids:
            return STARTING
        if player_id in self.substitute_player_ids:
            return SUBSTITUTE
        return NOT_IN_SQUAD


def fetch_match_lineup(api_key: str, match_id: str, *, session: Any | None = None) -> MatchLineup:
    """
    GET /football/matches/{match_id}/lineups.

    Raises LineupNotAvailable (never returns a guessed/empty lineup) on a
    404 -- callers (see scheduler.run_starting_xi_check) are expected to
    treat that as "try again later", not "assume nobody plays".
    """

    try:
        payload = thestatsapi_get(f"/football/matches/{match_id}/lineups", api_key, session=session)
    except Exception as exc:
        if is_thestatsapi_not_found(exc):
            raise LineupNotAvailable(f"Lineup for match {match_id!r} has not been announced yet.") from exc
        raise

    data = payload.get("data", payload)
    home = data.get("home") or {}
    away = data.get("away") or {}

    starting_ids = _player_ids(home, "starting_xi") | _player_ids(away, "starting_xi")
    substitute_ids = _player_ids(home, "substitutes") | _player_ids(away, "substitutes")

    return MatchLineup(
        match_id=str(data.get("match_id") or match_id),
        # `confirmed` defaults to False (not True) when absent: an
        # ambiguous/malformed payload should never be silently trusted as a
        # confirmed team sheet.
        confirmed=bool(data.get("confirmed", False)),
        starting_player_ids=frozenset(starting_ids),
        substitute_player_ids=frozenset(substitute_ids),
        home_team_id=str(home["id"]) if home.get("id") is not None else None,
        away_team_id=str(away["id"]) if away.get("id") is not None else None,
    )


def _player_ids(team_side: dict[str, Any], key: str) -> set[str]:
    return {str(player["id"]) for player in (team_side.get(key) or []) if player.get("id") is not None}
