"""Backward-compatible re-exports; hits guardrails live in guardrails.py."""
from __future__ import annotations

from baseball_props.analysis.guardrails import (
    GameContext,
    HitsPropEvaluation,
    evaluate_hits_prop,
    fetch_game_weather_mlb,
)

TBPropEvaluation = HitsPropEvaluation


def evaluate_total_bases_prop(
    player_id: str,
    opponent_pitcher_id: str,
    game_context: GameContext,
    *,
    proj_tb: float,
    market_line: float,
    over_odds: float | None,
    under_odds: float | None,
    prop_lines=None,
    xbh_profile=None,
) -> HitsPropEvaluation:
    """Deprecated alias — maps TB projection to hits guardrail path."""
    del prop_lines, xbh_profile
    return evaluate_hits_prop(
        player_id,
        opponent_pitcher_id,
        game_context,
        proj_hits=float(proj_tb),
        market_line=market_line,
        over_odds=over_odds,
        under_odds=under_odds,
    )
