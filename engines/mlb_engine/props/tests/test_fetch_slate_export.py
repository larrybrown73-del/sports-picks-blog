from __future__ import annotations

from datetime import date

import pandas as pd

from baseball_props.analysis.game_splits import (
    build_betting_intel_export,
    filter_conviction_plays,
)
from baseball_props.data.schemas import SlateContext
from baseball_props.pipeline.slate_run import SlateRunResult
from baseball_props.types import SlateFrames


def test_filter_conviction_plays_line_side_top_n() -> None:
    plays = [
        {"player": "A", "line": 1.5, "rec": "Over", "edge": 10.0, "confidence_tier": "Tier-1", "suggested_stake": 20.0},
        {"player": "B", "line": 1.0, "rec": "Over", "edge": 50.0, "confidence_tier": "Tier-1", "suggested_stake": 20.0},
        {"player": "C", "line": 1.5, "rec": "Under", "edge": 40.0, "confidence_tier": "Tier-1", "suggested_stake": 20.0},
        {"player": "D", "line": 1.5, "rec": "Over", "edge": 30.0, "confidence_tier": "Tier-2", "suggested_stake": 15.0},
        {"player": "E", "line": 1.5, "rec": "Over", "edge": 5.0, "confidence_tier": "Tier-3", "suggested_stake": 10.0},
    ]
    filtered = filter_conviction_plays(plays, market_line=1.5, side="Over", top_n=2)
    assert [play["player"] for play in filtered] == ["D", "A"]
    assert all(play["line"] == 1.5 for play in filtered)
    assert all(play["rec"] == "Over" for play in filtered)


def test_build_betting_intel_export_applies_filters() -> None:
    conviction = pd.DataFrame(
        [
            {
                "player_name": "Player One",
                "market": "batter_total_bases",
                "model_value": 3.0,
                "market_line": 1.5,
                "probability_pct": 90.0,
                "edge_pct": 55.0,
                "recommendation": "Over",
                "ev_per_unit": 2.5,
                "confidence_tier": "Tier-1 High Conviction",
                "kelly_fraction": 0.02,
                "suggested_stake": 20.0,
            },
            {
                "player_name": "Player Two",
                "market": "batter_total_bases",
                "model_value": 2.0,
                "market_line": 1.0,
                "probability_pct": 85.0,
                "edge_pct": 80.0,
                "recommendation": "Over",
                "ev_per_unit": 3.0,
                "confidence_tier": "Tier-1 High Conviction",
                "kelly_fraction": 0.02,
                "suggested_stake": 20.0,
            },
        ]
    )
    frames: SlateFrames = {
        "slate_games": pd.DataFrame([{"game_date": "2026-07-03"}]),
        "lineups": pd.DataFrame(),
        "vegas_totals": pd.DataFrame(),
        "player_baselines": pd.DataFrame(),
        "matchup_splits": pd.DataFrame(),
        "pitcher_tendencies": pd.DataFrame(),
        "team_pitching": pd.DataFrame(),
    }
    context = SlateContext(
        player_games=pd.DataFrame(),
        games=pd.DataFrame([{"game_date": "2026-07-03"}]),
        fallback_counts={},
    )
    result = SlateRunResult(
        frames=frames,
        context=context,
        projected=pd.DataFrame(),
        pitcher_outs=pd.DataFrame(),
        conviction=conviction,
        meta={"data_health": {"warning_count": 0, "warnings": []}},
    )

    intel = build_betting_intel_export(
        result,
        slate_date=date(2026, 7, 3),
        market_line=1.5,
        side="Over",
        top_n=10,
    )

    assert intel["slate_date"] == "2026-07-03"
    assert len(intel["conviction_plays"]) == 1
    assert intel["conviction_plays"][0]["player"] == "Player One"
    assert intel["summary_stats"]["play_count"] == 1
    assert intel["summary_stats"]["total_suggested_exposure"] == 20.0
    assert intel["summary_stats"]["tier_counts"]["Tier-1 High Conviction"] == 1
