from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from baseball_props.data.mock_slate import build_mock_slate
from baseball_props.pipeline.slate_run import run_slate


def test_run_slate_mock_completes_full_stack() -> None:
    result = run_slate(source="mock")
    assert not result.projected.empty
    assert not result.pitcher_outs.empty
    assert "proj_total_bases" in result.projected.columns
    assert (result.pitcher_outs["proj_outs"] <= 27.0).all()
    assert result.batter_edge_sheet is not None
    assert not result.batter_edge_sheet.empty
    assert result.pitcher_edge_sheet is not None
    assert not result.pitcher_edge_sheet.empty
    assert result.conviction is None
    assert result.conviction_message is not None
    assert "mock" in result.conviction_message.lower()
    assert result.parlay_tickets is not None


def test_run_slate_live_path_with_props_mocked() -> None:
    frames = build_mock_slate()
    game_id = "abc123event456789012345678901234"
    frames["slate_games"] = frames["slate_games"].iloc[:1].copy()
    frames["slate_games"]["game_id"] = game_id
    frames["lineups"] = frames["lineups"][frames["lineups"]["game_id"] == "G001"].copy()
    frames["lineups"]["game_id"] = game_id
    frames["vegas_totals"] = frames["vegas_totals"].iloc[:1].copy()
    frames["vegas_totals"]["game_id"] = game_id
    frames["odds_event_ids"] = [game_id]

    prop_lines = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Over",
                "line": 1.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Under",
                "line": 1.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
        ]
    )

    from baseball_props.analysis.guardrails import HitsPropEvaluation

    def _playable_hits_eval(*_args, **_kwargs) -> HitsPropEvaluation:
        return HitsPropEvaluation(
            verdict="Play",
            base_proj_hits=1.5,
            adjusted_prob_over=0.62,
            edge_pct=6.5,
            recommendation="Over",
        )

    with (
        patch("baseball_props.pipeline.slate_run.load_slate_frames", return_value=frames),
        patch("baseball_props.pipeline.slate_run.fetch_active_injuries", return_value={}),
        patch("baseball_props.pipeline.slate_run.fetch_all_player_props", return_value=prop_lines),
        patch(
            "baseball_props.analysis.guardrails.compute_contact_profile",
            return_value={"k_pct": 0.15, "contact_pct": 0.82, "babip": 0.30, "pa": 30.0},
        ),
        patch(
            "baseball_props.analysis.guardrails.evaluate_hits_prop",
            side_effect=_playable_hits_eval,
        ),
    ):
        result = run_slate(source="live")

    assert result.conviction is not None
    assert not result.conviction.empty
    assert "edge_pct" in result.conviction.columns
    assert result.batter_edge_sheet is not None
    assert result.pitcher_edge_sheet is not None
    assert "verdict" in result.batter_edge_sheet.columns
    assert result.parlay_tickets is not None
