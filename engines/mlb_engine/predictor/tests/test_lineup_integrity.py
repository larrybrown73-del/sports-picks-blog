"""Tests for lineup integrity helpers (delegating / compatibility wrappers)."""

from __future__ import annotations

from unittest.mock import patch

from config import MISSING_STAR_BAT_PENALTY
from hitter_discipline import LineupBatter
from lineup_integrity import (
    apply_lineup_integrity_to_runs,
    missing_star_bat_scalar,
    missing_star_batter_ids,
)


def test_missing_star_on_il() -> None:
    with (
        patch("lineup_integrity.team_star_power_batter_ids", return_value=(592450, 660271)),
        patch("lineup_integrity.team_injured_player_ids", return_value=frozenset({592450})),
    ):
        missing = missing_star_batter_ids(team_id=147, season=2026, lineup=[])
    assert missing == [592450]


def test_missing_star_scratched_from_posted_lineup() -> None:
    lineup = [
        LineupBatter(player_id=i, lineup_slot=slot)
        for slot, i in enumerate(range(1, 10), start=1)
    ]
    with (
        patch("lineup_integrity.team_star_power_batter_ids", return_value=(99, 2)),
        patch("lineup_integrity.team_injured_player_ids", return_value=frozenset()),
    ):
        missing = missing_star_batter_ids(team_id=147, season=2026, lineup=lineup)
    assert missing == [99]


def test_missing_star_penalty_stacks() -> None:
    with (
        patch("lineup_integrity.team_star_power_batter_ids", return_value=(1, 2, 3)),
        patch("lineup_integrity.team_injured_player_ids", return_value=frozenset({1, 2})),
    ):
        scalar, tags = missing_star_bat_scalar(
            team_id=147,
            season=2026,
            lineup=None,
            label="home_offense",
        )
    assert scalar == MISSING_STAR_BAT_PENALTY**2
    assert any("missing_star_bat:1" in tag for tag in tags)
    assert any("missing_star_bat:2" in tag for tag in tags)


def test_apply_lineup_integrity_haircuts_runs() -> None:
    with patch(
        "lineup_integrity.missing_star_bat_scalar",
        return_value=(MISSING_STAR_BAT_PENALTY, ["home_offense:missing_star_bat:1:0.92"]),
    ):
        runs, tags = apply_lineup_integrity_to_runs(
            5.0,
            team_id=147,
            season=2026,
            lineup=None,
            label="home_offense",
        )
    assert runs == 5.0 * MISSING_STAR_BAT_PENALTY
    assert tags
