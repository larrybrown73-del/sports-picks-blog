"""Tests for predictor lineup discipline scalars."""

from __future__ import annotations

from hitter_discipline import (
    BatterDisciplineProfile,
    LineupBatter,
    lineup_offense_scalar,
    lineup_slot_scalar,
)
from config import BOTTOM_ORDER_PENALTY, DISCIPLINE_BONUS, PREMIUM_SLOT_SCALAR


def test_lineup_slot_scalar_tiers() -> None:
    assert lineup_slot_scalar(1) == PREMIUM_SLOT_SCALAR
    assert lineup_slot_scalar(6) == 1.0
    assert lineup_slot_scalar(9) == BOTTOM_ORDER_PENALTY


def test_lineup_offense_scalar_weighted(monkeypatch) -> None:
    profiles = {
        1: BatterDisciplineProfile(1, k_pct=18.0, bb_pct=14.0),
        2: BatterDisciplineProfile(2, k_pct=30.0, bb_pct=5.0),
    }

    def _fake_fetch(player_id: int, season: int) -> BatterDisciplineProfile:
        return profiles[player_id]

    monkeypatch.setattr("hitter_discipline.fetch_batter_discipline_profile", _fake_fetch)

    lineup = [
        LineupBatter(player_id=1, lineup_slot=1),
        LineupBatter(player_id=2, lineup_slot=9),
    ]
    scalar, tags = lineup_offense_scalar(lineup, 2026)
    assert scalar > 0
    assert any("discipline" in tag for tag in tags)
    assert any("erratic" in tag for tag in tags)
    assert scalar < DISCIPLINE_BONUS
