"""Tests for hitter discipline scalars."""

from __future__ import annotations

from baseball_props.analysis.hitter_discipline import (
    BatterDisciplineProfile,
    apply_hitter_discipline_to_projection,
    is_elite_discipline,
    is_erratic_swinger,
    lineup_slot_prob_scalar,
)
from baseball_props.config import (
    BOTTOM_ORDER_PENALTY,
    DISCIPLINE_BONUS,
    ERRATIC_SWINGER_PENALTY,
    PREMIUM_SLOT_SCALAR,
)


def test_premium_slot_bonus() -> None:
    scalar, tag = lineup_slot_prob_scalar(2)
    assert tag == "premium_slot"
    assert scalar == PREMIUM_SLOT_SCALAR


def test_bottom_order_penalty() -> None:
    scalar, tag = lineup_slot_prob_scalar(9)
    assert tag == "bottom_order_penalty"
    assert scalar == BOTTOM_ORDER_PENALTY


def test_middle_order_neutral() -> None:
    scalar, tag = lineup_slot_prob_scalar(6)
    assert tag is None
    assert scalar == 1.0


def test_discipline_bonus_on_projection() -> None:
    profile = BatterDisciplineProfile("1", k_pct=18.0, bb_pct=14.0)
    adjustments: dict[str, float] = {}
    proj = apply_hitter_discipline_to_projection(2.0, profile, adjustments)
    assert proj == 2.0 * DISCIPLINE_BONUS
    assert adjustments["discipline_bonus"] == DISCIPLINE_BONUS
    assert is_elite_discipline(profile)


def test_erratic_swinger_penalty_on_projection() -> None:
    profile = BatterDisciplineProfile("2", k_pct=30.0, bb_pct=5.0)
    adjustments: dict[str, float] = {}
    proj = apply_hitter_discipline_to_projection(2.0, profile, adjustments)
    assert proj == 2.0 * ERRATIC_SWINGER_PENALTY
    assert adjustments["erratic_swinger_penalty"] == ERRATIC_SWINGER_PENALTY
    assert is_erratic_swinger(profile)
