"""Tests for moneyline defensive caps and team streak bonus."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from momentum import apply_team_streak_bonus, team_win_streak
from run_odds_slate import _passes_moneyline_guardrails


def test_moneyline_probability_floor() -> None:
    assert not _passes_moneyline_guardrails(0.39, 150)
    assert _passes_moneyline_guardrails(0.40, 150)


def test_moneyline_odds_cap() -> None:
    assert not _passes_moneyline_guardrails(0.45, 161)
    assert _passes_moneyline_guardrails(0.45, 160)


def test_team_streak_bonus_applies() -> None:
    with patch("momentum.team_win_streak", return_value=3):
        boosted = apply_team_streak_bonus(0.50, 147, date(2026, 7, 9))
    assert boosted == 0.515


def test_team_streak_bonus_skipped_when_cold() -> None:
    with patch("momentum.team_win_streak", return_value=2):
        unchanged = apply_team_streak_bonus(0.50, 147, date(2026, 7, 9))
    assert unchanged == 0.50


def test_team_win_streak_counts_from_final_games() -> None:
    games = [
        {
            "game_type": "R",
            "status": "Final",
            "game_date": "2026-07-01",
            "home_id": 147,
            "away_id": 111,
            "home_score": 5,
            "away_score": 2,
        },
        {
            "game_type": "R",
            "status": "Final",
            "game_date": "2026-07-02",
            "home_id": 111,
            "away_id": 147,
            "home_score": 1,
            "away_score": 4,
        },
        {
            "game_type": "R",
            "status": "Final",
            "game_date": "2026-07-03",
            "home_id": 147,
            "away_id": 111,
            "home_score": 3,
            "away_score": 2,
        },
    ]
    team_win_streak.cache_clear()
    with patch("statsapi.schedule", return_value=games):
        assert team_win_streak(147, "2026-07-04") == 3
