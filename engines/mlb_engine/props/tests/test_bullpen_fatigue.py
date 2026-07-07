from __future__ import annotations

from unittest.mock import patch

from baseball_props.data.bullpen_fatigue import (
    BullpenFatigueResult,
    _arm_fatigue_score,
    compute_bullpen_fatigue_score,
)


def test_arm_fatigue_score_heavy_workload() -> None:
    score = _arm_fatigue_score(pitches=90, active_days=3)
    assert score >= 0.65


def test_fatigue_status_thresholds() -> None:
    from baseball_props.config import TB_BULLPEN_FATIGUE_THRESHOLD

    result = BullpenFatigueResult(
        score=TB_BULLPEN_FATIGUE_THRESHOLD,
        status="Fatigued",
        reliever_count=3,
    )
    assert result.status == "Fatigued"


def test_non_numeric_team_returns_neutral() -> None:
    result = compute_bullpen_fatigue_score("NYY")
    assert result.status == "Moderate"
    assert result.score == 0.35


def test_bullpen_fatigue_with_mocked_workload() -> None:
    with (
        patch(
            "baseball_props.data.bullpen_fatigue._fetch_active_pitcher_ids",
            return_value=["123", "456"],
        ),
        patch(
            "baseball_props.data.bullpen_fatigue._pitcher_workload_last_days",
            side_effect=[(85, 3), (10, 1)],
        ),
    ):
        compute_bullpen_fatigue_score.cache_clear()
        result = compute_bullpen_fatigue_score("147")
    assert result.reliever_count == 2
    assert result.score >= 0.65
    assert result.status == "Fatigued"
