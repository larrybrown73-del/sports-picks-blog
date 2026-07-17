"""Tests for completed-season schedule baseline caching."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import baseball_data


def test_load_schedule_baseline_reads_local_json(tmp_path: Path) -> None:
    baseline = tmp_path / "mlb_schedule_2024.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "game_id": 1,
                    "game_date": "2024-04-01",
                    "game_type": "R",
                    "status": "Final",
                    "home_id": 10,
                    "away_id": 20,
                    "home_name": "Home",
                    "away_name": "Away",
                    "home_score": 5,
                    "away_score": 3,
                }
            ]
        ),
        encoding="utf-8",
    )

    with patch.object(baseball_data, "SCHEDULE_BASELINES_DIR", tmp_path):
        games = baseball_data._load_schedule_baseline(2024)

    assert games is not None
    assert len(games) == 1
    assert games[0]["game_id"] == 1


def test_load_or_fetch_prefers_baseline_over_live_api(tmp_path: Path) -> None:
    baseline = tmp_path / "mlb_schedule_2024.json"
    baseline.write_text(
        json.dumps(
            [
                {
                    "game_id": 99,
                    "game_date": "2024-05-01",
                    "game_type": "R",
                    "status": "Final",
                    "home_id": 1,
                    "away_id": 2,
                    "home_name": "A",
                    "away_name": "B",
                    "home_score": 2,
                    "away_score": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    with (
        patch.object(baseball_data, "SCHEDULE_BASELINES_DIR", tmp_path),
        patch.object(baseball_data, "_fetch_schedule_chunked") as mock_fetch,
    ):
        games = baseball_data._load_or_fetch_season_schedule(2024)

    assert games[0]["game_id"] == 99
    mock_fetch.assert_not_called()
