from __future__ import annotations

import pandas as pd
import pytest

from baseball_props.core.baselines import build_effective_baselines
from baseball_props.config import RollingWeights


@pytest.fixture
def sample_baselines() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "H001",
                "season_woba": 0.320,
                "roll14_woba": 0.330,
                "roll30_woba": 0.325,
                "season_iso": 0.150,
                "roll14_iso": 0.155,
                "roll30_iso": 0.152,
                "season_k_pct": 0.200,
                "roll14_k_pct": 0.195,
                "roll30_k_pct": 0.198,
                "season_bb_pct": 0.080,
                "roll14_bb_pct": 0.082,
                "roll30_bb_pct": 0.081,
            },
            {
                "player_id": "H002",
                "season_woba": 0.290,
                "roll14_woba": None,
                "roll30_woba": 0.300,
                "season_iso": 0.130,
                "roll14_iso": None,
                "roll30_iso": 0.135,
                "season_k_pct": 0.250,
                "roll14_k_pct": None,
                "roll30_k_pct": None,
                "season_bb_pct": 0.070,
                "roll14_bb_pct": None,
                "roll30_bb_pct": None,
            },
        ]
    )


class TestBuildEffectiveBaselines:
    def test_weighted_blend(self, sample_baselines: pd.DataFrame) -> None:
        weights = RollingWeights(w14=0.50, w30=0.30, w_season=0.20)
        result = build_effective_baselines(sample_baselines, ["woba"], weights=weights)

        expected = 0.50 * 0.330 + 0.30 * 0.325 + 0.20 * 0.320
        assert result.loc[0, "effective_woba"] == pytest.approx(expected)

    def test_roll14_fallback_to_roll30(self, sample_baselines: pd.DataFrame) -> None:
        weights = RollingWeights(w14=0.45, w30=0.35, w_season=0.20)
        result = build_effective_baselines(sample_baselines, ["woba"], weights=weights)

        roll14_filled = 0.300
        roll30_filled = 0.300
        season = 0.290
        expected = 0.45 * roll14_filled + 0.35 * roll30_filled + 0.20 * season
        assert result.loc[1, "effective_woba"] == pytest.approx(expected)

    def test_full_fallback_to_season(self, sample_baselines: pd.DataFrame) -> None:
        weights = RollingWeights(w14=0.45, w30=0.35, w_season=0.20)
        result = build_effective_baselines(sample_baselines, ["k_pct"], weights=weights)

        season = 0.250
        expected = 0.45 * season + 0.35 * season + 0.20 * season
        assert result.loc[1, "effective_k_pct"] == pytest.approx(expected)

    def test_output_columns(self, sample_baselines: pd.DataFrame) -> None:
        metrics = ["woba", "iso", "k_pct", "bb_pct"]
        result = build_effective_baselines(sample_baselines, metrics)
        for metric in metrics:
            assert f"effective_{metric}" in result.columns
        assert list(result["player_id"]) == ["H001", "H002"]
