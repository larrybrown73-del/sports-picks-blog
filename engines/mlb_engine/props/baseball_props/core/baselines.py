from __future__ import annotations

import pandas as pd

from baseball_props.config import DEFAULT_ROLLING_WEIGHTS, RollingWeights
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


def _coalesce_with_fallback(
    df: pd.DataFrame,
    target_col: str,
    primary_col: str,
    fallback_col: str,
    metric: str,
) -> pd.Series:
    """Use primary when present, else fallback; log each fallback."""
    primary = df[primary_col]
    fallback = df[fallback_col]
    missing = primary.isna()
    if missing.any():
        count = int(missing.sum())
        logger.warning(
            "Baseline fallback for %s: %d players missing %s, using %s",
            metric,
            count,
            primary_col,
            fallback_col,
        )
    return primary.fillna(fallback)


def build_effective_baselines(
    baselines: pd.DataFrame,
    metrics: list[str],
    weights: RollingWeights | None = None,
) -> pd.DataFrame:
    """
    Vectorized weighted blend: w14*roll14 + w30*roll30 + w_season*season.

    Fallback chain per metric:
      roll14 -> roll30 -> season (with logging)
    """
    w = weights or DEFAULT_ROLLING_WEIGHTS
    result = baselines[["player_id"]].copy()

    for metric in metrics:
        roll14_col = f"roll14_{metric}"
        roll30_col = f"roll30_{metric}"
        season_col = f"season_{metric}"
        effective_col = f"effective_{metric}"

        roll30_filled = _coalesce_with_fallback(
            baselines, effective_col, roll30_col, season_col, metric
        )
        roll14_filled = baselines[roll14_col].fillna(roll30_filled)

        missing_roll14 = baselines[roll14_col].isna()
        missing_roll30 = baselines[roll30_col].isna() & baselines[roll14_col].isna()
        if missing_roll14.any():
            logger.warning(
                "Baseline fallback for %s: %d players missing roll14, using roll30/season",
                metric,
                int(missing_roll14.sum()),
            )
        if missing_roll30.any():
            logger.warning(
                "Baseline fallback for %s: %d players missing roll14+roll30, using season",
                metric,
                int(missing_roll30.sum()),
            )

        result[effective_col] = (
            w.w14 * roll14_filled + w.w30 * roll30_filled + w.w_season * baselines[season_col]
        )

    return result
