from __future__ import annotations

import numpy as np
import pandas as pd

from baseball_props.config import (
    TEMP_BASELINE_F,
    TEMP_RUNS_FACTOR_PER_DEG,
    WIND_OUT_FACTOR_PER_MPH,
)
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_OUT_WIND_DIRECTIONS = {"out_to_lf", "out_to_cf", "out_to_rf", "out"}


def compute_environment_multiplier(df: pd.DataFrame) -> pd.Series:
    """
    Compute a combined park + weather multiplier for rate stats.

    Phase 1 stub: park_factor_runs * temp adjustment * wind adjustment.
    """
    if "park_factor_runs" not in df.columns:
        logger.warning("No park_factor_runs column; returning neutral multiplier 1.0")
        return pd.Series(1.0, index=df.index)

    park = df["park_factor_runs"].fillna(1.0)

    temp = df.get("temp_f", pd.Series(TEMP_BASELINE_F, index=df.index)).fillna(TEMP_BASELINE_F)
    temp_adj = 1.0 + (temp - TEMP_BASELINE_F) * TEMP_RUNS_FACTOR_PER_DEG

    wind_mph = df.get("wind_mph", pd.Series(0.0, index=df.index)).fillna(0.0)
    wind_dir = df.get("wind_dir", pd.Series("", index=df.index)).fillna("")
    is_out = wind_dir.isin(_OUT_WIND_DIRECTIONS)
    wind_adj = np.where(is_out, 1.0 + wind_mph * WIND_OUT_FACTOR_PER_MPH, 1.0)

    multiplier = park * temp_adj * wind_adj
    logger.debug("Environment multiplier range: %.3f – %.3f", multiplier.min(), multiplier.max())
    return pd.Series(multiplier, index=df.index)


def apply_environment_to_rates(
    df: pd.DataFrame,
    rate_columns: list[str],
) -> pd.DataFrame:
    """Apply environment multiplier to projected rate columns (vectorized)."""
    result = df.copy()
    env_mult = compute_environment_multiplier(result)
    for col in rate_columns:
        if col in result.columns:
            result[col] = result[col] * env_mult
    result["env_multiplier"] = env_mult
    return result
