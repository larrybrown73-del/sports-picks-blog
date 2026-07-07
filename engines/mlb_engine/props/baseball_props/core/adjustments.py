from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

from baseball_props.config import STRICT_MODE
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


def _as_array(value: pd.Series | np.ndarray | float) -> np.ndarray:
    if isinstance(value, pd.Series):
        return value.to_numpy(dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float)
    return np.asarray(value, dtype=float)


def _restore_type(
    result: np.ndarray,
    template: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    if isinstance(template, pd.Series):
        return pd.Series(result, index=template.index, name=template.name)
    if isinstance(template, np.ndarray):
        return result
    return float(result.reshape(-1)[0])


@overload
def calculate_projected_rate(
    player_rate: float,
    opponent_rate_allowed: float,
    league_avg: float,
    *,
    clip_min: float | None = ...,
    clip_max: float | None = ...,
) -> float: ...


@overload
def calculate_projected_rate(
    player_rate: pd.Series,
    opponent_rate_allowed: pd.Series | float,
    league_avg: pd.Series | float,
    *,
    clip_min: float | None = ...,
    clip_max: float | None = ...,
) -> pd.Series: ...


@overload
def calculate_projected_rate(
    player_rate: np.ndarray,
    opponent_rate_allowed: np.ndarray | float,
    league_avg: np.ndarray | float,
    *,
    clip_min: float | None = ...,
    clip_max: float | None = ...,
) -> np.ndarray: ...


def calculate_projected_rate(
    player_rate: pd.Series | np.ndarray | float,
    opponent_rate_allowed: pd.Series | np.ndarray | float,
    league_avg: pd.Series | np.ndarray | float,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> pd.Series | np.ndarray | float:
    """
    Odds-ratio style rate projection.

    Formula: (player_rate * opponent_rate_allowed) / league_avg
    """
    player = _as_array(player_rate)
    opponent = _as_array(opponent_rate_allowed)
    league = _as_array(league_avg)

    zero_league = league == 0
    if zero_league.any():
        msg = f"league_avg contains zero values ({int(zero_league.sum())} rows)"
        if STRICT_MODE:
            raise ValueError(msg)
        logger.error("%s; falling back to player_rate for those rows", msg)

    projected = np.where(
        zero_league,
        player,
        (player * opponent) / league,
    )

    if clip_min is not None or clip_max is not None:
        lo = clip_min if clip_min is not None else -np.inf
        hi = clip_max if clip_max is not None else np.inf
        projected = np.clip(projected, lo, hi)

    return _restore_type(projected, player_rate)


def rate_to_log_odds(p: np.ndarray | pd.Series | float, eps: float = _EPS) -> np.ndarray:
    """Convert probability/rate in (0, 1) to log-odds."""
    arr = _as_array(p)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def log_odds_to_rate(logit_p: np.ndarray | pd.Series | float) -> np.ndarray:
    """Convert log-odds back to probability/rate."""
    arr = _as_array(logit_p)
    return 1.0 / (1.0 + np.exp(-arr))


def calculate_projected_probability(
    player_p: pd.Series | np.ndarray | float,
    opponent_p_allowed: pd.Series | np.ndarray | float,
    league_p: float,
) -> pd.Series | np.ndarray | float:
    """
    Log-odds form of the multiplicative odds ratio for bounded rates.

    Equivalent to calculate_projected_rate for probabilities in (0, 1).
    """
    if league_p <= 0 or league_p >= 1:
        raise ValueError(f"league_p must be in (0, 1), got {league_p}")

    player = _as_array(player_p)
    opponent = _as_array(opponent_p_allowed)

    player_or = player / (1.0 - np.clip(player, _EPS, 1.0 - _EPS))
    opponent_or = opponent / (1.0 - np.clip(opponent, _EPS, 1.0 - _EPS))
    league_or = league_p / (1.0 - league_p)

    projected_or = (player_or * opponent_or) / league_or
    projected = projected_or / (1.0 + projected_or)

    return _restore_type(projected, player_p)
