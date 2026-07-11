"""
Vectorized (NumPy/Pandas) backtesting layer for the soccer micro-market EV engine.

WHY THIS FILE EXISTS SEPARATELY FROM ev_engine_core.py:

`ev_engine_core.py` is the source-of-truth, unit-tested SCALAR implementation
(one MarketLeg + one probability estimate at a time). It's the right shape
for live/production grading, where you're evaluating a handful of legs for
today's slate. It is the wrong shape for backtesting: replaying a season of
historical closing lines means grading tens of thousands of rows, and a
Python-level `for` loop over `evaluate_leg()` for each one is needlessly slow.

This module re-expresses the IDENTICAL math from ev_engine_core.py using
NumPy array operations over a Pandas DataFrame, so a full historical dataset
can be graded in one vectorized pass. To keep the two implementations from
silently drifting apart, `tests/test_backtest.py` includes a cross-check that
runs the same inputs through both `ev_engine_core.evaluate_leg` (scalar) and
`compute_ev_columns` (vectorized) and asserts bit-for-bit-equivalent results.

ARCHITECTURE CONTRACT (mirrors ev_engine_core.py -- see that module's
top-of-file docstring for the full rationale):
  - Odds-side data (`legs_to_frame`) and probability-side data (an externally
    supplied `true_probability` Series) are built and validated completely
    independently. They are only ever combined inside `compute_ev_columns`.
  - No probability/edge/EV column is rounded. Everything stays float64 until
    a caller explicitly formats it for a report or UI.
  - `line` stays a float column; it is never cast to an integer dtype.
  - Rows with missing/NaN odds are dropped outright (`_drop_rows_with_missing_odds`),
    never filled/interpolated/imputed.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ev_engine_core import MarketLeg

# Columns describing the sportsbook price itself. Never touched by anything
# on the probability side until `compute_ev_columns`.
ODDS_COLUMNS = (
    "game_id",
    "market_type",
    "market_family",
    "selection",
    "entity_name",
    "team",
    "line",
    "side",
    "odds",
    "odds_format",
    "sportsbook",
    "sample_size",
    "volatility",
)


def legs_to_frame(legs: Iterable[MarketLeg]) -> pd.DataFrame:
    """
    Convert an iterable of MarketLeg records (odds-side only) into a flat
    DataFrame suitable for vectorized grading.

    Deliberately carries ONLY sportsbook-observed fields -- no model output
    is merged in here. That makes it structurally impossible to accidentally
    leak a "true probability" into odds-side calculations before the EV step,
    the same separation `MarketLeg` vs. `ProbabilityEstimate` enforces in the
    scalar engine.
    """

    rows = [
        {
            "game_id": leg.game_id,
            "market_type": leg.market_type,
            "market_family": leg.market_family,
            "selection": leg.selection,
            "entity_name": leg.entity_name,
            "team": leg.team,
            "line": leg.line,  # kept as float64 via the frame's dtype inference -- see ARCHITECTURE CONTRACT
            "side": leg.side,
            "odds": leg.odds,
            "odds_format": leg.odds_format,
            "sportsbook": leg.sportsbook,
            "sample_size": leg.sample_size,
            "volatility": leg.volatility,
        }
        for leg in legs
    ]
    frame = pd.DataFrame(rows, columns=ODDS_COLUMNS)
    if not frame.empty:
        frame["line"] = frame["line"].astype("float64")
        frame["odds"] = frame["odds"].astype("float64")
    frame = _drop_rows_with_missing_odds(frame)
    return frame.reset_index(drop=True)


def _drop_rows_with_missing_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Ruthless missing-data policy, vectorized: any row with a null/NaN price
    is unusable and must be discarded outright -- never imputed with a
    market average, a prior close, or a neighboring book's number. Returning
    a new frame (rather than a silent in-place `.dropna()` buried elsewhere)
    keeps this an explicit, auditable pipeline step.
    """

    if frame.empty:
        return frame
    return frame.loc[frame["odds"].notna()]


def _decimal_odds_array(odds: np.ndarray, odds_format: np.ndarray) -> np.ndarray:
    """
    Vectorized American/decimal -> decimal-odds conversion. Mirrors
    `ev_engine_core.odds_to_decimal` / `american_to_decimal` element-for-element;
    see `tests/test_backtest.py::test_decimal_odds_array_matches_scalar_engine`
    for the cross-check that keeps this in sync with the scalar functions.
    """

    odds = odds.astype("float64")
    is_decimal = odds_format == "decimal"
    american_positive = odds > 0
    american_decimal = np.where(american_positive, 1.0 + odds / 100.0, 1.0 + 100.0 / np.abs(odds))
    return np.where(is_decimal, odds, american_decimal)


def implied_probability_column(frame: pd.DataFrame) -> pd.Series:
    """
    Vectorized, single-outcome implied probability. Vig-inclusive by design
    (see the module-level vig note in ev_engine_core.py) -- this is what a
    specific price implies, not a de-vigged fair probability.
    """

    decimal_odds = _decimal_odds_array(frame["odds"].to_numpy(), frame["odds_format"].to_numpy())
    return pd.Series(1.0 / decimal_odds, index=frame.index, name="implied_probability")


def devig_group(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    probability_col: str = "implied_probability",
) -> pd.Series:
    """
    Proportional (multiplicative) de-vig, vectorized across every market
    grouping in the frame at once (e.g. group by game_id + market_type +
    sportsbook + line to isolate exactly one book's one market snapshot).

    Scales each outcome's implied probability down by its group's total
    overround so the group sums to 1.0. This is an analysis/calibration
    utility ONLY -- it must never be substituted into `compute_ev_columns`,
    which has to grade against the real, vig-included price.
    """

    if probability_col not in frame.columns:
        raise KeyError(f"Frame is missing required column: {probability_col!r}")
    totals = frame.groupby(list(group_cols))[probability_col].transform("sum")
    return frame[probability_col] / totals


def compute_ev_columns(
    frame: pd.DataFrame,
    true_probability: pd.Series,
    *,
    stake: float = 1.0,
) -> pd.DataFrame:
    """
    Attach EV columns to `frame` given an externally supplied true-probability
    Series (the vectorized output of your projection model), aligned to
    `frame` by index. This is the ONLY function in this module where the
    odds side and the probability side are allowed to touch.

    EV = (True_Win_Probability * Potential_Profit)
       - (True_Loss_Probability * Wager_Amount)

    No column produced here is rounded; everything stays float64. Round only
    when formatting a report or UI payload.
    """

    if not true_probability.index.equals(frame.index):
        raise ValueError(
            "true_probability index must exactly match frame index -- a "
            "silent misalignment here would pair the wrong model output "
            "with the wrong market price and produce a meaningless EV."
        )
    if stake <= 0:
        raise ValueError("Stake must be positive")

    result = frame.copy()
    decimal_odds = _decimal_odds_array(result["odds"].to_numpy(), result["odds_format"].to_numpy())

    # Clamp is a safety net against a malformed model output (e.g. a bug
    # producing 1.4 or -0.1), not a rounding step -- the underlying float
    # precision is otherwise left completely untouched.
    prob = np.clip(true_probability.to_numpy(dtype="float64"), 0.0, 1.0)
    true_loss_probability = 1.0 - prob

    potential_profit = stake * (decimal_odds - 1.0)
    ev = prob * potential_profit - true_loss_probability * stake

    result["decimal_odds"] = decimal_odds
    result["implied_probability"] = 1.0 / decimal_odds
    result["true_probability"] = prob
    result["edge"] = prob - result["implied_probability"]
    result["potential_profit"] = potential_profit
    result["ev"] = ev
    result["ev_per_unit"] = result["ev"] / stake
    result["positive_ev"] = result["ev"] > 0.0
    return result


def backtest_summary(graded: pd.DataFrame) -> pd.Series:
    """
    Minimal flat rollup over a graded (post `compute_ev_columns`) frame.

    Deliberately does not attempt hit-rate / ROI, since that requires
    joining realized match results onto each leg -- a separate, dedicated
    historical-results pipeline, not something this generic summary should
    guess at.
    """

    positive = graded.loc[graded["positive_ev"]]
    return pd.Series(
        {
            "total_legs_considered": len(graded),
            "positive_ev_legs": len(positive),
            "avg_edge_positive_ev": positive["edge"].mean() if len(positive) else float("nan"),
            "avg_ev_per_unit_positive_ev": positive["ev_per_unit"].mean() if len(positive) else float("nan"),
            "sum_ev_per_unit_positive_ev": positive["ev_per_unit"].sum() if len(positive) else 0.0,
        }
    )
