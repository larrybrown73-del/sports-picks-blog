from __future__ import annotations

import math
from typing import Any

import pandas as pd

from baseball_props.config import DEFAULT_BANKROLL
from baseball_props.market.calculations import (
    MARKET_METRIC_COLUMNS,
    apply_wager_metrics_to_row,
    compute_wager_metrics,
    empty_wager_columns,
)

__all__ = [
    "MARKET_METRIC_COLUMNS",
    "american_to_decimal",
    "compute_ev",
    "confidence_tier",
    "confidence_score",
    "enrich_edge_sheet_with_market_metrics",
    "enrich_row_market_metrics",
    "fractional_kelly",
    "full_kelly_fraction",
    "odds_for_recommendation",
    "suggested_stake",
]


def american_to_decimal(odds: float) -> float | None:
    """Convert American odds to decimal payout multiplier."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return None
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    from baseball_props.market.calculations import american_to_decimal as _calc_decimal

    return _calc_decimal(value)


def compute_ev(true_prob: float, decimal_odds: float) -> float | None:
    """Expected value per unit staked."""
    if true_prob is None or decimal_odds is None:
        return None
    try:
        p = float(true_prob)
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return None
    if math.isnan(p) or math.isnan(d) or d <= 1.0:
        return None
    if p < 0.0 or p > 1.0:
        return None
    return (p * d) - (1.0 - p)


def confidence_tier(edge_pct: float | None) -> str | None:
    from baseball_props.market.calculations import confidence_tier_from_edge

    if edge_pct is None or (isinstance(edge_pct, float) and math.isnan(edge_pct)):
        return None
    try:
        edge = float(edge_pct)
    except (TypeError, ValueError):
        return None
    label = confidence_tier_from_edge(edge)
    return None if label == "No Bet" else label


def full_kelly_fraction(true_prob: float, decimal_odds: float) -> float | None:
    """Full Kelly fraction: f* = (p*d - 1) / (d - 1)."""
    if true_prob is None or decimal_odds is None:
        return None
    try:
        p = float(true_prob)
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return None
    if math.isnan(p) or math.isnan(d) or d <= 1.0:
        return None
    if p <= 0.0 or p >= 1.0:
        return None
    numerator = (p * d) - 1.0
    denominator = d - 1.0
    if denominator <= 0:
        return None
    return max(0.0, numerator / denominator)


def fractional_kelly(
    true_prob: float,
    decimal_odds: float,
    *,
    fraction: float | None = None,
    max_stake_pct: float | None = None,
) -> float | None:
    """Kelly fraction capped at max stake percent of bankroll."""
    from baseball_props.config import KELLY_FRACTION, KELLY_MAX_STAKE_PCT

    if fraction is None:
        fraction = KELLY_FRACTION
    if max_stake_pct is None:
        max_stake_pct = KELLY_MAX_STAKE_PCT

    full = full_kelly_fraction(true_prob, decimal_odds)
    if full is None:
        return None
    scaled = full * fraction
    return min(scaled, max_stake_pct) if scaled > 0 else None


def suggested_stake(
    kelly_fraction: float | None,
    bankroll: float = DEFAULT_BANKROLL,
) -> float | None:
    from baseball_props.market.calculations import suggested_stake_from_kelly

    if kelly_fraction is None or (isinstance(kelly_fraction, float) and math.isnan(kelly_fraction)):
        return None
    try:
        frac = float(kelly_fraction)
    except (TypeError, ValueError):
        return None
    if frac <= 0:
        return None
    return suggested_stake_from_kelly(frac, bankroll=bankroll)


def odds_for_recommendation(
    recommendation: str,
    over_odds: float | None,
    under_odds: float | None,
) -> float | None:
    rec = str(recommendation or "").strip()
    if rec == "Over" and over_odds is not None:
        return float(over_odds)
    if rec == "Under" and under_odds is not None:
        return float(under_odds)
    return None


def confidence_score(true_prob: float | None, edge_pct: float | None) -> int | None:
    from baseball_props.market.calculations import confidence_score_from_edge_and_prob

    if true_prob is None or edge_pct is None:
        return None
    try:
        p = float(true_prob)
        edge = float(edge_pct)
    except (TypeError, ValueError):
        return None
    return confidence_score_from_edge_and_prob(edge, p)


def enrich_row_market_metrics(
    row: dict[str, Any],
    *,
    over_odds: float | None = None,
    under_odds: float | None = None,
    bankroll: float = DEFAULT_BANKROLL,
    data_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Attach market metric fields to a single edge row dict."""
    rec = str(row.get("recommendation", ""))
    if rec in {"Pass", "Pass (No Data)"} or rec.startswith("Pass"):
        row.update(empty_wager_columns(data_warnings=data_warnings))
        return row

    recommended = row.get("recommended_odds")
    if recommended is None and over_odds is not None and under_odds is not None:
        recommended = odds_for_recommendation(rec, over_odds, under_odds)

    prob_pct = row.get("probability_pct")
    true_prob = float(prob_pct) / 100.0 if prob_pct is not None and not pd.isna(prob_pct) else None
    edge_pct = row.get("edge_pct")

    if true_prob is None:
        row.update(empty_wager_columns(data_warnings=data_warnings))
        return row

    warnings = list(data_warnings or [])
    metrics = compute_wager_metrics(
        true_prob,
        recommended,
        edge_pct=float(edge_pct) if edge_pct is not None and not pd.isna(edge_pct) else None,
        data_warnings=warnings,
        bankroll=bankroll,
    )
    return apply_wager_metrics_to_row(row, metrics, bankroll=bankroll)


def enrich_edge_sheet_with_market_metrics(
    df: pd.DataFrame,
    *,
    bankroll: float = DEFAULT_BANKROLL,
) -> pd.DataFrame:
    """Add EV, confidence tier, Kelly, and stake columns to an edge sheet."""
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + MARKET_METRIC_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        enrich_row_market_metrics(payload, bankroll=bankroll)
        rows.append(payload)

    out = pd.DataFrame(rows)
    for col in MARKET_METRIC_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out
