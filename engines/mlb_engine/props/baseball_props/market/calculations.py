from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baseball_props.config import (
    CONFIDENCE_TIERS,
    DEFAULT_BANKROLL,
    FRACTIONAL_KELLY,
    KELLY_MAX_STAKE_PCT,
    MAX_BET_PCT,
    MIN_PLAYABLE_EV,
    TIER_KELLY_MULTIPLIERS,
)

MARKET_METRIC_COLUMNS = [
    "recommended_odds",
    "decimal_odds",
    "true_probability",
    "ev_per_unit",
    "confidence_tier",
    "confidence_score",
    "kelly_fraction",
    "fractional_kelly_pct",
    "suggested_stake",
    "data_warnings",
]


@dataclass(frozen=True)
class WagerMetrics:
    model_prob: float
    american_odds: int | None
    decimal_odds: float | None
    implied_prob: float | None
    edge_pct: float | None
    ev_per_unit: float | None
    full_kelly: float | None
    kelly_fraction: float | None
    fractional_kelly_pct: float | None
    suggested_stake: float | None
    confidence_score: int
    confidence_tier: str
    data_warnings: list[str] = field(default_factory=list)


def american_to_implied(american_odds: float | int) -> float:
    odds = int(round(float(american_odds)))
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(american_odds: float | int) -> float | None:
    odds = int(round(float(american_odds)))
    if odds == 0:
        return None
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def full_kelly_fraction(model_prob: float, american_odds: float | int) -> float:
    odds = int(round(float(american_odds)))
    if odds == 0:
        return 0.0
    b = odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    q = 1.0 - model_prob
    numerator = b * model_prob - q
    if b <= 0:
        return 0.0
    return max(0.0, numerator / b)


def confidence_tier_from_edge(edge_pct: float | None) -> str:
    if edge_pct is None:
        return "No Bet"
    edge = abs(float(edge_pct))
    for threshold, label in CONFIDENCE_TIERS:
        if edge >= threshold:
            return label
    return "Below Threshold"


def confidence_score_from_edge_and_prob(edge_pct: float, model_prob: float) -> int:
    return min(100, max(0, round(edge_pct * 4.0 + (model_prob - 0.50) * 80)))


def tier_kelly_fraction(full_kelly: float | None, tier_label: str) -> float | None:
    if full_kelly is None or full_kelly <= 0:
        return None
    multiplier = TIER_KELLY_MULTIPLIERS.get(tier_label, 0.0)
    if multiplier <= 0:
        return None
    scaled = full_kelly * multiplier
    capped = min(scaled, KELLY_MAX_STAKE_PCT)
    return capped if capped > 0 else None


def suggested_stake_from_kelly(
    kelly_fraction: float | None,
    bankroll: float = DEFAULT_BANKROLL,
) -> float | None:
    if kelly_fraction is None or kelly_fraction <= 0 or bankroll <= 0:
        return None
    return round(bankroll * kelly_fraction, 2)


def compute_wager_metrics(
    model_prob: float,
    american_odds: float | int | None,
    *,
    edge_pct: float | None = None,
    kelly_fraction: float = FRACTIONAL_KELLY,
    max_bet_pct: float = MAX_BET_PCT,
    data_warnings: list[str] | None = None,
    bankroll: float = DEFAULT_BANKROLL,
) -> WagerMetrics:
    warnings = list(data_warnings or [])
    prob = min(max(float(model_prob), 0.0), 1.0)

    if american_odds is None:
        return WagerMetrics(
            model_prob=prob,
            american_odds=None,
            decimal_odds=None,
            implied_prob=None,
            edge_pct=round(float(edge_pct), 2) if edge_pct is not None else None,
            ev_per_unit=None,
            full_kelly=None,
            kelly_fraction=None,
            fractional_kelly_pct=None,
            suggested_stake=None,
            confidence_score=0,
            confidence_tier="No Bet",
            data_warnings=warnings,
        )

    odds_int = int(round(float(american_odds)))
    decimal = american_to_decimal(odds_int)
    implied = american_to_implied(odds_int)
    edge = float(edge_pct) if edge_pct is not None else (prob - implied) * 100.0
    ev = prob * decimal - (1.0 - prob) if decimal is not None else None
    kelly_full = full_kelly_fraction(prob, odds_int)
    tier = confidence_tier_from_edge(edge)
    score = confidence_score_from_edge_and_prob(edge, prob)

    kelly_frac = tier_kelly_fraction(kelly_full, tier)
    if ev is not None and ev < MIN_PLAYABLE_EV:
        kelly_frac = None

    return WagerMetrics(
        model_prob=prob,
        american_odds=odds_int,
        decimal_odds=round(decimal, 4) if decimal is not None else None,
        implied_prob=round(implied, 4),
        edge_pct=round(edge, 2),
        ev_per_unit=round(ev, 4) if ev is not None else None,
        full_kelly=round(kelly_full, 4),
        kelly_fraction=round(kelly_frac, 4) if kelly_frac is not None else None,
        fractional_kelly_pct=round(kelly_frac * 100.0, 2) if kelly_frac is not None else None,
        suggested_stake=suggested_stake_from_kelly(kelly_frac, bankroll=bankroll),
        confidence_score=score,
        confidence_tier=tier,
        data_warnings=warnings,
    )


def empty_wager_columns(*, data_warnings: list[str] | None = None) -> dict[str, Any]:
    warning_text = "; ".join(data_warnings or [])
    return {
        col: None if col != "data_warnings" else warning_text
        for col in MARKET_METRIC_COLUMNS
    }


def wager_metrics_to_edge_columns(
    metrics: WagerMetrics,
    *,
    bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    tier = metrics.confidence_tier
    if tier in {"No Bet", "Below Threshold"}:
        tier_out = None if tier == "No Bet" else tier
    else:
        tier_out = tier

    kelly_frac = metrics.kelly_fraction
    if kelly_frac is None and metrics.full_kelly is not None:
        kelly_frac = tier_kelly_fraction(metrics.full_kelly, metrics.confidence_tier)

    return {
        "recommended_odds": metrics.american_odds,
        "decimal_odds": metrics.decimal_odds,
        "true_probability": round(metrics.model_prob, 4),
        "ev_per_unit": metrics.ev_per_unit,
        "confidence_tier": tier_out,
        "confidence_score": metrics.confidence_score if tier_out else None,
        "kelly_fraction": kelly_frac,
        "fractional_kelly_pct": round(kelly_frac * 100.0, 2) if kelly_frac is not None else None,
        "suggested_stake": metrics.suggested_stake
        or suggested_stake_from_kelly(kelly_frac, bankroll=bankroll),
        "data_warnings": "; ".join(metrics.data_warnings),
    }


def apply_wager_metrics_to_row(
    row: dict[str, Any],
    metrics: WagerMetrics,
    *,
    bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    row.update(wager_metrics_to_edge_columns(metrics, bankroll=bankroll))
    return row
