from __future__ import annotations

from dataclasses import dataclass, field

from config import (
    FRACTIONAL_KELLY,
    MAX_BET_PCT,
    MIN_PLAYABLE_EV,
    TIER1_MIN_EDGE_PCT,
    TIER2_MIN_EDGE_PCT,
    TIER3_MIN_EDGE_PCT,
)


@dataclass(frozen=True)
class WagerMetrics:
    model_prob: float
    american_odds: int | None
    decimal_odds: float | None
    implied_prob: float | None
    edge_pct: float | None
    ev_per_unit: float | None
    full_kelly: float | None
    fractional_kelly_pct: float | None
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


def american_to_decimal(american_odds: float | int) -> float:
    odds = int(round(float(american_odds)))
    if odds == 0:
        return 2.0
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
    edge = float(edge_pct)
    if edge >= TIER1_MIN_EDGE_PCT:
        return "Tier-1 High Conviction"
    if edge >= TIER2_MIN_EDGE_PCT:
        return "Tier-2 Standard"
    if edge > TIER3_MIN_EDGE_PCT:
        return "Tier-3 Marginal"
    return "No Bet"


def confidence_score_from_edge_and_prob(edge_pct: float, model_prob: float) -> int:
    return min(100, max(0, round(edge_pct * 4.0 + (model_prob - 0.50) * 80)))


def confidence_label_from_score(score: int) -> str:
    if score >= 85:
        return "Elite"
    if score >= 70:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def confidence_from_edge_and_prob(edge_pct: float, model_prob: float) -> tuple[int, str]:
    score = confidence_score_from_edge_and_prob(edge_pct, model_prob)
    return score, confidence_label_from_score(score)


def compute_wager_metrics(
    model_prob: float,
    american_odds: float | int | None,
    *,
    kelly_fraction: float = FRACTIONAL_KELLY,
    max_bet_pct: float = MAX_BET_PCT,
    data_warnings: list[str] | None = None,
) -> WagerMetrics:
    warnings = list(data_warnings or [])
    prob = min(max(float(model_prob), 0.0), 1.0)

    if american_odds is None:
        return WagerMetrics(
            model_prob=prob,
            american_odds=None,
            decimal_odds=None,
            implied_prob=None,
            edge_pct=None,
            ev_per_unit=None,
            full_kelly=None,
            fractional_kelly_pct=None,
            confidence_score=0,
            confidence_tier="No Bet",
            data_warnings=warnings,
        )

    odds_int = int(round(float(american_odds)))
    decimal = american_to_decimal(odds_int)
    implied = american_to_implied(odds_int)
    edge = (prob - implied) * 100.0
    ev = prob * decimal - (1.0 - prob)
    kelly_full = full_kelly_fraction(prob, odds_int)
    kelly_pct = min(kelly_fraction * kelly_full * 100.0, max_bet_pct)
    if ev < MIN_PLAYABLE_EV:
        kelly_pct = 0.0

    tier = confidence_tier_from_edge(edge)
    score = confidence_score_from_edge_and_prob(edge, prob)

    return WagerMetrics(
        model_prob=prob,
        american_odds=odds_int,
        decimal_odds=round(decimal, 4),
        implied_prob=round(implied, 4),
        edge_pct=round(edge, 2),
        ev_per_unit=round(ev, 4),
        full_kelly=round(kelly_full, 4),
        fractional_kelly_pct=round(kelly_pct, 2),
        confidence_score=score,
        confidence_tier=tier,
        data_warnings=warnings,
    )
