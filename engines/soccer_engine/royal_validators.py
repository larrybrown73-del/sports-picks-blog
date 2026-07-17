"""
Royal Picks situational validators + adversarial "Why Am I Wrong?" analysis.

Kept alongside daily_model (not inside the odds/EV math path) so form,
hit-rate, and position checks stay blind to sportsbook prices -- they only
inspect model projections and historical player rates.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from dataclasses import replace

from ev_engine_core import EVResult, MarketLeg
from player_props_model import (
    PlayerRateProfile,
    STAT_FIELD_BY_MARKET_TYPE,
    normalize_position,
    player_prop_probability,
    position_variance_scale,
)

# Form conflict: projected clear-prob more than 35% above the player's own
# season baseline (closest stable proxy without a per-match last-5 feed).
FORM_DEVIATION_THRESHOLD = 0.35
FORM_CONFLICT_SCORE_PENALTY = 15

# Historical hit-rate gate: under 20% clear-rate proxy => max "Value Angle".
HIT_RATE_FLOOR = 0.20
HIT_RATE_TIER_CAP_SCORE = 89

FOCAL_MIDFIELD_POSITIONS = frozenset({"CAM", "AM", "CM", "M", "WG", "W"})


def estimated_hit_rate(profile: PlayerRateProfile, stat_field: str, line: float | None) -> float | None:
    """
    Season-rate proxy for "how often has this player cleared this line".

    Without a reliable last-20-appearances feed from TheStatsAPI for
    international/knockout splits, the Poisson season lambda is the honest
    proxy -- never invent match-by-match clears we did not observe.
    """

    try:
        return player_prop_probability(profile, stat_field, line, opponent_adjustment=1.0)
    except KeyError:
        return None


def form_deviation_ratio(projected_lambda: float, baseline_lambda: float) -> float | None:
    if baseline_lambda <= 0:
        return None
    return abs(projected_lambda - baseline_lambda) / baseline_lambda


def build_downside_analysis(
    result: EVResult,
    *,
    profile: PlayerRateProfile | None = None,
    hit_rate: float | None = None,
    form_conflict: bool = False,
    market_divergence: bool = False,
    contradiction: bool = False,
) -> dict[str, str]:
    """
    Algorithmic counter-argument for every play that clears the alert floor.

    Every field is derived from observable gates/warnings -- never free-text
    hallucination about tactics we cannot verify from data.
    """

    leg = result.leg
    warnings = list(result.warnings)
    sample = result.sample_size if result.sample_size is not None else (profile.appearances if profile else None)

    if sample is not None and sample < 10:
        dependency = "Relies heavily on low-sample international/season data."
    elif profile is not None and profile.appearances < 15:
        dependency = "Season sample is thin -- rates can swing hard on a few outlier matches."
    elif any("no confirmed lineup" in w.lower() for w in warnings):
        dependency = "Minutes assumption is historical only -- lineup not confirmed yet."
    else:
        dependency = "Primary risk is model misspecification vs. live match context."

    if contradiction:
        tactical = "Tactical contradiction with team ML/Over thesis -- focal creator Under SOT conflicts with game-script."
    elif form_conflict:
        tactical = "Projection sits >35% off the player's own season baseline -- form/role change risk."
    elif profile is not None and normalize_position(getattr(profile, "position", None)) in {"DM", "CB", "D"}:
        tactical = "Role is defensive; counting-stat volume collapses if the side sits in a low block."
    elif leg.market_type in {"total_goals", "btts"}:
        tactical = "If either side parks a low block, chance creation (and total goals) compresses."
    else:
        tactical = "If the opponent denies space in midfield/final third, volume props compress first."

    if market_divergence:
        price_check = "Market is steaming the other way vs. this side -- high EV may be stale model vs. informed money."
    elif result.true_probability < 0.55 and result.ev_per_unit > 0.5:
        price_check = "High EV is largely a product of a longshot bookie price, not a high raw win probability."
    elif result.edge > 0.15:
        price_check = "Large model-vs-market edge -- treat as potential misprice OR model error until vetted."
    else:
        price_check = "Edge is modest; a small line move or vig shift can erase the EV."

    if hit_rate is not None and hit_rate < HIT_RATE_FLOOR:
        dependency = f"Historical clear-rate proxy is only {hit_rate * 100:.0f}% (<20% gate) -- {dependency}"

    return {
        "main_dependency_risk": dependency,
        "tactical_failure_point": tactical,
        "price_vs_likelihood": price_check,
    }


def enrich_results_with_situational_validators(
    results: Sequence[EVResult],
    profiles: Mapping[str, PlayerRateProfile] | None = None,
) -> list[EVResult]:
    """
    Apply form conflict, hit-rate gate, and position-aware flags to a graded
    board. Safe no-op for team markets with no player profile.
    """

    profiles = profiles or {}
    enriched: list[EVResult] = []

    for result in results:
        leg = result.leg
        profile = profiles.get(leg.entity_id) if leg.entity_id else None
        score = result.confidence_score
        warnings = list(result.warnings)
        form_validated = True
        hit_rate_ok = True
        hit_rate: float | None = None
        form_conflict = False
        lineup_confirmed = not any("no confirmed lineup" in w.lower() for w in warnings)
        if not leg.entity_id:
            # Team-level markets: form/hit-rate N/A -- treat as passed so they
            # remain eligible for Royal approval on the other gates.
            form_validated = True
            hit_rate_ok = True
            lineup_confirmed = True

        if profile is not None:
            market_key = _player_stat_market_key(leg)
            stat_field = STAT_FIELD_BY_MARKET_TYPE.get(market_key or "")
            if stat_field:
                # Season clear-prob is the honest last-N proxy without inventing
                # match logs. Projection >35% hotter than season => form flag.
                baseline_clear = player_prop_probability(profile, stat_field, leg.line)
                if baseline_clear > 0 and result.true_probability > baseline_clear * (1.0 + FORM_DEVIATION_THRESHOLD):
                    form_conflict = True
                    form_validated = False
                    score = max(1, score - FORM_CONFLICT_SCORE_PENALTY)
                    warnings.append(
                        f"form conflict: projection >{int(FORM_DEVIATION_THRESHOLD * 100)}% above season baseline"
                    )

                hit_rate = estimated_hit_rate(profile, stat_field, leg.line)
                if hit_rate is not None and hit_rate < HIT_RATE_FLOOR:
                    hit_rate_ok = False
                    score = min(score, HIT_RATE_TIER_CAP_SCORE)
                    warnings.append(f"historical hit-rate proxy {hit_rate * 100:.0f}% under 20% gate")

        # Role scale is already baked into player_prop_probability via
        # position_variance_scale inside _lambda_for_stat; keep a no-op
        # reference so the import stays intentional for callers/tests.
        _ = position_variance_scale(getattr(profile, "position", None) if profile else None)

        downside = build_downside_analysis(
            result,
            profile=profile,
            hit_rate=hit_rate,
            form_conflict=form_conflict,
        )

        enriched.append(
            replace(
                result,
                confidence_score=score,
                warnings=warnings,
                form_validated=form_validated,
                hit_rate_ok=hit_rate_ok,
                lineup_confirmed=lineup_confirmed,
                downside_analysis=downside,
            )
        )

    return enriched


def _player_stat_market_key(leg: MarketLeg) -> str | None:
    market = (leg.market_type or "").lower().replace(" ", "_")
    if market in STAT_FIELD_BY_MARKET_TYPE:
        return market
    # TheStatsAPI player odds often put display names in market_type
    # ("Player Shots") while side carries the machine key.
    side = (leg.side or "").lower()
    if side in STAT_FIELD_BY_MARKET_TYPE:
        return side
    aliases = {
        "player_shots_on_target": "player_shots_on_target",
        "shots_on_target": "player_shots_on_target",
        "player_shots": "player_shots",
        "shots": "player_shots",
        "player_assists": "player_assists",
        "assists": "player_assists",
        "anytime_goalscorer": "anytime_goalscorer",
        "anytime_goal_scorer": "anytime_goalscorer",
    }
    for key, mapped in aliases.items():
        if key in market or key in side:
            return mapped
    return None
