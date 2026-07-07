from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from baseball_props.analysis.edge_sheet_health import EdgeSheetHealthReport, record_edge_skip
from baseball_props.analysis.market_metrics import odds_for_recommendation
from baseball_props.config import (
    DEFAULT_BANKROLL,
    EDGE_HITS_SIGMA,
    EDGE_OUTS_SIGMA,
    HITS_PROP_TARGET_LINES,
)
from baseball_props.market.calculations import (
    apply_wager_metrics_to_row,
    compute_wager_metrics,
    empty_wager_columns,
)

from baseball_props.analysis.edge_sheets import (
    PASS_NO_DATA,
    _build_game_context,
    _format_odds_pair,
    _is_valid_number,
    _no_data_batter_row,
    _no_data_pitcher_row,
    _resolve_proj_hits,
    _safe_float,
    best_side_edge,
    prob_over_continuous,
)


def _is_hits_target_line(market_line: float) -> bool:
    return any(abs(float(market_line) - float(target)) < 1e-6 for target in HITS_PROP_TARGET_LINES)


@dataclass(frozen=True)
class UmpireModifiers:
    umpire_name: str
    zone_size_modifier: float = 1.0
    run_environment_modifier: float = 1.0


# Normalized umpire name → run environment modifier (1.0 = league average)
UMPIRE_RUN_MULTIPLIERS: dict[str, float] = {
    "default": 1.0,
    "cb bucknor": 1.04,
    "angel hernandez": 1.03,
    "marvin hudson": 0.97,
    "dan iassogna": 0.98,
    "joe west": 0.99,
    "vic carapazza": 1.02,
    "lance barrett": 0.98,
    "ryan additon": 1.01,
}

UMPIRE_ZONE_MULTIPLIERS: dict[str, float] = {
    "default": 1.0,
    "cb bucknor": 0.96,
    "angel hernandez": 0.95,
    "marvin hudson": 1.03,
    "dan iassogna": 1.02,
}


def normalize_umpire_name(raw: str) -> str:
    """Normalize crew string for dictionary lookup."""
    text = str(raw or "").strip().lower()
    text = re.sub(r"\s+jr\.?$", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lookup_umpire_modifiers(crew_string: str | None) -> UmpireModifiers:
    """Map home-plate crew string to run/zone multipliers; unknown → 1.0."""
    if not crew_string or str(crew_string).strip().upper() in {"TBD", "UNKNOWN", ""}:
        return UmpireModifiers(umpire_name="", zone_size_modifier=1.0, run_environment_modifier=1.0)

    display_name = str(crew_string).strip()
    key = normalize_umpire_name(display_name)
    if not key:
        return UmpireModifiers(umpire_name="", zone_size_modifier=1.0, run_environment_modifier=1.0)

    run_mod = UMPIRE_RUN_MULTIPLIERS.get(key, UMPIRE_RUN_MULTIPLIERS.get("default", 1.0))
    zone_mod = UMPIRE_ZONE_MULTIPLIERS.get(key, UMPIRE_ZONE_MULTIPLIERS.get("default", 1.0))
    return UmpireModifiers(
        umpire_name=display_name,
        zone_size_modifier=zone_mod,
        run_environment_modifier=run_mod,
    )


def _umpire_warnings_from_row(row: pd.Series) -> list[str]:
    """Append umpire context to edge data warnings when modifier is non-neutral."""
    warnings: list[str] = []
    modifier = row.get("run_environment_modifier")
    name = row.get("umpire_name")
    if name and str(name).strip() and str(name).strip().upper() != "TBD":
        try:
            mod = float(modifier) if modifier is not None else 1.0
        except (TypeError, ValueError):
            mod = 1.0
        if mod != 1.0:
            warnings.append(f"Umpire {name} run modifier {mod:.2f}")
    return warnings


def _merge_umpire_warnings(result: dict[str, Any], row: pd.Series) -> None:
    """Append non-neutral umpire notes to row data_warnings in-place."""
    extra = _umpire_warnings_from_row(row)
    if not extra:
        return
    existing = str(result.get("data_warnings") or "").strip()
    parts = [p for p in ([existing] if existing else []) + extra if p]
    result["data_warnings"] = "; ".join(parts)


def _parse_warnings(warnings: str | list[str] | None) -> list[str]:
    if not warnings:
        return []
    if isinstance(warnings, list):
        return [w for w in warnings if w]
    return [part for part in str(warnings).split("; ") if part]


def _attach_wager_to_row(
    row: dict[str, Any],
    *,
    over_odds: float | None = None,
    under_odds: float | None = None,
    bankroll: float = DEFAULT_BANKROLL,
    data_warnings: list[str] | None = None,
) -> dict[str, Any]:
    rec = str(row.get("recommendation", ""))
    if rec in {PASS_NO_DATA} or rec.startswith("Pass"):
        row.update(empty_wager_columns(data_warnings=data_warnings))
        return row

    recommended = row.get("recommended_odds")
    if recommended is None and over_odds is not None and under_odds is not None:
        recommended = odds_for_recommendation(rec, over_odds, under_odds)
        row["recommended_odds"] = recommended

    prob_pct = row.get("probability_pct")
    if prob_pct is None or not _is_valid_number(prob_pct):
        row.update(empty_wager_columns(data_warnings=data_warnings))
        return row

    true_prob = float(prob_pct) / 100.0
    edge_pct = row.get("edge_pct")
    warnings = list(data_warnings or [])
    warnings.extend(_parse_warnings(row.get("warnings")))

    metrics = compute_wager_metrics(
        true_prob,
        recommended,
        edge_pct=float(edge_pct) if edge_pct is not None and _is_valid_number(edge_pct) else None,
        data_warnings=warnings,
        bankroll=bankroll,
    )
    return apply_wager_metrics_to_row(row, metrics, bankroll=bankroll)


def _finalize_batter_edge(
    base: dict[str, Any],
    *,
    market_line: float | None,
    over_odds: float | None,
    under_odds: float | None,
    model_prob_over: float | None,
    recommendation: str,
    edge_pct: float | None,
    verdict: str = "Play",
    warnings: str = "",
    k_pct: float | None = None,
    contact_pct: float | None = None,
    babip: float | None = None,
    env_bonus: float | None = None,
    bullpen_bonus: float | None = None,
    bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    filter_pass = (
        recommendation != PASS_NO_DATA
        and (recommendation == "Pass" or str(recommendation).startswith("Pass ("))
    )
    warning_list = _parse_warnings(warnings)

    if not _is_valid_number(market_line):
        base.update(
            {
                "market_line": None,
                "over_under_odds": _format_odds_pair(over_odds, under_odds),
                "probability_pct": None,
                "edge_pct": None,
                "recommendation": PASS_NO_DATA,
                "verdict": "Pass",
            }
        )
        base.update(empty_wager_columns(data_warnings=warning_list))
        return base

    if recommendation == PASS_NO_DATA or (
        not filter_pass
        and (
            model_prob_over is None
            or not _is_valid_number(model_prob_over)
            or edge_pct is None
            or not _is_valid_number(edge_pct)
        )
    ):
        base.update(
            {
                "market_line": float(market_line),
                "over_under_odds": _format_odds_pair(over_odds, under_odds),
                "probability_pct": None,
                "edge_pct": None,
                "recommendation": PASS_NO_DATA,
                "verdict": "Pass",
            }
        )
        base.update(empty_wager_columns(data_warnings=warning_list))
        return base

    prob_pct = round(float(model_prob_over) * 100.0, 1) if _is_valid_number(model_prob_over) else None
    edge_value = round(float(edge_pct), 1) if _is_valid_number(edge_pct) else None
    rec_odds = odds_for_recommendation(recommendation, over_odds, under_odds)

    base.update(
        {
            "market_line": float(market_line),
            "over_under_odds": _format_odds_pair(over_odds, under_odds),
            "probability_pct": prob_pct,
            "edge_pct": edge_value,
            "recommendation": recommendation,
            "verdict": verdict,
            "warnings": warnings,
            "k_pct": k_pct,
            "contact_pct": contact_pct,
            "babip": babip,
            "env_bonus": env_bonus,
            "bullpen_bonus": bullpen_bonus,
            "recommended_odds": rec_odds,
        }
    )
    return _attach_wager_to_row(
        base,
        over_odds=over_odds,
        under_odds=under_odds,
        bankroll=bankroll,
        data_warnings=warning_list,
    )


def _evaluate_hits_target_line(
    row: pd.Series,
    *,
    proj_hits: float,
    market_line: float,
    over_odds: float | None,
    under_odds: float | None,
    prop_lines: pd.DataFrame,
    base: dict[str, Any],
    bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    from baseball_props.analysis.guardrails import evaluate_hits_prop

    evaluation = evaluate_hits_prop(
        str(row.get("player_id", "")),
        str(row.get("opp_sp_id", "")),
        _build_game_context(row),
        proj_hits=float(proj_hits),
        market_line=market_line,
        over_odds=over_odds,
        under_odds=under_odds,
        prop_lines=prop_lines,
    )
    env_bonus = (
        evaluation.adjustments.get("park_hit_bonus")
        or evaluation.adjustments.get("temp_bonus")
        or evaluation.adjustments.get("wind_out_bonus")
        or evaluation.adjustments.get("env_multiplier")
    )
    common = {
        "market_line": market_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "warnings": "; ".join(evaluation.warnings),
        "k_pct": evaluation.adjustments.get("k_pct"),
        "contact_pct": evaluation.adjustments.get("contact_pct"),
        "babip": evaluation.adjustments.get("babip"),
        "env_bonus": env_bonus,
        "bullpen_bonus": evaluation.adjustments.get("bullpen_bonus"),
        "bankroll": bankroll,
    }

    if evaluation.recommendation == PASS_NO_DATA:
        return _finalize_batter_edge(
            base,
            model_prob_over=None,
            recommendation=PASS_NO_DATA,
            edge_pct=None,
            verdict="Pass",
            **common,
        )

    recommendation = (
        "Pass"
        if evaluation.verdict == "Pass" or evaluation.recommendation.startswith("Pass")
        else evaluation.recommendation
    )
    prob_side = evaluation.adjusted_prob_over if _is_valid_number(evaluation.adjusted_prob_over) else None
    edge_pct = evaluation.edge_pct if _is_valid_number(evaluation.edge_pct) else None

    if recommendation == "Pass" or edge_pct is None:
        return _finalize_batter_edge(
            base,
            model_prob_over=prob_side,
            recommendation="Pass" if recommendation != PASS_NO_DATA else PASS_NO_DATA,
            edge_pct=edge_pct,
            verdict="Pass",
            **common,
        )

    return _finalize_batter_edge(
        base,
        model_prob_over=prob_side,
        recommendation=recommendation,
        edge_pct=edge_pct,
        verdict=evaluation.verdict,
        **common,
    )


def _evaluate_hits_continuous(
    *,
    proj_hits: float,
    market_line: float,
    over_odds: float | None,
    under_odds: float | None,
    base: dict[str, Any],
    bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
    model_prob_over = prob_over_continuous(float(proj_hits), EDGE_HITS_SIGMA, market_line)
    recommendation, prob_side, edge_pct = best_side_edge(
        model_prob_over if model_prob_over is not None else float("nan"),
        over_odds,
        under_odds,
    )
    return _finalize_batter_edge(
        base,
        market_line=market_line,
        over_odds=over_odds,
        under_odds=under_odds,
        model_prob_over=prob_side,
        recommendation=recommendation,
        edge_pct=edge_pct,
        verdict="Play",
        bankroll=bankroll,
    )


def build_batter_edge_row(
    row: pd.Series,
    quote_row: pd.Series | None,
    prop_lines: pd.DataFrame,
    edge_health: EdgeSheetHealthReport | None = None,
    *,
    bankroll: float = DEFAULT_BANKROLL,
    skip_label: str | None = None,
) -> dict[str, Any]:
    player = str(row.get("player_name", ""))
    proj_hits = _resolve_proj_hits(row)
    if not _is_valid_number(proj_hits):
        record_edge_skip(edge_health, "batter_no_projection", player=player, detail="missing proj_hits")
        out = _no_data_batter_row(row)
        out.update(empty_wager_columns())
        return out

    base = _no_data_batter_row(row, proj_hits=float(proj_hits))
    base["recommendation"] = PASS_NO_DATA
    base["verdict"] = "Pass"

    if quote_row is None:
        record_edge_skip(
            edge_health,
            skip_label or "batter_no_market_quote",
            player=player,
        )
        out = dict(base)
        out.update(empty_wager_columns())
        return out

    market_line = _safe_float(quote_row.get("market_line"))
    if market_line is None:
        record_edge_skip(edge_health, "batter_no_line", player=player)
        out = dict(base)
        out.update(empty_wager_columns())
        return out

    over_odds = quote_row.get("over_odds")
    under_odds = quote_row.get("under_odds")
    if not _is_valid_number(over_odds) and not _is_valid_number(under_odds):
        record_edge_skip(edge_health, "batter_no_odds", player=player)

    if _is_hits_target_line(market_line):
        result = _evaluate_hits_target_line(
            row,
            proj_hits=float(proj_hits),
            market_line=market_line,
            over_odds=over_odds,
            under_odds=under_odds,
            prop_lines=prop_lines,
            base=base,
            bankroll=bankroll,
        )
        rec = str(result.get("recommendation", ""))
        if rec.startswith("Pass") and rec != PASS_NO_DATA:
            record_edge_skip(edge_health, "batter_hits_filter_pass", player=player)
        _merge_umpire_warnings(result, row)
        return result

    result = _evaluate_hits_continuous(
        proj_hits=float(proj_hits),
        market_line=market_line,
        over_odds=over_odds,
        under_odds=under_odds,
        base=base,
        bankroll=bankroll,
    )
    _merge_umpire_warnings(result, row)
    return result


def build_pitcher_edge_row(
    row: pd.Series,
    quote_row: pd.Series | None,
    edge_health: EdgeSheetHealthReport | None = None,
    *,
    bankroll: float = DEFAULT_BANKROLL,
    skip_label: str | None = None,
) -> dict[str, Any]:
    player = str(row.get("pitcher_name", ""))
    proj_outs = _safe_float(row.get("proj_outs"))
    if proj_outs is None:
        record_edge_skip(edge_health, "pitcher_no_projection", player=player)
        out = _no_data_pitcher_row(row)
        out.update(empty_wager_columns())
        return out

    base = _no_data_pitcher_row(row, proj_outs=proj_outs)

    if quote_row is None:
        record_edge_skip(
            edge_health,
            skip_label or "pitcher_no_market_quote",
            player=player,
        )
        out = dict(base)
        out.update(empty_wager_columns())
        return out

    market_line = _safe_float(quote_row.get("market_line"))
    if market_line is None:
        record_edge_skip(edge_health, "pitcher_no_line", player=player)
        out = dict(base)
        out.update(empty_wager_columns())
        return out

    over_odds = quote_row.get("over_odds")
    under_odds = quote_row.get("under_odds")
    if not _is_valid_number(over_odds) and not _is_valid_number(under_odds):
        record_edge_skip(edge_health, "pitcher_no_odds", player=player)

    model_prob_over = prob_over_continuous(proj_outs, EDGE_OUTS_SIGMA, market_line)
    recommendation, prob_side, edge_pct = best_side_edge(
        model_prob_over if model_prob_over is not None else float("nan"),
        over_odds,
        under_odds,
    )

    if model_prob_over is None or edge_pct is None or not _is_valid_number(edge_pct):
        base.update(
            {
                "market_line": market_line,
                "probability_pct": None,
                "edge_pct": None,
                "recommendation": PASS_NO_DATA,
            }
        )
        base.update(empty_wager_columns())
        return base

    prob_pct = round(float(prob_side) * 100.0, 1) if _is_valid_number(prob_side) else None
    rec_odds = odds_for_recommendation(recommendation, over_odds, under_odds)
    base.update(
        {
            "market_line": market_line,
            "probability_pct": prob_pct,
            "edge_pct": round(float(edge_pct), 1),
            "recommendation": recommendation,
            "recommended_odds": rec_odds,
        }
    )
    return _attach_wager_to_row(base, over_odds=over_odds, under_odds=under_odds, bankroll=bankroll)
