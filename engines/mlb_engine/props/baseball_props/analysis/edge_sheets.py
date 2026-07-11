from __future__ import annotations

import math
from datetime import date as date_type
from typing import Any

import pandas as pd

from baseball_props.analysis.edge_sheet_health import EdgeSheetHealthReport
from baseball_props.analysis.prop_matching import filter_plausible_market_lines, match_name
from baseball_props.config import (
    LEAGUE_PITCHES_PER_OUT,
    MAX_PROP_PROB,
    MIN_PROP_PROB,
    TB_PER_WOBA_PA,
)
from baseball_props.analysis.market_metrics import MARKET_METRIC_COLUMNS
from baseball_props.data.odds_props import consolidated_prop_quotes

PASS_NO_DATA = "Pass (No Data)"
SKIP_DISPLAY_RECOMMENDATIONS = frozenset({PASS_NO_DATA, "No line"})

BATTER_SHEET_COLUMNS = [
    "player_id",
    "player_name",
    "team_id",
    "proj_hits",
    "market_line",
    "over_under_odds",
    "probability_pct",
    "edge_pct",
    "recommendation",
    "verdict",
    "warnings",
    "lineup_slot",
    "k_pct",
    "contact_pct",
    "babip",
    "env_bonus",
    "bullpen_bonus",
    "game_id",
    "market",
] + MARKET_METRIC_COLUMNS

PITCHER_SHEET_COLUMNS = [
    "pitcher_name",
    "team_id",
    "proj_outs",
    "proj_pitch_count",
    "pitches_per_out_baseline",
    "market_line",
    "probability_pct",
    "edge_pct",
    "recommendation",
    "game_id",
    "market",
] + MARKET_METRIC_COLUMNS

CONVICTION_COLUMNS = [
    "player_name",
    "market",
    "model_value",
    "market_line",
    "probability_pct",
    "edge_pct",
    "recommendation",
] + MARKET_METRIC_COLUMNS


def _is_valid_number(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if not _is_valid_number(value):
        return default
    return float(value)


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (no vig removal)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / (-odds + 100.0)


def scaled_prop_sigma(base_sigma: float, base_mu: float, adjusted_mu: float) -> float:
    """Scale projection sigma with the mean so modifiers do not collapse variance."""
    if not _is_valid_number(base_sigma) or float(base_sigma) <= 0:
        return float(base_sigma) if _is_valid_number(base_sigma) else 0.0
    if not _is_valid_number(base_mu) or float(base_mu) <= 0:
        return float(base_sigma)
    if not _is_valid_number(adjusted_mu):
        return float(base_sigma)
    ratio = float(adjusted_mu) / float(base_mu)
    return max(1e-6, float(base_sigma) * ratio)


def clamp_prop_probability(prob: float | None) -> float | None:
    """Hard-cap model prop probabilities to [MIN_PROP_PROB, MAX_PROP_PROB]."""
    if not _is_valid_number(prob):
        return None
    return min(MAX_PROP_PROB, max(MIN_PROP_PROB, float(prob)))


def prob_over_continuous(mu: float, sigma: float, line: float) -> float | None:
    """P(stat > line) under Normal(mu, sigma). Returns None when inputs are invalid."""
    if not _is_valid_number(mu) or not _is_valid_number(line) or not _is_valid_number(sigma):
        return None
    mu_f = float(mu)
    line_f = float(line)
    sigma_f = float(sigma)
    if sigma_f <= 0:
        raw = 1.0 if mu_f > line_f else 0.0 if mu_f < line_f else 0.5
        return clamp_prop_probability(raw)
    z = (line_f - mu_f) / (sigma_f * math.sqrt(2.0))
    return clamp_prop_probability(0.5 * (1.0 - math.erf(z)))


def _format_odds_pair(over_odds: float | None, under_odds: float | None) -> str:
    if not _is_valid_number(over_odds) or not _is_valid_number(under_odds):
        return ""
    over_int = int(round(float(over_odds)))
    under_int = int(round(float(under_odds)))
    over_str = f"+{over_int}" if over_int > 0 else str(over_int)
    under_str = f"+{under_int}" if under_int > 0 else str(under_int)
    return f"{over_str} / {under_str}"


def best_side_edge(
    model_prob_over: float,
    over_odds: float | None,
    under_odds: float | None,
) -> tuple[str, float | None, float | None]:
    """
    Pick Over or Under with the highest positive edge; if both negative, least-bad side.

    Returns (recommendation, model_prob_for_side, edge_pct). edge_pct is None when odds
    are missing or model_prob_over is invalid.
    """
    if not _is_valid_number(model_prob_over):
        return "Over", None, None

    prob_over = float(model_prob_over)
    if not _is_valid_number(over_odds) or not _is_valid_number(under_odds):
        lean = "Over" if prob_over >= 0.5 else "Under"
        prob = prob_over if lean == "Over" else (1.0 - prob_over)
        return lean, prob, None

    implied_over = american_to_implied(float(over_odds))
    implied_under = american_to_implied(float(under_odds))
    model_prob_under = 1.0 - prob_over
    edge_over = (prob_over - implied_over) * 100.0
    edge_under = (model_prob_under - implied_under) * 100.0

    if edge_over >= edge_under:
        return "Over", prob_over, edge_over
    return "Under", model_prob_under, edge_under


def _resolve_proj_hits(row: pd.Series) -> float | None:
    if "proj_hits" in row.index and pd.notna(row.get("proj_hits")):
        value = _safe_float(row.get("proj_hits"))
        if value is not None:
            return value
    if "proj_total_bases" in row.index and pd.notna(row.get("proj_total_bases")):
        tb = _safe_float(row.get("proj_total_bases"))
        if tb is not None:
            return round(tb * 0.45, 3)
    if (
        "proj_woba" in row.index
        and "proj_pa" in row.index
        and _is_valid_number(row.get("proj_woba"))
        and _is_valid_number(row.get("proj_pa"))
    ):
        return round(float(row["proj_woba"]) * float(row["proj_pa"]) * 0.235, 3)
    return None


def _resolve_proj_tb(row: pd.Series) -> float | None:
    """Deprecated — use _resolve_proj_hits."""
    return _resolve_proj_hits(row)


def _no_data_batter_row(row: pd.Series, proj_hits: float | None = None) -> dict[str, Any]:
    resolved = proj_hits if proj_hits is not None else _resolve_proj_hits(row)
    return {
        "player_id": row.get("player_id", ""),
        "player_name": row.get("player_name", ""),
        "team_id": row.get("team_id", ""),
        "proj_hits": round(resolved, 3) if _is_valid_number(resolved) else None,
        "market_line": None,
        "over_under_odds": "",
        "probability_pct": None,
        "edge_pct": None,
        "recommendation": PASS_NO_DATA,
        "verdict": "Pass",
        "warnings": "",
        "lineup_slot": row.get("lineup_slot"),
        "k_pct": None,
        "contact_pct": None,
        "babip": None,
        "env_bonus": None,
        "bullpen_bonus": None,
        "game_id": row.get("game_id", ""),
        "market": "batter_hits",
    }


def _no_data_pitcher_row(row: pd.Series, proj_outs: float | None = None) -> dict[str, Any]:
    resolved = proj_outs if proj_outs is not None else _safe_float(row.get("proj_outs"))
    return {
        "pitcher_name": row.get("pitcher_name", ""),
        "team_id": row.get("team_id", ""),
        "proj_outs": round(resolved, 2) if _is_valid_number(resolved) else None,
        "proj_pitch_count": round(float(row.get("proj_pitch_count", 0)), 1)
        if _is_valid_number(row.get("proj_pitch_count"))
        else None,
        "pitches_per_out_baseline": LEAGUE_PITCHES_PER_OUT,
        "market_line": None,
        "edge_pct": None,
        "recommendation": PASS_NO_DATA,
        "game_id": row.get("game_id", ""),
        "market": "pitcher_outs",
    }


def _parse_game_date(row: pd.Series) -> date_type:
    raw = row.get("game_date")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return date_type.today()
    if isinstance(raw, date_type):
        return raw
    try:
        return date_type.fromisoformat(str(raw)[:10])
    except ValueError:
        return date_type.today()


def _build_game_context(row: pd.Series):
    from baseball_props.analysis.guardrails import GameContext

    venue_raw = row.get("park_id") or row.get("venue_id")
    venue_id: int | None = None
    if venue_raw is not None and str(venue_raw).strip().isdigit():
        venue_id = int(str(venue_raw))

    mlb_pk = row.get("mlb_game_pk")
    mlb_game_pk = int(mlb_pk) if mlb_pk is not None and str(mlb_pk).strip().isdigit() else None

    umpire_mod = 1.0
    if pd.notna(row.get("run_environment_modifier")):
        try:
            umpire_mod = float(row["run_environment_modifier"])
        except (TypeError, ValueError):
            umpire_mod = 1.0

    return GameContext(
        game_id=str(row.get("game_id", "")),
        player_id=str(row.get("player_id", "")),
        player_name=str(row.get("player_name", "")),
        opponent_pitcher_id=str(row.get("opp_sp_id", "")),
        opponent_team_id=str(row.get("opp_team_id", "")),
        batting_team_id=str(row.get("team_id", "")),
        lineup_slot=int(row["lineup_slot"]) if pd.notna(row.get("lineup_slot")) else None,
        venue_id=venue_id,
        park_tb_factor=float(row.get("park_tb_factor", row.get("park_factor_runs", 1.0)) or 1.0),
        temp_f=float(row["temp_f"]) if pd.notna(row.get("temp_f")) else None,
        wind_mph=float(row["wind_mph"]) if pd.notna(row.get("wind_mph")) else None,
        wind_dir=str(row.get("wind_dir")) if pd.notna(row.get("wind_dir")) else None,
        game_date=_parse_game_date(row),
        opp_bullpen_status=str(row.get("opp_bullpen_fatigue_status"))
        if pd.notna(row.get("opp_bullpen_fatigue_status"))
        else None,
        home_team_id=str(row.get("home_team_id", "")) if pd.notna(row.get("home_team_id")) else None,
        away_team_id=str(row.get("away_team_id", "")) if pd.notna(row.get("away_team_id")) else None,
        mlb_game_pk=mlb_game_pk,
        umpire_run_modifier=umpire_mod,
    )


def _consolidated_hits_quotes(prop_lines: pd.DataFrame) -> pd.DataFrame:
    """One quote per player/game/hits line (Over 0.5 and Over 1.5 tracked separately)."""
    from baseball_props.config import HITS_PROP_TARGET_LINES

    empty_cols = [
        "game_id",
        "player_name",
        "market",
        "market_line",
        "over_odds",
        "under_odds",
    ]
    if prop_lines.empty:
        return pd.DataFrame(columns=empty_cols)

    hits = prop_lines[prop_lines["market"] == "batter_hits"].copy()
    if hits.empty or "line" not in hits.columns:
        return pd.DataFrame(columns=empty_cols)

    hits = hits[hits["line"].isin(HITS_PROP_TARGET_LINES)]
    if hits.empty:
        return pd.DataFrame(columns=empty_cols)

    rows: list[dict[str, Any]] = []
    for (game_id, player_name, line), grp in hits.groupby(["game_id", "player_name", "line"]):
        over = grp.loc[grp["side"] == "Over", "odds"].median()
        under = grp.loc[grp["side"] == "Under", "odds"].median()
        rows.append(
            {
                "game_id": game_id,
                "player_name": player_name,
                "market": "batter_hits",
                "market_line": float(line),
                "over_odds": over if pd.notna(over) else None,
                "under_odds": under if pd.notna(under) else None,
            }
        )
    quotes = pd.DataFrame(rows, columns=empty_cols)
    return filter_plausible_market_lines(quotes)


def _prepare_batter_quotes(prop_lines: pd.DataFrame) -> pd.DataFrame:
    return _consolidated_hits_quotes(prop_lines)


def _prepare_pitcher_quotes(prop_lines: pd.DataFrame) -> pd.DataFrame:
    quotes = pd.DataFrame(
        columns=["game_id", "player_name", "market", "market_line", "over_odds", "under_odds"]
    )
    if prop_lines.empty:
        return quotes
    quotes = filter_plausible_market_lines(consolidated_prop_quotes(prop_lines))
    return quotes[quotes["market"] == "pitcher_outs"]


def _match_batter_quote(row: pd.Series, quotes: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    from baseball_props.config import HITS_PROP_PRIMARY_LINE

    if quotes.empty:
        return None, "batter_no_market_quote"
    game_id = str(row["game_id"])
    props = quotes[
        (quotes["game_id"].astype(str) == game_id) & (quotes["market"] == "batter_hits")
    ]
    if props.empty:
        return None, "batter_no_market_quote"
    matched = match_name(str(row["player_name"]), props["player_name"])
    if matched is None:
        return None, "batter_name_unmatched"
    player_quotes = props.loc[props["player_name"] == matched]
    primary = player_quotes[player_quotes["market_line"] == HITS_PROP_PRIMARY_LINE]
    if not primary.empty:
        return primary.iloc[0], None
    return player_quotes.iloc[0], None


def _match_pitcher_quote(row: pd.Series, quotes: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    if quotes.empty:
        return None, "pitcher_no_market_quote"
    game_id = str(row["game_id"])
    props = quotes[
        (quotes["game_id"].astype(str) == game_id) & (quotes["market"] == "pitcher_outs")
    ]
    if props.empty:
        return None, "pitcher_no_market_quote"
    matched = match_name(str(row["pitcher_name"]), props["player_name"])
    if matched is None:
        return None, "pitcher_name_unmatched"
    return props.loc[props["player_name"] == matched].iloc[0], None


def build_batter_hits_edge_sheet(
    projected: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    edge_health: EdgeSheetHealthReport | None = None,
) -> pd.DataFrame:
    """Build batter hits projection + market edge sheet."""
    from baseball_props.analysis.edge_row_builder import build_batter_edge_row

    if projected.empty:
        return pd.DataFrame(columns=BATTER_SHEET_COLUMNS)

    quotes = _prepare_batter_quotes(prop_lines)
    sort_cols = ["game_id"]
    if "lineup_slot" in projected.columns:
        sort_cols.append("lineup_slot")

    rows: list[dict[str, Any]] = []
    for _, row in projected.sort_values(sort_cols).iterrows():
        quote, skip_label = _match_batter_quote(row, quotes)
        rows.append(
            build_batter_edge_row(
                row,
                quote,
                prop_lines,
                edge_health,
                skip_label=skip_label,
            )
        )
    return pd.DataFrame(rows, columns=BATTER_SHEET_COLUMNS)


def build_batter_tb_edge_sheet(
    projected: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    edge_health: EdgeSheetHealthReport | None = None,
) -> pd.DataFrame:
    """Deprecated alias — hits edge sheet."""
    return build_batter_hits_edge_sheet(projected, prop_lines, edge_health=edge_health)


def build_pitcher_outs_edge_sheet(
    pitcher_outs: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    edge_health: EdgeSheetHealthReport | None = None,
) -> pd.DataFrame:
    """Build pitcher outs + workload projection sheet with market edges."""
    from baseball_props.analysis.edge_row_builder import build_pitcher_edge_row

    if pitcher_outs.empty:
        return pd.DataFrame(columns=PITCHER_SHEET_COLUMNS)

    quotes = _prepare_pitcher_quotes(prop_lines)
    rows: list[dict[str, Any]] = []
    for _, row in pitcher_outs.sort_values(["game_id", "team_id"]).iterrows():
        quote, skip_label = _match_pitcher_quote(row, quotes)
        rows.append(
            build_pitcher_edge_row(row, quote, edge_health, skip_label=skip_label)
        )
    return pd.DataFrame(rows, columns=PITCHER_SHEET_COLUMNS)


def _is_playable_conviction_row(row: pd.Series) -> bool:
    rec = str(row.get("recommendation", ""))
    if rec in SKIP_DISPLAY_RECOMMENDATIONS or rec.startswith("Pass"):
        return False
    if not _is_valid_number(row.get("market_line")) or not _is_valid_number(row.get("edge_pct")):
        return False
    if "verdict" in row.index and str(row.get("verdict", "")) == "Pass":
        return False
    return True


def aggregate_top_conviction(
    batter_sheet: pd.DataFrame,
    pitcher_sheet: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Merge both edge sheets and rank by absolute Edge %."""
    rows: list[dict[str, Any]] = []

    if not batter_sheet.empty:
        for _, row in batter_sheet.iterrows():
            if not _is_playable_conviction_row(row):
                continue
            rows.append(
                {
                    "player_name": row["player_name"],
                    "market": row["market"],
                    "model_value": row["proj_hits"],
                    "market_line": row["market_line"],
                    "probability_pct": row["probability_pct"],
                    "edge_pct": row["edge_pct"],
                    "recommendation": row["recommendation"],
                    **{col: row.get(col) for col in MARKET_METRIC_COLUMNS},
                }
            )

    if not pitcher_sheet.empty:
        for _, row in pitcher_sheet.iterrows():
            if not _is_playable_conviction_row(row):
                continue
            rows.append(
                {
                    "player_name": row["pitcher_name"],
                    "market": row["market"],
                    "model_value": row["proj_outs"],
                    "market_line": row["market_line"],
                    "probability_pct": row.get("probability_pct"),
                    "edge_pct": row["edge_pct"],
                    "recommendation": row["recommendation"],
                    **{col: row.get(col) for col in MARKET_METRIC_COLUMNS},
                }
            )

    if not rows:
        return pd.DataFrame(columns=CONVICTION_COLUMNS)

    result = pd.DataFrame(rows)
    result["abs_edge_pct"] = result["edge_pct"].abs()
    result = (
        result.sort_values("abs_edge_pct", ascending=False)
        .head(top_n)
        .drop(columns=["abs_edge_pct"])
        .reset_index(drop=True)
    )
    return result[CONVICTION_COLUMNS]
