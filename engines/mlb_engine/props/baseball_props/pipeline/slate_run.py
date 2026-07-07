from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from baseball_props.analysis.batter_projection import (
    project_batter_hits,
    project_batter_total_bases,
)
from baseball_props.analysis.edge_sheet_health import EdgeSheetHealthReport
from baseball_props.analysis.edge_sheets import (
    aggregate_top_conviction,
    build_batter_tb_edge_sheet,
    build_pitcher_outs_edge_sheet,
)
from baseball_props.analysis.parlay_builder import ParlayTicket, build_diversified_tickets
from baseball_props.analysis.pitcher_projection import project_pitcher_outs_and_pitches
from baseball_props.core.projections import project_player_rates
from baseball_props.config import DEFAULT_SLATE_SOURCE
from baseball_props.data.ingest import (
    SlateSource,
    build_slate_context,
    load_slate_frames,
    resolve_odds_event_ids_for_slate,
)
from baseball_props.data.injuries import fetch_active_injuries
from baseball_props.data.mlb_live import filter_injured_from_lineups
from baseball_props.data.odds_props import fetch_all_player_props
from baseball_props.data.schemas import SlateContext, resolve_display_name
from baseball_props.logging_utils import get_logger
from baseball_props.opportunity.batters import project_batter_pa
from baseball_props.types import SlateFrames

logger = get_logger(__name__)

_PROP_COLUMNS = [
    "game_id",
    "player_name",
    "market",
    "side",
    "line",
    "odds",
    "bookmaker",
]


@dataclass
class SlateRunResult:
    frames: SlateFrames
    context: SlateContext
    projected: pd.DataFrame
    pitcher_outs: pd.DataFrame
    prop_lines: pd.DataFrame | None = None
    batter_edge_sheet: pd.DataFrame | None = None
    pitcher_edge_sheet: pd.DataFrame | None = None
    conviction: pd.DataFrame | None = None
    conviction_message: str | None = None
    parlay_tickets: list[ParlayTicket] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _resolve_pitcher_names(
    pitcher_outs: pd.DataFrame,
    pitcher_tendencies: pd.DataFrame,
) -> pd.DataFrame:
    out = pitcher_outs.copy()
    if "pitcher_name" not in out.columns:
        out["pitcher_name"] = resolve_display_name(out, "pitcher_id", "pitcher_name")

    tendency_names = (
        pitcher_tendencies.set_index("pitcher_id")["pitcher_name"].astype(str).to_dict()
    )

    def _display(row: pd.Series) -> str:
        name = str(row["pitcher_name"])
        if re.fullmatch(r"SP \d+", name):
            return tendency_names.get(str(row["pitcher_id"]), name)
        return name

    out["pitcher_name"] = out.apply(_display, axis=1)
    return out


def _build_edge_outputs(
    *,
    source: SlateSource,
    frames: SlateFrames,
    projected: pd.DataFrame,
    pitcher_outs: pd.DataFrame,
    top_n: int,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    list[ParlayTicket] | None,
    str | None,
    dict[str, Any],
]:
    meta: dict[str, Any] = {"event_ids_count": 0, "prop_rows": 0}
    empty_props = pd.DataFrame(columns=_PROP_COLUMNS)
    edge_health = EdgeSheetHealthReport()

    batter_sheet = build_batter_tb_edge_sheet(projected, empty_props, edge_health=edge_health)
    pitcher_sheet = build_pitcher_outs_edge_sheet(pitcher_outs, empty_props, edge_health=edge_health)
    parlay_tickets = build_diversified_tickets(batter_sheet)

    meta["edge_health"] = edge_health.to_dict()

    if source == "mock":
        return (
            None,
            batter_sheet,
            pitcher_sheet,
            None,
            parlay_tickets,
            "Player prop edges require --source live (mock game IDs are not live market events).",
            meta,
        )

    try:
        odds_event_ids = frames.get("odds_event_ids") or []
        if not odds_event_ids:
            odds_event_ids = resolve_odds_event_ids_for_slate(frames["slate_games"])
        meta["event_ids_count"] = len(odds_event_ids)

        if not odds_event_ids:
            return (
                None,
                batter_sheet,
                pitcher_sheet,
                None,
                parlay_tickets,
                "No live market event IDs matched for this slate; skipping prop fetch.",
                meta,
            )

        prop_lines = fetch_all_player_props(odds_event_ids)
        meta["prop_rows"] = len(prop_lines)

        batter_sheet = build_batter_tb_edge_sheet(projected, prop_lines, edge_health=edge_health)
        pitcher_sheet = build_pitcher_outs_edge_sheet(pitcher_outs, prop_lines, edge_health=edge_health)
        meta["edge_health"] = edge_health.to_dict()
        parlay_tickets = build_diversified_tickets(batter_sheet)

        if prop_lines.empty:
            return (
                prop_lines,
                batter_sheet,
                pitcher_sheet,
                None,
                parlay_tickets,
                "No matched player prop lines for edge calculation.",
                meta,
            )

        conviction = aggregate_top_conviction(batter_sheet, pitcher_sheet, top_n=top_n)
        if conviction.empty:
            return (
                prop_lines,
                batter_sheet,
                pitcher_sheet,
                None,
                parlay_tickets,
                "No matched player prop lines for edge calculation.",
                meta,
            )
        return prop_lines, batter_sheet, pitcher_sheet, conviction, parlay_tickets, None, meta
    except Exception as exc:
        logger.warning("Conviction summary unavailable: %s", exc)
        return (
            None,
            batter_sheet,
            pitcher_sheet,
            None,
            parlay_tickets,
            "Conviction summary unavailable (check THE_ODDS_API_KEY and prop market access).",
            meta,
        )


def run_slate(
    *,
    source: SlateSource = DEFAULT_SLATE_SOURCE,
    slate_date: date | None = None,
    conviction_top_n: int = 10,
) -> SlateRunResult:
    """Run the full slate projection stack: ingest, injuries, markets, conviction."""
    from baseball_props.data.data_health import DataHealthReport

    effective_date = slate_date
    if source == "live" and effective_date is None:
        effective_date = date.today()

    frames = load_slate_frames(source=source, slate_date=effective_date)
    health = frames.get("data_health")
    if health is None:
        health = DataHealthReport()
    injury_lookup = fetch_active_injuries(data_health=health)
    lineups = frames.get("lineups")
    if lineups is not None and not lineups.empty:
        frames["lineups"] = filter_injured_from_lineups(lineups, injury_lookup)
    context = build_slate_context(
        frames,
        injury_lookup=injury_lookup,
        slate_date=effective_date,
        data_health=health,
    )
    projected = project_player_rates(context.player_games)

    pa = project_batter_pa(frames["lineups"], frames["vegas_totals"], frames["slate_games"])
    projected = projected.merge(
        pa[["game_id", "player_id", "proj_pa"]], on=["game_id", "player_id"]
    )
    projected = project_batter_total_bases(projected, injury_lookup=injury_lookup)
    projected = project_batter_hits(projected, injury_lookup=injury_lookup)

    if "player_name" not in projected.columns:
        projected["player_name"] = resolve_display_name(
            projected, "player_id", "player_name"
        )

    pitcher_outs = project_pitcher_outs_and_pitches(
        frames["slate_games"],
        frames["pitcher_tendencies"],
        projected,
        frames["team_pitching"],
    )
    pitcher_outs = _resolve_pitcher_names(pitcher_outs, frames["pitcher_tendencies"])

    (
        prop_lines,
        batter_sheet,
        pitcher_sheet,
        conviction,
        parlay_tickets,
        conviction_message,
        conv_meta,
    ) = _build_edge_outputs(
        source=source,
        frames=frames,
        projected=projected,
        pitcher_outs=pitcher_outs,
        top_n=conviction_top_n,
    )

    meta = {
        "source": source,
        "fallback_counts": dict(context.fallback_counts),
        "n_players": context.n_players,
        "lineup_sources": dict(frames.get("lineup_source_counts") or {}),
        "ticket_count": len(parlay_tickets or []),
        **conv_meta,
    }
    if context.data_health is not None:
        merged_health = context.data_health.to_dict()
        if "edge_health" in meta:
            merged_health["edge_skip_counts"] = meta["edge_health"].get("edge_skip_counts", {})
        meta["data_health"] = merged_health
    elif "edge_health" in meta:
        meta["data_health"] = meta["edge_health"]

    return SlateRunResult(
        frames=frames,
        context=context,
        projected=projected,
        pitcher_outs=pitcher_outs,
        prop_lines=prop_lines,
        batter_edge_sheet=batter_sheet,
        pitcher_edge_sheet=pitcher_sheet,
        conviction=conviction,
        conviction_message=conviction_message,
        parlay_tickets=parlay_tickets,
        meta=meta,
    )
