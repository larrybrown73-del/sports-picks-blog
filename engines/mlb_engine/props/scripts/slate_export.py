"""Shared slate CSV/JSON/canvas export helpers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from baseball_props.analysis.game_splits import build_betting_intel_export, build_game_split_export
from baseball_props.analysis.parlay_builder import tickets_to_records
from baseball_props.pipeline.slate_run import SlateRunResult

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXPORT_DIR = ROOT / "exports" / "today"
DEFAULT_CANVAS_PATH = (
    Path.home()
    / ".cursor"
    / "projects"
    / "d-Juniors-Files-baseball-props-model"
    / "canvases"
    / "mlb-props-july-3.canvas.tsx"
)


def export_csvs(
    export_path: Path,
    *,
    batter_sheet: pd.DataFrame | None,
    pitcher_sheet: pd.DataFrame | None,
    conviction: pd.DataFrame | None,
    projected: pd.DataFrame | None = None,
) -> None:
    if export_path.suffix.lower() == ".csv":
        base = export_path.with_suffix("")
        out_dir = export_path.parent
    else:
        base = export_path / "slate_edges"
        out_dir = export_path

    out_dir.mkdir(parents=True, exist_ok=True)

    if batter_sheet is not None and not batter_sheet.empty:
        batter_sheet.to_csv(out_dir / f"{base.name}_batter_edges.csv", index=False)
    if pitcher_sheet is not None and not pitcher_sheet.empty:
        pitcher_sheet.to_csv(out_dir / f"{base.name}_pitcher_edges.csv", index=False)
    if conviction is not None and not conviction.empty:
        conviction.to_csv(out_dir / f"{base.name}_conviction.csv", index=False)
    if projected is not None and not projected.empty and "proj_hits" in projected.columns:
        hits_cols = [
            c
            for c in [
                "player_name",
                "team_id",
                "lineup_slot",
                "proj_pa",
                "proj_hits",
                "proj_total_bases",
                "hits_per_pa",
                "game_id",
            ]
            if c in projected.columns
        ]
        projected[hits_cols].sort_values("proj_hits", ascending=False).to_csv(
            out_dir / f"{base.name}_hits_projections.csv",
            index=False,
        )

    print(f"\nExported edge sheets to {out_dir}")


def export_parlay_tickets(export_path: Path, result: SlateRunResult) -> None:
    if not result.parlay_tickets:
        return
    if export_path.suffix.lower() == ".csv":
        out_dir = export_path.parent
        base_name = export_path.with_suffix("").name
    else:
        out_dir = export_path
        base_name = "slate_edges"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{base_name}_parlay_tickets.json"
    out_file.write_text(
        json.dumps(tickets_to_records(result.parlay_tickets), indent=2),
        encoding="utf-8",
    )
    print(f"Exported parlay tickets to {out_file}")


def export_data_health(export_path: Path, result: SlateRunResult) -> None:
    if not result.meta.get("data_health"):
        return
    if export_path.suffix.lower() == ".csv":
        health_dir = export_path.parent
        health_base = export_path.with_suffix("").name
    else:
        health_dir = export_path
        health_base = "slate_edges"
    health_dir.mkdir(parents=True, exist_ok=True)
    health_file = health_dir / f"{health_base}_data_health.json"
    health_file.write_text(
        json.dumps(result.meta["data_health"], indent=2),
        encoding="utf-8",
    )
    print(f"Exported data health report to {health_file}")


def export_canvas_games(
    export_path: Path,
    result: SlateRunResult,
    *,
    include_pitch_locations: bool = True,
    slate_date: date | None = None,
    market_line: float | None = None,
    side: str | None = None,
    top_n: int | None = None,
) -> Path:
    if export_path.suffix.lower() == ".csv":
        out_dir = export_path.parent
    else:
        out_dir = export_path
    out_dir.mkdir(parents=True, exist_ok=True)
    games = build_game_split_export(
        result,
        include_pitch_locations=include_pitch_locations,
    )
    out_file = out_dir / "canvas_games.json"
    out_file.write_text(json.dumps(games, indent=2), encoding="utf-8")
    print(f"Exported canvas game splits to {out_file}")

    intel = build_betting_intel_export(
        result,
        slate_date=slate_date,
        market_line=market_line,
        side=side,
        top_n=top_n,
    )
    intel_file = out_dir / "canvas_betting_intel.json"
    intel_file.write_text(json.dumps(intel, indent=2), encoding="utf-8")
    print(f"Exported canvas betting intel to {intel_file}")
    return out_file
