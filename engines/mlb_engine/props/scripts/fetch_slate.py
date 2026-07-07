#!/usr/bin/env python3
"""Fetch live MLB slate, run model predictions, and export to canvas."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseball_props.data.ingest import get_odds_api_key
from baseball_props.pipeline.slate_run import run_slate
from scripts.slate_export import (
    DEFAULT_CANVAS_PATH,
    DEFAULT_EXPORT_DIR,
    export_canvas_games,
    export_csvs,
    export_data_health,
    export_parlay_tickets,
)


def _print_prediction_summary(intel: dict) -> None:
    plays = intel.get("conviction_plays") or []
    if not plays:
        print("\n(no conviction plays matched export filters)")
        return

    print("\n=== TOP CONVICTION PREDICTIONS (canvas export) ===")
    headers = ["Player", "Line", "Edge %", "EV", "Tier", "Kelly %", "Stake $", "Rec"]
    rows: list[list[str]] = []
    for play in plays:
        edge = play.get("edge")
        kelly = play.get("kelly_fraction")
        stake = play.get("suggested_stake")
        rows.append(
            [
                str(play.get("player") or ""),
                f"{play.get('line'):.1f}" if play.get("line") is not None else "—",
                f"{edge:+.1f}%" if edge is not None else "—",
                f"{play.get('ev_per_unit'):.3f}" if play.get("ev_per_unit") is not None else "—",
                str(play.get("confidence_tier") or "—"),
                f"{kelly * 100:.1f}%" if kelly is not None else "—",
                f"${stake:.0f}" if stake is not None else "—",
                str(play.get("rec") or "—"),
            ]
        )

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live MLB slate, run model edges, export canvas JSON"
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Slate date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Directory for CSV/JSON exports",
    )
    parser.add_argument(
        "--sync-canvas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sync exports into the canvas TSX after run (default: on)",
    )
    parser.add_argument(
        "--canvas-path",
        type=Path,
        default=None,
        help="Target .canvas.tsx path for --sync-canvas",
    )
    parser.add_argument(
        "--skip-pitch-locations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip Statcast pitch location fetch for faster export (default: on)",
    )
    parser.add_argument(
        "--tb-line",
        type=float,
        default=1.5,
        help="Filter canvas conviction export to this hits market line (0.5 or 1.5)",
    )
    parser.add_argument(
        "--side",
        default="Over",
        help="Filter canvas conviction export to this recommendation side",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Cap conviction rows in canvas betting intel export",
    )
    parser.add_argument(
        "--conviction-top-n",
        type=int,
        default=10,
        help="Top-N conviction rows computed during slate run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level API diagnostics",
    )
    args = parser.parse_args()

    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    for logger_name in list(logging.Logger.manager.loggerDict):
        if logger_name.startswith("baseball_props"):
            logging.getLogger(logger_name).setLevel(level)

    get_odds_api_key(required=True)

    slate_date = date.fromisoformat(args.date) if args.date else date.today()
    print(f"Fetching live slate for {slate_date.isoformat()}...")
    result = run_slate(
        source="live",
        slate_date=slate_date,
        conviction_top_n=args.conviction_top_n,
    )

    export_dir = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    export_csvs(
        export_dir,
        batter_sheet=result.batter_edge_sheet,
        pitcher_sheet=result.pitcher_edge_sheet,
        conviction=result.conviction,
        projected=result.projected,
    )
    export_parlay_tickets(export_dir, result)
    export_data_health(export_dir, result)

    canvas_json = export_canvas_games(
        export_dir,
        result,
        include_pitch_locations=not args.skip_pitch_locations,
        slate_date=slate_date,
        market_line=args.tb_line,
        side=args.side,
        top_n=args.top_n,
    )

    import json

    intel = json.loads((canvas_json.parent / "canvas_betting_intel.json").read_text(encoding="utf-8"))
    _print_prediction_summary(intel)

    summary = intel.get("summary_stats") or {}
    print(
        f"\nExported {summary.get('play_count', 0)} conviction play(s) "
        f"to {canvas_json.parent / 'canvas_betting_intel.json'}"
    )

    if args.sync_canvas:
        from scripts.sync_canvas_splits import sync_canvas

        canvas_target = args.canvas_path or DEFAULT_CANVAS_PATH
        sync_canvas(
            canvas_target,
            canvas_json,
            intel_json=canvas_json.parent / "canvas_betting_intel.json",
        )


if __name__ == "__main__":
    main()
