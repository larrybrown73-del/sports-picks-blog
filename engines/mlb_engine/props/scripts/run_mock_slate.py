"""End-to-end mock slate demo."""



from __future__ import annotations



import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))

from baseball_props.analysis.edge_sheets import (
    PASS_NO_DATA,
    SKIP_DISPLAY_RECOMMENDATIONS,
)
from baseball_props.config import DEFAULT_SLATE_SOURCE
from baseball_props.pipeline.slate_run import SlateRunResult, run_slate
from scripts.slate_export import (
    DEFAULT_CANVAS_PATH,
    DEFAULT_EXPORT_DIR,
    export_canvas_games,
    export_csvs,
    export_data_health,
    export_parlay_tickets,
)



BATTER_EDGE_DISPLAY: dict[str, str] = {

    "player_name": "Player",

    "team_id": "Team",

    "proj_tb": "Proj TB",

    "market_line": "Market Line",

    "over_under_odds": "Over/Under Odds",

    "probability_pct": "Probability %",

    "edge_pct": "Edge %",

    "ev_per_unit": "EV",

    "confidence_tier": "Tier",

    "kelly_fraction": "Kelly %",

    "suggested_stake": "Stake $",

    "recommendation": "Recommendation",

}



BATTER_EDGE_COL_SPACE: dict[str, int] = {

    "Player": 18,

    "Team": 5,

    "Proj TB": 8,

    "Market Line": 12,

    "Over/Under Odds": 16,

    "Probability %": 14,

    "Edge %": 8,

    "EV": 8,

    "Tier": 22,

    "Kelly %": 8,

    "Stake $": 8,

    "Recommendation": 14,

}



HITS_PROJECTION_DISPLAY: dict[str, str] = {

    "player_name": "Player",

    "team_id": "Team",

    "lineup_slot": "Slot",

    "proj_pa": "Proj PA",

    "proj_hits": "Proj Hits",

    "proj_total_bases": "Proj TB",

}



HITS_PROJECTION_COL_SPACE: dict[str, int] = {

    "Player": 22,

    "Team": 5,

    "Slot": 5,

    "Proj PA": 8,

    "Proj Hits": 10,

    "Proj TB": 8,

}



PITCHER_EDGE_DISPLAY: dict[str, str] = {

    "pitcher_name": "Pitcher",

    "team_id": "Team",

    "proj_outs": "Proj Outs",

    "proj_pitch_count": "Proj Pitches",

    "pitches_per_out_baseline": "Pitches/Out Baseline",

    "market_line": "Market Line",

    "edge_pct": "Edge %",

    "ev_per_unit": "EV",

    "confidence_tier": "Tier",

    "kelly_fraction": "Kelly %",

    "suggested_stake": "Stake $",

    "recommendation": "Recommendation",

}



PITCHER_EDGE_COL_SPACE: dict[str, int] = {

    "Pitcher": 18,

    "Team": 5,

    "Proj Outs": 9,

    "Proj Pitches": 12,

    "Pitches/Out Baseline": 20,

    "Market Line": 12,

    "Edge %": 8,

    "EV": 8,

    "Tier": 22,

    "Kelly %": 8,

    "Stake $": 8,

    "Recommendation": 14,

}



CONVICTION_DISPLAY: dict[str, str] = {

    "player_name": "Player",

    "market": "Market",

    "model_value": "Proj",

    "market_line": "Market Line",

    "probability_pct": "Probability %",

    "edge_pct": "Edge %",

    "ev_per_unit": "EV",

    "confidence_tier": "Tier",

    "kelly_fraction": "Kelly %",

    "suggested_stake": "Stake $",

    "recommendation": "Recommendation",

}



CONVICTION_COL_SPACE: dict[str, int] = {

    "Player": 18,

    "Market": 18,

    "Proj": 8,

    "Market Line": 12,

    "Probability %": 14,

    "Edge %": 8,

    "EV": 8,

    "Tier": 22,

    "Kelly %": 8,

    "Stake $": 8,

    "Recommendation": 14,

}



VEGAS_DISPLAY_COLUMNS: dict[str, str] = {

    "home_implied_runs": "Home Implied Runs",

    "away_implied_runs": "Away Implied Runs",

    "game_total": "Game Total",

}



VEGAS_COL_SPACE: dict[str, int] = {

    "Home Implied Runs": 18,

    "Away Implied Runs": 18,

    "Game Total": 11,

}



MARKET_DISPLAY_NAMES: dict[str, str] = {

    "batter_total_bases": "Total Bases",

    "pitcher_outs": "Pitcher Outs",

}





def _format_for_display(df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:

    """Return a copy with display-only columns and renamed headers for terminal output."""

    present = {col: label for col, label in column_map.items() if col in df.columns}

    return df[list(present.keys())].rename(columns=present)


def _displayable_edge_rows(df: pd.DataFrame, *, verbose: bool) -> pd.DataFrame:
    """Hide Pass (No Data) / No line rows unless verbose mode is enabled."""
    if df.empty or verbose or "recommendation" not in df.columns:
        return df
    return df[~df["recommendation"].isin(SKIP_DISPLAY_RECOMMENDATIONS)].copy()


def _count_skipped_no_data(df: pd.DataFrame) -> int:
    if df.empty or "recommendation" not in df.columns:
        return 0
    return int(df["recommendation"].isin(SKIP_DISPLAY_RECOMMENDATIONS).sum())


def _format_numeric_display(value: object, *, fmt: str, na_label: str = "N/A") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return na_label
    try:
        if pd.isna(value):
            return na_label
    except (TypeError, ValueError):
        return na_label
    return fmt.format(value)





def _format_edge_sheet_values(df: pd.DataFrame, *, use_na: bool = False) -> pd.DataFrame:

    out = df.copy()
    missing = "N/A" if use_na else ""

    if "market_line" in out.columns:

        out["market_line"] = out["market_line"].map(
            lambda x: _format_numeric_display(x, fmt="{:.1f}", na_label=missing)
        )

    if "probability_pct" in out.columns:

        out["probability_pct"] = out["probability_pct"].map(
            lambda x: _format_numeric_display(x, fmt="{:.1f}%", na_label=missing)
        )

    if "edge_pct" in out.columns:

        out["edge_pct"] = out["edge_pct"].map(
            lambda x: _format_numeric_display(x, fmt="{:+.1f}%", na_label=missing)
        )

    if "ev_per_unit" in out.columns:

        out["ev_per_unit"] = out["ev_per_unit"].map(
            lambda x: _format_numeric_display(x, fmt="{:+.3f}", na_label=missing)
        )

    if "kelly_fraction" in out.columns:

        out["kelly_fraction"] = out["kelly_fraction"].map(
            lambda x: _format_numeric_display(
                float(x) * 100.0 if x is not None and not pd.isna(x) else x,
                fmt="{:.2f}%",
                na_label=missing,
            )
        )

    if "suggested_stake" in out.columns:

        out["suggested_stake"] = out["suggested_stake"].map(
            lambda x: _format_numeric_display(x, fmt="${:.2f}", na_label=missing)
        )

    if "proj_tb" in out.columns:

        out["proj_tb"] = out["proj_tb"].map(
            lambda x: _format_numeric_display(x, fmt="{:.2f}", na_label=missing)
        )

    if "proj_outs" in out.columns:

        out["proj_outs"] = out["proj_outs"].map(
            lambda x: _format_numeric_display(x, fmt="{:.1f}", na_label=missing)
        )

    if "proj_pitch_count" in out.columns:

        out["proj_pitch_count"] = out["proj_pitch_count"].map(
            lambda x: _format_numeric_display(x, fmt="{:.1f}", na_label=missing)
        )

    if "pitches_per_out_baseline" in out.columns:

        out["pitches_per_out_baseline"] = out["pitches_per_out_baseline"].map(
            lambda x: _format_numeric_display(x, fmt="{:.1f}", na_label=missing)
        )

    return out





def _format_conviction_values(df: pd.DataFrame, *, use_na: bool = False) -> pd.DataFrame:

    out = _format_edge_sheet_values(df.copy(), use_na=use_na)

    if "market" in out.columns:

        out["market"] = out["market"].map(

            lambda m: MARKET_DISPLAY_NAMES.get(str(m), str(m)) if pd.notna(m) else ""

        )

    if "model_value" in out.columns:

        out["model_value"] = out["model_value"].map(
            lambda x: _format_numeric_display(x, fmt="{:.2f}", na_label="N/A" if use_na else "")
        )

    return out





def _print_table(

    df: pd.DataFrame,

    column_map: dict[str, str],

    col_space: dict[str, int] | int,

) -> None:

    display = _format_for_display(df, column_map)

    print(display.to_string(index=False, col_space=col_space))





def main() -> None:

    parser = argparse.ArgumentParser(description="Run baseball props slate projection demo")

    parser.add_argument(

        "--source",

        choices=["mock", "live"],

        default=DEFAULT_SLATE_SOURCE,

        help="mock = synthetic slate; live = MLB lineups + pybaseball + live market lines",

    )

    parser.add_argument(

        "--date",

        default=None,

        help="Slate date YYYY-MM-DD (defaults to today when --source live)",

    )

    parser.add_argument(

        "--verbose",

        action="store_true",

        help="Show INFO-level API diagnostics (default: warnings and tables only)",

    )

    parser.add_argument(

        "--export-csv",

        default=str(DEFAULT_EXPORT_DIR),

        metavar="PATH",

        help="Export batter/pitcher edge sheets and conviction CSVs to PATH (dir or file prefix)",

    )

    parser.add_argument(

        "--sync-canvas",

        action="store_true",

        default=True,

        help="After export, inject canvas JSON into the slate canvas TSX file (default: on)",

    )

    parser.add_argument(

        "--no-sync-canvas",

        action="store_false",

        dest="sync_canvas",

        help="Skip syncing canvas_games.json into the .canvas.tsx file",

    )

    parser.add_argument(

        "--canvas-path",

        default=None,

        help="Optional .canvas.tsx path for --sync-canvas",

    )

    parser.add_argument(

        "--skip-pitch-locations",

        action="store_true",

        default=True,

        help="Skip Statcast pitch location fetch when exporting canvas_games.json (default: on)",

    )

    parser.add_argument(

        "--with-pitch-locations",

        action="store_false",

        dest="skip_pitch_locations",

        help="Include Statcast pitch location fetch in canvas export (slower)",

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



    if args.source == "live":
        slate_date = date.fromisoformat(args.date) if args.date else date.today()
    elif args.date:
        slate_date = date.fromisoformat(args.date)
    else:
        slate_date = None

    result = run_slate(source=args.source, slate_date=slate_date)



    pd.set_option("display.max_rows", 40)

    pd.set_option("display.width", 220)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")



    print("\n=== BATTER TOTAL BASES PROJECTIONS & MARKET EDGES ===")

    if result.batter_edge_sheet is not None and not result.batter_edge_sheet.empty:

        skipped = _count_skipped_no_data(result.batter_edge_sheet)
        display_df = _displayable_edge_rows(result.batter_edge_sheet, verbose=args.verbose)
        if display_df.empty and skipped:
            print(f"(no playable batter edges; {skipped} skipped as {PASS_NO_DATA})")
        else:
            _print_table(
                _format_edge_sheet_values(display_df, use_na=args.verbose),
                BATTER_EDGE_DISPLAY,
                BATTER_EDGE_COL_SPACE,
            )
            if skipped and not args.verbose:
                print(
                    f"Skipped {skipped} player(s) with {PASS_NO_DATA} — use --verbose to show."
                )

    else:

        print("(no batter projections)")



    print("\n=== BATTER HITS PROJECTIONS ===")

    if result.projected is not None and not result.projected.empty and "proj_hits" in result.projected.columns:

        hits_cols = [c for c in HITS_PROJECTION_DISPLAY if c in result.projected.columns]

        hits_df = result.projected[hits_cols].sort_values(

            ["proj_hits", "lineup_slot"], ascending=[False, True]

        )

        hits_display = _format_for_display(hits_df, HITS_PROJECTION_DISPLAY)

        if "Proj Hits" in hits_display.columns:

            hits_display["Proj Hits"] = hits_display["Proj Hits"].map(lambda x: f"{x:.2f}")

        if "Proj PA" in hits_display.columns:

            hits_display["Proj PA"] = hits_display["Proj PA"].map(lambda x: f"{x:.2f}")

        if "Proj TB" in hits_display.columns:

            hits_display["Proj TB"] = hits_display["Proj TB"].map(lambda x: f"{x:.2f}")

        print(hits_display.to_string(index=False, col_space=HITS_PROJECTION_COL_SPACE))

    else:

        print("(no batter hits projections)")



    print("\n=== PITCHER OUTS & WORKLOAD PROJECTIONS ===")

    if result.pitcher_edge_sheet is not None and not result.pitcher_edge_sheet.empty:

        skipped = _count_skipped_no_data(result.pitcher_edge_sheet)
        display_df = _displayable_edge_rows(result.pitcher_edge_sheet, verbose=args.verbose)
        if display_df.empty and skipped:
            print(f"(no playable pitcher edges; {skipped} skipped as {PASS_NO_DATA})")
        else:
            _print_table(
                _format_edge_sheet_values(display_df, use_na=args.verbose),
                PITCHER_EDGE_DISPLAY,
                PITCHER_EDGE_COL_SPACE,
            )
            if skipped and not args.verbose:
                print(
                    f"Skipped {skipped} pitcher(s) with {PASS_NO_DATA} — use --verbose to show."
                )

    else:

        print("(no pitcher projections)")



    print("\n=== Vegas Totals ===")

    vegas_display = _format_for_display(

        result.frames["vegas_totals"], VEGAS_DISPLAY_COLUMNS

    )

    print(vegas_display.to_string(index=False, col_space=VEGAS_COL_SPACE))



    print(f"\nFallback summary: {result.meta.get('fallback_counts', {})}")
    data_health = result.meta.get("data_health")
    if data_health:
        warning_count = data_health.get("warning_count", len(data_health.get("warnings", [])))
        print(f"Data health: {warning_count} missing-data warning(s)")
        for warning in data_health.get("warnings", [])[:3]:
            print(f"  - {warning}")
        if warning_count > 3:
            print(f"  ... and {warning_count - 3} more")
    lineup_sources = result.meta.get("lineup_sources") or {}
    if lineup_sources:
        print(f"Lineup sources: {lineup_sources}")
        if lineup_sources.get("boxscore", 0) < sum(lineup_sources.values()):
            print(
                "Note: Some games use projected/previous lineups — run closer to first pitch "
                "for posted MLB boxscore orders, or check --verbose for per-game sources."
            )
    elif args.source == "mock":
        print("Lineup source: mock fixture (use --source live for today's MLB schedule)")

    print(f"Total player-games: {result.meta.get('n_players', 0)}")



    print("\n=== TB O1.5 FILTERED PLAYS ===")
    if result.batter_edge_sheet is not None and not result.batter_edge_sheet.empty:
        tb_plays = result.batter_edge_sheet[
            (result.batter_edge_sheet["market_line"] == 1.5)
            & result.batter_edge_sheet["verdict"].notna()
        ]
        if not tb_plays.empty:
            display_cols = [
                "player_name",
                "team_id",
                "lineup_slot",
                "proj_tb",
                "edge_pct",
                "verdict",
                "recommendation",
                "warnings",
                "alt_market",
            ]
            present = [c for c in display_cols if c in tb_plays.columns]
            print(tb_plays[present].to_string(index=False))
        else:
            print("(no Over 1.5 TB lines evaluated)")
    else:
        print("(no batter edge sheet)")

    print("\n=== MODEL HIGHEST CONVICTION PREDICTIONS ===")

    if result.conviction is not None and not result.conviction.empty:

        _print_table(

            _format_conviction_values(result.conviction, use_na=args.verbose),

            CONVICTION_DISPLAY,

            CONVICTION_COL_SPACE,

        )

    elif result.conviction_message:

        print(result.conviction_message)

    print("\n=== DIVERSIFIED PARLAY TICKETS ===")
    if result.parlay_tickets:
        for ticket in result.parlay_tickets:
            leg_labels = ", ".join(
                f"{leg.player_name} {leg.recommendation} {leg.line} ({leg.edge_pct:+.1f}%)"
                for leg in ticket.legs
            )
            print(
                f"Ticket {ticket.ticket_id} "
                f"(avg edge {ticket.combined_edge_proxy:+.1f}%): {leg_labels}"
            )
    else:
        print("(no qualifying Over 1.5 TB legs after filters)")

    if args.export_csv:

        export_path = Path(args.export_csv)

        export_csvs(
            export_path,
            batter_sheet=result.batter_edge_sheet,
            pitcher_sheet=result.pitcher_edge_sheet,
            conviction=result.conviction,
            projected=result.projected,
        )
        export_parlay_tickets(export_path, result)
        export_data_health(export_path, result)

        canvas_json = export_canvas_games(
            export_path,
            result,
            include_pitch_locations=not args.skip_pitch_locations,
            slate_date=slate_date,
        )

        if args.sync_canvas:

            from scripts.sync_canvas_splits import sync_canvas

            canvas_target = Path(args.canvas_path) if args.canvas_path else DEFAULT_CANVAS_PATH

            sync_canvas(canvas_target, canvas_json, intel_json=canvas_json.parent / "canvas_betting_intel.json")





if __name__ == "__main__":

    main()

