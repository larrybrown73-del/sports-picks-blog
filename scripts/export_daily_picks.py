#!/usr/bin/env python3
"""Export daily MLB picks from baseball-predictor and baseball-props-model to JSON."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKS_DIR = PROJECT_ROOT / "data" / "picks"
POWERBI_PICKS_DIR = PICKS_DIR / "powerbi"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
UNIT_STAKE = 100.0

MLB_ENGINE_ROOT = PROJECT_ROOT / "engines" / "mlb_engine"
DEFAULT_PREDICTOR_PATH = MLB_ENGINE_ROOT / "predictor"
DEFAULT_PROPS_PATH = MLB_ENGINE_ROOT / "props"

PROPS_TIMEOUT_SECONDS = int(os.environ.get("PROPS_EXPORT_TIMEOUT_SECONDS", "1800"))


def load_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from .env.local."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_paths() -> tuple[Path, Path]:
    env = load_env_file(PROJECT_ROOT / ".env.local")
    predictor_raw = (
        os.environ.get("BASEBALL_PREDICTOR_PATH")
        or env.get("BASEBALL_PREDICTOR_PATH")
        or str(DEFAULT_PREDICTOR_PATH)
    )
    props_raw = (
        os.environ.get("BASEBALL_PROPS_PATH")
        or env.get("BASEBALL_PROPS_PATH")
        or str(DEFAULT_PROPS_PATH)
    )
    predictor_path = Path(predictor_raw)
    props_path = Path(props_raw)
    if not predictor_path.is_absolute():
        predictor_path = (PROJECT_ROOT / predictor_path).resolve()
    if not props_path.is_absolute():
        props_path = (PROJECT_ROOT / props_path).resolve()
    return predictor_path, props_path


def add_to_syspath(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_props_env(props_path: Path) -> None:
    env_file = props_path / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_predictor_env(predictor_path: Path) -> None:
    env_file = predictor_path / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_existing_picks(game_date: date) -> dict[str, Any] | None:
    path = PICKS_DIR / f"{game_date.isoformat()}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _merge_with_existing(payload: dict[str, Any], game_date: date) -> dict[str, Any]:
    existing = _load_existing_picks(game_date)
    if not existing:
        return payload

    if not payload.get("slate") and existing.get("slate"):
        payload["slate"] = existing["slate"]
        print("  Preserved slate from previous export")

    existing_props = existing.get("propPicks") or {}
    new_props = payload.get("propPicks") or {}
    if not new_props.get("conviction") and existing_props.get("conviction"):
        payload["propPicks"] = existing_props
        print("  Preserved prop picks from previous export")

    return payload


def _fetch_schedule_game_count(props_path: Path, game_date: date) -> int:
    add_to_syspath(props_path)
    load_props_env(props_path)
    try:
        from baseball_props.data.mlb_live import fetch_todays_schedule

        return len(fetch_todays_schedule(game_date))
    except Exception:
        return 0


def _run_grade_yesterday(game_date: date) -> None:
    grade_script = Path(__file__).resolve().parent / "grade_picks.py"
    yesterday = game_date - timedelta(days=1)
    try:
        subprocess.run(
            [sys.executable, str(grade_script), yesterday.isoformat()],
            check=False,
            timeout=120,
        )
    except Exception as exc:
        print(f"  Warning: could not grade yesterday's picks: {exc}")


def _load_graded_result_lookup(game_date: date) -> dict[tuple[str, str], dict[str, Any]]:
    path = RESULTS_DIR / f"{game_date.isoformat()}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("picks") or []:
        matchup = str(row.get("matchup") or "")
        if " @ " not in matchup:
            continue
        away, home = matchup.split(" @ ", 1)
        lookup[(away.strip(), home.strip())] = row
    return lookup


def _american_to_implied(american_odds: int) -> float:
    odds = int(american_odds)
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _confidence_score(edge_pct: float, model_prob: float) -> int:
    return min(100, max(0, round(edge_pct * 4.0 + (model_prob - 0.50) * 80)))


def _confidence_label(score: int) -> str:
    if score >= 85:
        return "Elite"
    if score >= 70:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def serialize_power_bi_pick(pick: Any, graded: dict[str, Any] | None = None) -> dict[str, Any]:
    """Map a SlatePick to the flat Power BI pick schema (numeric types preserved)."""
    win_probability = float(pick.model_prob)
    implied_probability = float(pick.market_prob)
    edge_pct = float(pick.edge_pct)
    edge_decimal = win_probability - implied_probability
    opening_line = int(pick.american_odds)
    closing_line = opening_line
    opening_implied = _american_to_implied(opening_line)
    closing_implied = _american_to_implied(closing_line)
    confidence_score = _confidence_score(edge_pct, win_probability)
    ev_pct = getattr(pick, "ev_pct", None)
    if ev_pct is None and getattr(pick, "ev_per_unit", None) is not None:
        ev_pct = float(pick.ev_per_unit) * 100.0
    expected_value = round(float(ev_pct or 0.0), 1)
    result = "Pending"
    profit_loss = 0.0
    if graded:
        graded_result = str(graded.get("result") or "Pending")
        if graded_result in {"Win", "Loss", "Pending"}:
            result = graded_result
        elif graded_result:
            result = graded_result
        try:
            profit_loss = float(graded.get("profitLoss") or 0.0)
        except (TypeError, ValueError):
            profit_loss = 0.0

    units_won = 0.0
    if result in {"Win", "Loss"}:
        units_won = round(profit_loss / UNIT_STAKE, 2)

    home_away = "Home" if pick.play == pick.home_name else "Away"

    return {
        "matchup": f"{pick.away_name} @ {pick.home_name}",
        "homeTeam": pick.home_name,
        "awayTeam": pick.away_name,
        "pick": pick.play,
        "sportsbook": pick.book or "",
        "betType": "Moneyline",
        "league": "MLB",
        "homeAway": home_away,
        "confidence": confidence_score,
        "confidenceLabel": _confidence_label(confidence_score),
        "edge": round(edge_pct, 1),
        "closingLineValue": round(closing_implied - opening_implied, 2),
        "expectedValue": expected_value,
        "winProbability": round(win_probability, 6),
        "edgeDecimal": round(edge_decimal, 6),
        "openingLine": opening_line,
        "closingLine": closing_line,
        "impliedProbability": round(implied_probability, 6),
        "result": result,
        "profitLoss": float(round(profit_loss, 2)),
        "unitsWon": units_won,
        "weather": getattr(pick, "weather", "") or "",
        "temperature": int(getattr(pick, "temperature_f", 0) or 0),
        "umpire": getattr(pick, "umpire", "") or "",
        "startingPitcher": getattr(pick, "starting_pitcher", "") or "",
        "opposingPitcher": getattr(pick, "opposing_pitcher", "") or "",
    }


def serialize_moneyline_picks(predictor_path: Path, game_date: date) -> list[dict[str, Any]]:
    add_to_syspath(predictor_path)
    load_predictor_env(predictor_path)
    from run_odds_slate import evaluate_slate

    picks = evaluate_slate(game_date, write_log=False)
    graded_lookup = _load_graded_result_lookup(game_date)
    rows: list[dict[str, Any]] = []
    for pick in picks:
        graded = graded_lookup.get((pick.away_name, pick.home_name))
        rows.append(serialize_power_bi_pick(pick, graded))
    return rows


def serialize_slate(props_path: Path, game_date: date) -> list[dict[str, Any]]:
    add_to_syspath(props_path)
    load_props_env(props_path)

    from baseball_props.data.mlb_live import build_slate_from_schedule, fetch_todays_schedule

    slate_games, lineups, pitcher_names, _ = build_slate_from_schedule(game_date)
    if slate_games.empty:
        return []

    schedule = fetch_todays_schedule(game_date)
    meta_by_pk: dict[str, dict[str, Any]] = {}
    for game in schedule:
        game_pk = str(game["gamePk"])
        home = game.get("teams", {}).get("home", {})
        away = game.get("teams", {}).get("away", {})
        meta_by_pk[game_pk] = {
            "awayTeam": away.get("team", {}).get("name", ""),
            "homeTeam": home.get("team", {}).get("name", ""),
            "awayPitcher": away.get("probablePitcher", {}).get("fullName"),
            "homePitcher": home.get("probablePitcher", {}).get("fullName"),
        }

    games: list[dict[str, Any]] = []
    for _, row in slate_games.iterrows():
        game_id = str(row["game_id"])
        away_abbrev = str(row["away_team_id"])
        home_abbrev = str(row["home_team_id"])
        meta = meta_by_pk.get(game_id, {})

        sp_away = str(row.get("sp_away_id") or "")
        sp_home = str(row.get("sp_home_id") or "")

        game_lineups = lineups[lineups["game_id"].astype(str) == game_id]
        away_rows = game_lineups[game_lineups["team_id"].astype(str) == away_abbrev].sort_values(
            "lineup_slot"
        )
        home_rows = game_lineups[game_lineups["team_id"].astype(str) == home_abbrev].sort_values(
            "lineup_slot"
        )

        games.append(
            {
                "gameId": game_id,
                "awayTeam": meta.get("awayTeam") or away_abbrev,
                "homeTeam": meta.get("homeTeam") or home_abbrev,
                "awayAbbrev": away_abbrev,
                "homeAbbrev": home_abbrev,
                "awayPitcher": meta.get("awayPitcher") or pitcher_names.get(sp_away),
                "homePitcher": meta.get("homePitcher") or pitcher_names.get(sp_home),
                "awayLineup": [
                    {
                        "slot": int(r["lineup_slot"]),
                        "name": str(r["player_name"]),
                        "playerId": str(r["player_id"]),
                    }
                    for _, r in away_rows.iterrows()
                ],
                "homeLineup": [
                    {
                        "slot": int(r["lineup_slot"]),
                        "name": str(r["player_name"]),
                        "playerId": str(r["player_id"]),
                    }
                    for _, r in home_rows.iterrows()
                ],
            }
        )

    return games


def _parse_warnings(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    text = str(value).strip()
    return [text] if text else []


def _row_to_prop_pick(row: pd.Series, *, player_key: str) -> dict[str, Any]:
    probability_pct = row.get("probability_pct")
    model_prob = None
    if pd.notna(probability_pct):
        model_prob = round(float(probability_pct) / 100.0, 4)

    model_value = row.get("model_value")
    if pd.isna(model_value):
        model_value = row.get("proj_tb")
    if pd.isna(model_value):
        model_value = row.get("proj_outs")

    return {
        "player": str(row.get(player_key, "")),
        "market": str(row.get("market", "")),
        "line": float(row["market_line"]) if pd.notna(row.get("market_line")) else 0.0,
        "recommendation": str(row.get("recommendation", "")),
        "edgePct": float(row["edge_pct"]) if pd.notna(row.get("edge_pct")) else 0.0,
        "modelProb": model_prob,
        "modelValue": float(model_value) if pd.notna(model_value) else None,
        "evPerUnit": float(row["ev_per_unit"]) if pd.notna(row.get("ev_per_unit")) else None,
        "fractionalKellyPct": float(row["fractional_kelly_pct"])
        if pd.notna(row.get("fractional_kelly_pct"))
        else None,
        "confidenceTier": str(row.get("confidence_tier"))
        if pd.notna(row.get("confidence_tier"))
        else None,
        "confidenceScore": int(row["confidence_score"])
        if pd.notna(row.get("confidence_score"))
        else None,
        "dataWarnings": _parse_warnings(row.get("data_warnings")),
        "verdict": str(row.get("verdict")) if pd.notna(row.get("verdict")) else None,
    }


def _dataframe_to_prop_picks(df: pd.DataFrame | None, *, player_key: str) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []

    matched = df[df["market_line"].notna() & df["edge_pct"].notna()]
    picks = [_row_to_prop_pick(row, player_key=player_key) for _, row in matched.iterrows()]
    picks.sort(key=lambda item: abs(item["edgePct"]), reverse=True)
    return picks


def serialize_prop_picks(props_path: Path, game_date: date) -> dict[str, list[dict[str, Any]]]:
    add_to_syspath(props_path)
    load_props_env(props_path)
    from baseball_props.pipeline import run_slate

    result = run_slate(source="live", slate_date=game_date)

    conviction = _dataframe_to_prop_picks(result.conviction, player_key="player_name")
    batter_edges = _dataframe_to_prop_picks(result.batter_edge_sheet, player_key="player_name")
    pitcher_edges = _dataframe_to_prop_picks(result.pitcher_edge_sheet, player_key="pitcher_name")

    return {
        "conviction": conviction,
        "batterEdges": batter_edges,
        "pitcherEdges": pitcher_edges,
    }


def read_performance_stats(predictor_path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "accuracy": "N/A",
        "roi": "N/A",
        "netProfit": "N/A",
        "brierScore": "N/A",
        "gamesScored": 0,
        "betsPlaced": 0,
    }
    csv_path = predictor_path / "system_performance_log.csv"
    if not csv_path.exists():
        return defaults

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"  Warning: performance log is empty: {csv_path}")
            return defaults

        row = df.iloc[-1]
        return {
            "accuracy": str(row["Accuracy"]),
            "roi": str(row["ROI"]),
            "netProfit": str(row["Net_Profit"]),
            "brierScore": str(row["Brier_Score"]),
            "gamesScored": int(row["Games_Scored"]),
            "betsPlaced": int(row["Bets_Placed"]),
        }
    except Exception as exc:
        print(f"  Warning: could not read performance stats: {exc}")
        return defaults


def _run_props_subprocess(game_date: date) -> dict[str, list[dict[str, Any]]] | None:
    temp_path = PICKS_DIR / f".props-temp-{game_date.isoformat()}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--props-only",
        game_date.isoformat(),
        str(temp_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=PROPS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"  Warning: props export timed out after {PROPS_TIMEOUT_SECONDS}s")
        return None
    except subprocess.CalledProcessError as exc:
        print(f"  Warning: props subprocess failed: {exc}")
        return None

    if not temp_path.exists():
        return None

    try:
        return json.loads(temp_path.read_text(encoding="utf-8"))
    finally:
        temp_path.unlink(missing_ok=True)


def export_daily_picks(
    game_date: date | None = None,
    *,
    skip_props: bool = False,
    props_subprocess: bool = True,
) -> Path:
    game_date = game_date or date.today()
    predictor_path, props_path = resolve_paths()

    print(f"Exporting picks for {game_date.isoformat()}")
    print(f"  Predictor: {predictor_path}")
    print(f"  Props:     {props_path}")

    moneyline_picks: list[dict[str, Any]] = []
    slate: list[dict[str, Any]] = []
    prop_picks: dict[str, list[dict[str, Any]]] = {
        "conviction": [],
        "batterEdges": [],
        "pitcherEdges": [],
    }

    if predictor_path.exists():
        try:
            moneyline_picks = serialize_moneyline_picks(predictor_path, game_date)
            print(f"  Moneyline picks: {len(moneyline_picks)}")
        except Exception as exc:
            print(f"  Warning: moneyline export failed: {exc}")
            traceback.print_exc()
    else:
        print(f"  Warning: predictor path not found: {predictor_path}")

    if props_path.exists():
        try:
            slate = serialize_slate(props_path, game_date)
            print(f"  Slate games: {len(slate)}")
        except Exception as exc:
            print(f"  Warning: slate export failed: {exc}")
            traceback.print_exc()
    else:
        print(f"  Warning: props path not found: {props_path}")

    if props_path.exists() and not skip_props:
        try:
            if props_subprocess:
                exported = _run_props_subprocess(game_date)
                if exported is not None:
                    prop_picks = exported
            else:
                prop_picks = serialize_prop_picks(props_path, game_date)
            print(f"  Prop conviction picks: {len(prop_picks['conviction'])}")
        except Exception as exc:
            print(f"  Warning: props export failed: {exc}")
            traceback.print_exc()

    performance = (
        read_performance_stats(predictor_path)
        if predictor_path.exists()
        else {
            "accuracy": "N/A",
            "roi": "N/A",
            "netProfit": "N/A",
            "brierScore": "N/A",
            "gamesScored": 0,
            "betsPlaced": 0,
        }
    )

    schedule_games = _fetch_schedule_game_count(props_path, game_date) if props_path.exists() else 0
    if schedule_games and not slate:
        print(f"  WARNING: slate count (0) != schedule count ({schedule_games})")

    props_available = bool(
        prop_picks.get("conviction")
        or prop_picks.get("batterEdges")
        or prop_picks.get("pitcherEdges")
    )

    payload = {
        "date": game_date.isoformat(),
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "meta": {
            "slateGames": len(slate),
            "valuePicks": len(moneyline_picks),
            "propsAvailable": props_available,
            "scheduleGames": schedule_games or None,
        },
        "slate": slate,
        "moneylinePicks": moneyline_picks,
        "propPicks": prop_picks,
        "performance": performance,
    }

    payload = _merge_with_existing(payload, game_date)
    payload["meta"] = {
        "slateGames": len(payload.get("slate") or []),
        "valuePicks": len(payload.get("moneylinePicks") or []),
        "propsAvailable": bool(
            (payload.get("propPicks") or {}).get("conviction")
            or (payload.get("propPicks") or {}).get("batterEdges")
            or (payload.get("propPicks") or {}).get("pitcherEdges")
        ),
        "scheduleGames": schedule_games or payload.get("meta", {}).get("scheduleGames"),
    }

    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = PICKS_DIR / f"{game_date.isoformat()}.json"
    latest_path = PICKS_DIR / "latest.json"

    dated_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    POWERBI_PICKS_DIR.mkdir(parents=True, exist_ok=True)
    powerbi_rows = list(payload.get("moneylinePicks") or [])
    powerbi_dated = POWERBI_PICKS_DIR / f"{game_date.isoformat()}.json"
    powerbi_latest = POWERBI_PICKS_DIR / "latest.json"
    powerbi_dated.write_text(json.dumps(powerbi_rows, indent=2), encoding="utf-8")
    powerbi_latest.write_text(json.dumps(powerbi_rows, indent=2), encoding="utf-8")

    print(f"Wrote {dated_path}")
    print(f"Wrote {latest_path}")
    print(f"Wrote {powerbi_dated} ({len(powerbi_rows)} Power BI rows)")
    print(f"Wrote {powerbi_latest}")

    _run_grade_yesterday(game_date)
    return dated_path


def export_props_only(game_date: date, output_path: Path) -> None:
    _, props_path = resolve_paths()
    if not props_path.exists():
        raise FileNotFoundError(f"Props path not found: {props_path}")
    prop_picks = serialize_prop_picks(props_path, game_date)
    output_path.write_text(json.dumps(prop_picks, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--props-only":
        export_props_only(date.fromisoformat(sys.argv[2]), Path(sys.argv[3]))
    else:
        export_date = None
        skip_props = "--skip-props" in sys.argv
        if export_date_arg := next((arg for arg in sys.argv[1:] if arg[:1].isdigit()), None):
            export_date = date.fromisoformat(export_date_arg)
        export_daily_picks(export_date, skip_props=skip_props)
