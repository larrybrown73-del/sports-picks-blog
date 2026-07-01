#!/usr/bin/env python3
"""Grade published moneyline picks against final MLB results."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKS_DIR = PROJECT_ROOT / "data" / "picks"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEFAULT_PREDICTOR_PATH = Path(r"D:\Juniors Files\baseball-predictor")
UNIT_STAKE = 100.0


def load_env_file(path: Path) -> dict[str, str]:
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


def resolve_predictor_path() -> Path:
    env = load_env_file(PROJECT_ROOT / ".env.local")
    return Path(
        os.environ.get("BASEBALL_PREDICTOR_PATH")
        or env.get("BASEBALL_PREDICTOR_PATH")
        or DEFAULT_PREDICTOR_PATH
    )


def add_to_syspath(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_picks(game_date: date) -> dict[str, Any] | None:
    path = PICKS_DIR / f"{game_date.isoformat()}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_final_scores(predictor_path: Path, game_date: date) -> dict[tuple[str, str], dict[str, Any]]:
    add_to_syspath(predictor_path)
    import statsapi

    date_str = game_date.strftime("%m/%d/%Y")
    schedule = statsapi.schedule(date=date_str, sportId=1)
    results: dict[tuple[str, str], dict[str, Any]] = {}

    for game in schedule:
        if game.get("game_type") != "R":
            continue
        away_name = game.get("away_name", "")
        home_name = game.get("home_name", "")
        status = str(game.get("status", ""))
        key = (away_name, home_name)

        if status != "Final":
            results[key] = {
                "status": status,
                "away_score": None,
                "home_score": None,
                "winner": None,
            }
            continue

        try:
            linescore = statsapi.linescore(game["game_id"])
            away_runs = int(linescore["teams"]["away"]["runs"])
            home_runs = int(linescore["teams"]["home"]["runs"])
        except Exception:
            results[key] = {
                "status": status,
                "away_score": None,
                "home_score": None,
                "winner": None,
            }
            continue

        if away_runs > home_runs:
            winner = away_name
        elif home_runs > away_runs:
            winner = home_name
        else:
            winner = None

        results[key] = {
            "status": status,
            "away_score": away_runs,
            "home_score": home_runs,
            "winner": winner,
        }

    return results


def _profit_loss(american_odds: int, stake: float, won: bool) -> float:
    add_to_syspath(resolve_predictor_path())
    from backtest import american_odds_payout

    if american_odds == 0:
        return 0.0
    if won:
        return american_odds_payout(american_odds, stake)
    return -stake


def grade_picks(game_date: date | None = None) -> Path | None:
    game_date = game_date or (date.today() - timedelta(days=1))
    picks_payload = _load_picks(game_date)
    if picks_payload is None:
        print(f"No picks file found for {game_date.isoformat()}")
        return None

    moneyline_picks = picks_payload.get("moneylinePicks") or []
    if not moneyline_picks:
        print(f"No moneyline picks to grade for {game_date.isoformat()}")
        return None

    predictor_path = resolve_predictor_path()
    final_scores = _fetch_final_scores(predictor_path, game_date)

    graded: list[dict[str, Any]] = []
    wins = losses = pending = 0
    net_pl = 0.0

    for pick in moneyline_picks:
        away = pick["awayTeam"]
        home = pick["homeTeam"]
        play = pick["play"]
        american_odds = int(pick.get("americanOdds") or 0)
        matchup = f"{away} @ {home}"

        game_result = final_scores.get((away, home))
        if game_result is None:
            result = "Pending"
            profit_loss = 0.0
            pending += 1
        elif game_result["status"] != "Final":
            result = "Pending"
            profit_loss = 0.0
            pending += 1
        elif american_odds == 0:
            result = "NoLine"
            profit_loss = 0.0
            pending += 1
        elif game_result["winner"] is None:
            result = "Push"
            profit_loss = 0.0
        elif game_result["winner"] == play:
            result = "Win"
            profit_loss = _profit_loss(american_odds, UNIT_STAKE, True)
            wins += 1
            net_pl += profit_loss
        else:
            result = "Loss"
            profit_loss = _profit_loss(american_odds, UNIT_STAKE, False)
            losses += 1
            net_pl += profit_loss

        graded.append(
            {
                "date": game_date.isoformat(),
                "matchup": matchup,
                "pick": play,
                "americanOdds": american_odds,
                "result": result,
                "profitLoss": round(profit_loss, 2),
                "stake": UNIT_STAKE,
            }
        )

    payload = {
        "date": game_date.isoformat(),
        "generatedAt": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "unitStake": UNIT_STAKE,
        "picks": graded,
        "summary": {
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "netProfitLoss": round(net_pl, 2),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{game_date.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Graded {len(graded)} picks for {game_date.isoformat()} -> {out_path}")
    print(f"  Record: {wins}-{losses} ({pending} pending), P/L: {net_pl:+.2f} units")
    return out_path


if __name__ == "__main__":
    grade_date = None
    if len(sys.argv) > 1 and sys.argv[1][:1].isdigit():
        grade_date = date.fromisoformat(sys.argv[1])
    grade_picks(grade_date)
