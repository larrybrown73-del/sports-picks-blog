"""Run the backtest harness and export daily value picks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

import baseball_data
from backtest import (
    DEFAULT_PREDICTION_SEASONS,
    EDGE_THRESHOLD,
    BacktestResult,
    _memoize_obp_lookups,
    _mock_moneylines,
    load_frozen_models,
    run_backtest,
)
from config import DEFAULT_ROLLING_WINDOW
from market.calculations import compute_wager_metrics, confidence_label_from_score
from slate_evaluation import append_slate_evaluation_log, evaluate_game

REPORT_FILE = "system_performance_log.csv"


@dataclass
class ValuePick:
    away_name: str
    home_name: str
    play: str
    quarter_kelly_pct: float
    edge_pct: float
    model_win_prob: float
    confidence_score: int
    confidence_label: str
    confidence_tier: str
    ev_per_unit: float
    american_odds: int
    temperature: str
    wind: str
    bullpen_status: str
    pred_home_runs: float
    pred_away_runs: float
    home_win_prob: float
    away_win_prob: float


def _format_log_row(result: BacktestResult) -> dict[str, str]:
    return {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Accuracy": f"{result.moneyline_winner_accuracy * 100:.1f}%",
        "ROI": f"{result.betting_roi_pct:+.2f}%",
        "Net_Profit": f"{result.net_profit_loss:+.2f}",
        "Brier_Score": f"{result.brier_score_calibrated:.4f}",
        "Games_Scored": str(result.games_scored),
        "Bets_Placed": str(result.bets_placed),
    }


def append_performance_log(result: BacktestResult, report_file: str = REPORT_FILE) -> None:
    row = _format_log_row(result)
    df = pd.DataFrame([row])
    if not os.path.exists(report_file):
        df.to_csv(report_file, index=False)
    else:
        df.to_csv(report_file, mode="a", header=False, index=False)


def _evaluate_game_pick(
    game: dict,
    *,
    models: dict,
    games_history: pd.DataFrame,
    game_date: date,
    window: int,
    edge_threshold: float = EDGE_THRESHOLD,
) -> ValuePick | None:
    try:
        evaluated = evaluate_game(game, game_date, games_history, models, window=window)
    except Exception:
        return None

    as_of_ts = pd.Timestamp(game_date)
    home_ml, away_ml, _ = _mock_moneylines(
        games_history,
        game["home_id"],
        game["away_id"],
        as_of_ts,
        window,
    )

    best_side: str | None = None
    best_edge = 0.0
    best_prob = 0.0
    best_odds = 0
    best_metrics = None

    for side, side_prob, american_odds in (
        ("home", evaluated.home_prob, home_ml),
        ("away", evaluated.away_prob, away_ml),
    ):
        metrics = compute_wager_metrics(side_prob, american_odds)
        edge = (metrics.edge_pct or 0.0) / 100.0
        if edge > best_edge:
            best_edge = edge
            best_side = side
            best_prob = side_prob
            best_odds = american_odds
            best_metrics = metrics

    if best_side is None or best_edge <= edge_threshold or best_metrics is None:
        return None

    play = game["home_name"] if best_side == "home" else game["away_name"]
    quarter_kelly_pct = best_metrics.fractional_kelly_pct or 0.0
    if quarter_kelly_pct <= 0:
        return None

    confidence_score = best_metrics.confidence_score
    confidence_label = confidence_label_from_score(confidence_score)

    return ValuePick(
        away_name=game["away_name"],
        home_name=game["home_name"],
        play=play,
        quarter_kelly_pct=quarter_kelly_pct,
        edge_pct=best_edge * 100,
        model_win_prob=best_prob,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        confidence_tier=best_metrics.confidence_tier,
        ev_per_unit=best_metrics.ev_per_unit or 0.0,
        american_odds=best_odds,
        temperature=evaluated.temperature,
        wind=evaluated.wind,
        bullpen_status=evaluated.bullpen_status,
        pred_home_runs=evaluated.home_runs,
        pred_away_runs=evaluated.away_runs,
        home_win_prob=evaluated.home_prob,
        away_win_prob=evaluated.away_prob,
    )


def get_upcoming_value_picks(
    game_date: date | None = None,
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
    write_log: bool = False,
) -> list[ValuePick]:
    game_date = game_date or date.today()
    slate = baseball_data.fetch_games_for_date(game_date)
    if not slate:
        return []

    _memoize_obp_lookups()
    models = load_frozen_models()
    seasons = list(DEFAULT_PREDICTION_SEASONS)
    if game_date.year not in seasons:
        seasons = sorted(set(seasons + [game_date.year]))
    history = baseball_data.games_for_prediction(seasons)
    if history.empty:
        raise RuntimeError("No historical games available for predictions.")

    picks: list[ValuePick] = []
    log_rows: list[dict[str, str]] = []

    for game in slate:
        pick = _evaluate_game_pick(
            game,
            models=models,
            games_history=history,
            game_date=game_date,
            window=window,
        )
        if pick is not None:
            picks.append(pick)
            log_rows.append(
                {
                    "Date": game_date.isoformat(),
                    "Matchup": f"{pick.away_name} @ {pick.home_name}",
                    "Temp": pick.temperature,
                    "Wind": pick.wind,
                    "Bullpen_Status": pick.bullpen_status,
                    "Edge_Pct": f"{pick.edge_pct:+.2f}",
                    "Play": pick.play,
                    "Result": "",
                }
            )

    picks.sort(key=lambda row: row.edge_pct, reverse=True)
    if write_log:
        append_slate_evaluation_log(log_rows)
    return picks


def _print_value_picks(picks: list[ValuePick], game_date: date) -> None:
    print(f"--- Mock Slate Value Picks ({game_date.isoformat()}) ---", flush=True)
    if not picks:
        print("No moneyline plays with model edge above threshold.")
        return

    print(
        f"{'Matchup':<40} {'Play':<20} {'Line':>6} {'Edge':>7} {'Temp':>6} "
        f"{'Wind':<18} {'Bullpen':<28}"
    )
    print("-" * 130)
    for pick in picks:
        matchup = f"{pick.away_name} @ {pick.home_name}"
        wind = pick.wind[:17]
        bullpen = pick.bullpen_status[:27]
        print(
            f"{matchup:<40} {pick.play:<20} {pick.american_odds:>+6d} "
            f"{pick.edge_pct:>+6.2f}% {pick.temperature:>6} {wind:<18} {bullpen:<28}"
        )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    game_date = date.today()
    if len(sys.argv) > 1:
        game_date = date.fromisoformat(sys.argv[1])

    write_log = "--log" in sys.argv
    picks = get_upcoming_value_picks(game_date, write_log=write_log)
    _print_value_picks(picks, game_date)


if __name__ == "__main__":
    main()
