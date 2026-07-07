"""Backtest harness, Kelly sizing, and mock moneyline helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import baseball_data
import model
from config import (
    DEFAULT_PREDICTION_SEASONS,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_SAMPLE_SIZE,
    EDGE_THRESHOLD,
    MOCK_MONEYLINE_VIG,
    QUARTER_KELLY,
)
from market.calculations import (
    american_to_implied,
    compute_wager_metrics,
    full_kelly_fraction,
)

_OBP_CACHE: dict[int, dict[int, float]] = {}


@dataclass
class BacktestResult:
    moneyline_winner_accuracy: float
    betting_roi_pct: float
    net_profit_loss: float
    brier_score_calibrated: float
    games_scored: int
    bets_placed: int


def _memoize_obp_lookups() -> None:
    global _OBP_CACHE
    if _OBP_CACHE:
        return
    for season in DEFAULT_PREDICTION_SEASONS:
        try:
            _OBP_CACHE[season] = baseball_data.get_team_obp_map(season)
        except Exception:
            _OBP_CACHE[season] = {}


def load_frozen_models() -> dict[str, Any]:
    models = model.load_model()
    if models is None:
        raise RuntimeError("No trained models found. Run retrain_models.py first.")
    return models


def american_implied_probability(american_odds: int) -> float:
    return american_to_implied(american_odds)


def probability_to_american(probability: float, vig: float = MOCK_MONEYLINE_VIG) -> int:
    probability = min(max(probability, 0.05), 0.95)
    fair = probability * vig
    if fair >= 0.5:
        return int(round(-100.0 * fair / (1.0 - fair)))
    return int(round(100.0 * (1.0 - fair) / fair))


def american_odds_payout(american_odds: int, stake: float) -> float:
    if american_odds > 0:
        return stake * american_odds / 100.0
    return stake * 100.0 / abs(american_odds)


def compute_market_edge(
    model_prob: float,
    american_odds: int,
) -> tuple[float, float, float]:
    metrics = compute_wager_metrics(model_prob, american_odds)
    market_prob = metrics.implied_prob if metrics.implied_prob is not None else american_to_implied(
        american_odds
    )
    edge = (metrics.edge_pct or 0.0) / 100.0
    ev = metrics.ev_per_unit or 0.0
    return market_prob, edge, ev


def _select_best_side_bet(
    home_prob: float,
    home_odds: int,
    home_won: bool,
    away_prob: float,
    away_odds: int,
    away_won: bool,
    *,
    edge_threshold: float = EDGE_THRESHOLD,
) -> tuple[int, bool] | None:
    """Pick one side per game by highest edge, matching live/mock slate behavior."""
    best_edge = 0.0
    best_prob = 0.0
    best_odds = 0
    best_won = False

    for prob, odds, won in (
        (home_prob, home_odds, home_won),
        (away_prob, away_odds, away_won),
    ):
        _, edge, _ = compute_market_edge(prob, odds)
        if edge > best_edge:
            best_edge = edge
            best_prob = prob
            best_odds = odds
            best_won = won

    if best_edge <= edge_threshold:
        return None
    best_metrics = compute_wager_metrics(best_prob, best_odds)
    if (best_metrics.fractional_kelly_pct or 0.0) <= 0:
        return None
    return best_odds, best_won


def _rolling_home_win_rate(
    games_history: pd.DataFrame,
    team_id: int,
    as_of_ts: pd.Timestamp,
    window: int,
) -> float:
    team_games = games_history[
        ((games_history["home_id"] == team_id) | (games_history["away_id"] == team_id))
        & (games_history["game_date"] < as_of_ts)
    ].tail(window)
    if team_games.empty:
        return 0.5

    wins = 0
    for row in team_games.itertuples(index=False):
        if row.home_id == team_id:
            wins += int(row.home_score > row.away_score)
        else:
            wins += int(row.away_score > row.home_score)
    return wins / len(team_games)


def _mock_moneylines(
    games_history: pd.DataFrame,
    home_id: int,
    away_id: int,
    as_of_ts: pd.Timestamp,
    window: int,
) -> tuple[int, int, str]:
    """Derive mock American moneylines from recent team win rates."""
    home_rate = _rolling_home_win_rate(games_history, home_id, as_of_ts, window)
    away_rate = _rolling_home_win_rate(games_history, away_id, as_of_ts, window)
    total = home_rate + away_rate
    if total <= 0:
        home_prob, away_prob = 0.55, 0.45
    else:
        home_prob = home_rate / total
        away_prob = away_rate / total
    return (
        probability_to_american(home_prob),
        probability_to_american(away_prob),
        "mock",
    )


def run_backtest(sample_size: int = DEFAULT_SAMPLE_SIZE) -> BacktestResult:
    """Run a rolling backtest on recent completed games."""
    from slate_evaluation import evaluate_game

    _memoize_obp_lookups()
    models = load_frozen_models()
    seasons = list(DEFAULT_PREDICTION_SEASONS)
    history = baseball_data.games_for_prediction(seasons)
    if history.empty:
        raise RuntimeError("No historical games available for backtest.")

    completed = history.sort_values("game_date").tail(sample_size)
    correct = 0
    scored = 0
    bets = 0
    net_pl = 0.0
    brier_total = 0.0

    for row in completed.itertuples(index=False):
        game = {
            "game_id": row.game_id,
            "home_id": row.home_id,
            "away_id": row.away_id,
            "home_name": row.home_name,
            "away_name": row.away_name,
            "venue_id": getattr(row, "venue_id", None),
        }
        game_day = pd.Timestamp(row.game_date).date()
        try:
            evaluated = evaluate_game(
                game,
                game_day,
                history,
                models,
                window=DEFAULT_ROLLING_WINDOW,
            )
        except Exception:
            continue

        scored += 1
        predicted_home = evaluated.home_prob >= evaluated.away_prob
        actual_home = row.home_score > row.away_score
        if predicted_home == actual_home:
            correct += 1

        brier_total += (evaluated.home_prob - float(actual_home)) ** 2

        as_of_ts = pd.Timestamp(game_day)
        home_ml, away_ml, _ = _mock_moneylines(
            history, row.home_id, row.away_id, as_of_ts, DEFAULT_ROLLING_WINDOW
        )
        selected = _select_best_side_bet(
            evaluated.home_prob,
            home_ml,
            actual_home,
            evaluated.away_prob,
            away_ml,
            not actual_home,
        )
        if selected is None:
            continue
        best_odds, best_won = selected
        bets += 1
        stake = 100.0
        net_pl += american_odds_payout(best_odds, stake) if best_won else -stake

    accuracy = correct / scored if scored else 0.0
    roi = (net_pl / (bets * 100.0) * 100.0) if bets else 0.0
    brier = brier_total / scored if scored else 0.0
    return BacktestResult(
        moneyline_winner_accuracy=accuracy,
        betting_roi_pct=roi,
        net_profit_loss=net_pl,
        brier_score_calibrated=brier,
        games_scored=scored,
        bets_placed=bets,
    )


# Backward-compatible re-exports
__all__ = [
    "BacktestResult",
    "DEFAULT_PREDICTION_SEASONS",
    "DEFAULT_SAMPLE_SIZE",
    "EDGE_THRESHOLD",
    "QUARTER_KELLY",
    "american_odds_payout",
    "compute_market_edge",
    "full_kelly_fraction",
    "load_frozen_models",
    "run_backtest",
    "_memoize_obp_lookups",
    "_mock_moneylines",
]
