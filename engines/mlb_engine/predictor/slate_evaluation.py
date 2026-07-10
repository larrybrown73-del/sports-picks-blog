"""Shared slate game evaluation with weather and bullpen adjustments."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

import baseball_data
import model
from bullpen_fatigue import BullpenStatus, apply_bullpen_penalty, apply_bullpen_to_runs, compute_bullpen_fatigue
from config import DEFAULT_ROLLING_WINDOW, SLATE_EVALUATION_LOG
from game_conditions import GameConditions, apply_run_environment, fetch_game_conditions
from pitcher_matchup import apply_pitcher_matchup_adjustments


@dataclass
class EvaluatedGame:
    away_name: str
    home_name: str
    home_runs: float
    away_runs: float
    home_prob: float
    away_prob: float
    conditions: GameConditions
    fatigue: BullpenStatus
    travel_tags: list[str] = field(default_factory=list)
    pitcher_matchup_tags: list[str] = field(default_factory=list)
    bullpen_tags: list[str] = field(default_factory=list)
    umpire_modifier: float = 1.0

    @property
    def temperature(self) -> str:
        return self.conditions.display_temp

    @property
    def wind(self) -> str:
        return self.conditions.display_wind

    @property
    def bullpen_status(self) -> str:
        return self.fatigue.display


def evaluate_game(
    game: dict[str, Any],
    game_date: date,
    history: pd.DataFrame,
    models: dict[str, Any],
    *,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> EvaluatedGame:
    """Build features, predict runs, apply weather/bullpen adjustments, return win probs."""
    conditions = fetch_game_conditions(game["game_id"])
    features = baseball_data.build_prediction_row(
        game["home_id"],
        game["away_id"],
        game_date,
        history,
        window=window,
        venue_id=game.get("venue_id"),
    )
    home_runs, away_runs, _ = model.predict_matchup(models, features)

    matchup = apply_pitcher_matchup_adjustments(
        home_runs,
        away_runs,
        game_id=int(game["game_id"]),
        home_id=int(game["home_id"]),
        away_id=int(game["away_id"]),
        season=game_date.year,
        game_date=game_date,
    )
    home_runs = matchup.home_runs
    away_runs = matchup.away_runs

    as_of = datetime.combine(game_date, datetime.min.time())
    fatigue = compute_bullpen_fatigue(
        game["home_id"],
        game["away_id"],
        as_of,
        game_id=game.get("game_id"),
        season=game_date.year,
    )
    home_runs, away_runs, bullpen_tags = apply_bullpen_to_runs(home_runs, away_runs, fatigue)

    home_runs, away_runs = apply_run_environment(
        home_runs, away_runs, conditions.run_env_multiplier
    )
    home_prob, away_prob = model.implied_win_probabilities(home_runs, away_runs)
    home_prob, away_prob = apply_bullpen_penalty(home_prob, away_prob, fatigue)

    return EvaluatedGame(
        away_name=game["away_name"],
        home_name=game["home_name"],
        home_runs=home_runs,
        away_runs=away_runs,
        home_prob=home_prob,
        away_prob=away_prob,
        conditions=conditions,
        fatigue=fatigue,
        travel_tags=[],
        pitcher_matchup_tags=matchup.tags,
        bullpen_tags=bullpen_tags,
        umpire_modifier=1.0,
    )


def append_slate_evaluation_log(
    rows: list[dict[str, Any]],
    *,
    log_file: str = SLATE_EVALUATION_LOG,
) -> None:
    """Append per-game evaluation rows to slate_evaluation_log.csv."""
    if not rows:
        return

    fieldnames = [
        "Date",
        "Matchup",
        "Temp",
        "Wind",
        "Bullpen_Status",
        "Edge_Pct",
        "Play",
        "Result",
    ]
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
