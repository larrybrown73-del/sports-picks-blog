"""JSON-backed ledger for WNBA win-probability Brier tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Default ledger next to this module (engine) or CWD-relative when overridden.
_DEFAULT_LEDGER = Path(__file__).resolve().parent / "historical_predictions.json"


class WNBABrierTracker:
    """
    Manages historical logging of manual or model-driven win probabilities
    and computes the Brier Score to keep predictive calibration sharp.
    """

    def __init__(self, log_file_path: str | Path | None = None):
        self.log_file_path = Path(log_file_path) if log_file_path else _DEFAULT_LEDGER
        self.predictions = self._load_logs()

    def _load_logs(self) -> list[dict[str, Any]]:
        """Load historical prediction data if the JSON file exists."""
        if self.log_file_path.exists():
            with open(self.log_file_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_logs(self) -> None:
        """Write current prediction memory back out to the JSON file."""
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(self.predictions, f, indent=4)

    def log_game_prediction(
        self,
        game_id: str,
        game_date: str,
        team: str,
        opponent: str,
        predicted_win_prob: float,
    ) -> None:
        """
        Record a pre-game baseline forecast before tip-off.
        ``predicted_win_prob`` must be a float between 0.0 and 1.0.
        """
        if not (0.0 <= predicted_win_prob <= 1.0):
            raise ValueError("Probability must sit strictly between 0.0 and 1.0")

        # Overwrite if game_id already exists, otherwise append
        self.predictions = [p for p in self.predictions if p["game_id"] != game_id]
        self.predictions.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "team": team,
                "opponent": opponent,
                "predicted_win_prob": float(predicted_win_prob),
                "actual_outcome": None,  # Settled post-game (1 Win, 0 Loss)
            }
        )
        self._save_logs()
        print(
            f"Logged pre-game card for {team} vs {opponent} "
            f"({predicted_win_prob:.1%})"
        )

    def settle_game_outcome(self, game_id: str, actual_outcome: int) -> None:
        """
        Update the ledger once the game goes final.
        ``actual_outcome``: 1 if the tracked team won, 0 if they lost.
        """
        if actual_outcome not in (0, 1):
            raise ValueError("Outcome must be exactly 1 (Win) or 0 (Loss)")

        for pred in self.predictions:
            if pred["game_id"] == game_id:
                pred["actual_outcome"] = actual_outcome
                self._save_logs()
                print(f"Settled game {game_id} with outcome: {actual_outcome}")
                return

        print(f"Game ID {game_id} not found in historical records.")

    def calculate_brier_score(self, window: int | None = None) -> float:
        """
        Calculate the Brier Score across settled history.

        Formula: BS = (1/N) * sum((prob - actual)^2)
        """
        settled = [p for p in self.predictions if p["actual_outcome"] is not None]

        if not settled:
            print("No settled games found to calculate Brier score.")
            return 0.25  # Baseline coin-flip uncertainty

        if window:
            settled = settled[-window:]

        errors = [
            (p["predicted_win_prob"] - p["actual_outcome"]) ** 2 for p in settled
        ]
        return float(np.mean(errors))


if __name__ == "__main__":
    print("Initializing Manual Brier Tracker Script...")

    tracker = WNBABrierTracker("test_wnba_ledger.json")

    print("\n--- STEP 1: Logging Tonight's Slate (July 13, 2026) ---")
    tracker.log_game_prediction(
        game_id="20260713_LAS_ATL",
        game_date="2026-07-13",
        team="Los Angeles Sparks",
        opponent="Atlanta Dream",
        predicted_win_prob=0.42,
    )
    tracker.log_game_prediction(
        game_id="20260713_PHX_MIN",
        game_date="2026-07-13",
        team="Minnesota Lynx",
        opponent="Phoenix Mercury",
        predicted_win_prob=0.78,
    )

    print("\n--- STEP 2: Simulating Post-Game Nightly Audit ---")
    tracker.settle_game_outcome(game_id="20260713_LAS_ATL", actual_outcome=0)
    tracker.settle_game_outcome(game_id="20260713_PHX_MIN", actual_outcome=1)

    print("\n--- STEP 3: Calculating Calibration ---")
    current_brier = tracker.calculate_brier_score()
    print(f"Realized Baseline Brier Score: {current_brier:.4f}")

    if current_brier < 0.20:
        print("Elite calibration. Percentages accurately mirror variance.")
    elif current_brier <= 0.25:
        print("Profit zone, but keep monitoring. Beating a blind coin flip.")
    else:
        print("Over-confident bias detected. Model is over-valuing names.")
