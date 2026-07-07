"""One-shot training script for full-game moneyline models."""

from __future__ import annotations

import baseball_data
import model
from config import DEFAULT_PREDICTION_SEASONS, DEFAULT_ROLLING_WINDOW


def main() -> None:
    print("Fetching season games...")
    games = baseball_data.fetch_season_games(DEFAULT_PREDICTION_SEASONS)
    print(f"Games: {len(games)}")
    print("Building training dataset...")
    training = baseball_data.build_training_dataset(games, window=DEFAULT_ROLLING_WINDOW)
    print(f"Training rows: {len(training)}")
    print("Training model...")
    result = model.train_model(training)
    model.save_model(result["model"])
    print(f"Saved models. Win accuracy: {result['win_accuracy'] * 100:.1f}%")


if __name__ == "__main__":
    main()
