"""Train and save full-game and F5 Random Forest run models."""

from __future__ import annotations

import sys

import baseball_data
import model
from config import DEFAULT_PREDICTION_SEASONS, DEFAULT_ROLLING_WINDOW


def _print_metrics(label: str, result: dict) -> None:
    print(
        f"{label} | win accuracy {result['win_accuracy'] * 100:.2f}% | "
        f"home MAE {result['home_mae']:.3f} | "
        f"away MAE {result['away_mae']:.3f} | "
        f"train {result['train_size']} / test {result['test_size']}",
        flush=True,
    )


def main() -> None:
    seasons = DEFAULT_PREDICTION_SEASONS
    window = DEFAULT_ROLLING_WINDOW

    print(f"Fetching games for seasons {seasons}...", flush=True)
    games_df = baseball_data.fetch_season_games(seasons)
    print(f"Loaded {len(games_df)} completed games.", flush=True)
    if games_df.empty:
        raise SystemExit("No games found.")

    obp_maps = {season: baseball_data.get_team_obp_map(season) for season in seasons}
    era_maps = {season: baseball_data.get_team_era_map(season) for season in seasons}

    print("Building full-game training features (rolling runs, OBP, weather)...", flush=True)
    training_df = baseball_data.build_training_dataset(
        games_df, window=window, obp_maps=obp_maps
    )
    print(f"Full-game training rows: {len(training_df)}", flush=True)
    if training_df.empty:
        raise SystemExit("Full-game training dataset empty.")

    print("Training full-game Random Forest run models...", flush=True)
    result = model.train_model(training_df)
    model.save_model(result["model"])
    print(
        "Saved models/home_runs_model.pkl and models/away_runs_model.pkl",
        flush=True,
    )
    _print_metrics("Full game", result)

    print(
        "Building F5 training features (linescores, rolling F5 averages, ERA, weather)...",
        flush=True,
    )
    f5_training_df = baseball_data.build_f5_training_dataset(
        games_df, window=window, obp_maps=obp_maps, era_maps=era_maps
    )
    print(f"F5 training rows: {len(f5_training_df)}", flush=True)
    if f5_training_df.empty:
        raise SystemExit("F5 training dataset empty.")

    print("Training F5 Random Forest run models...", flush=True)
    f5_result = model.train_f5_model(f5_training_df)
    model.save_f5_model(f5_result["model"])
    print(
        "Saved models/home_f5_runs_model.pkl and models/away_f5_runs_model.pkl",
        flush=True,
    )
    _print_metrics("F5", f5_result)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
