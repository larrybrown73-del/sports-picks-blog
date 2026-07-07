import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseball_data import (
    F5_FEATURE_COLUMNS,
    F5_TARGET_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
HOME_RUNS_MODEL_PATH = MODELS_DIR / "home_runs_model.pkl"
AWAY_RUNS_MODEL_PATH = MODELS_DIR / "away_runs_model.pkl"
HOME_F5_MODEL_PATH = MODELS_DIR / "home_f5_runs_model.pkl"
AWAY_F5_MODEL_PATH = MODELS_DIR / "away_f5_runs_model.pkl"

TEST_FRACTION = 0.2
PROB_CLAMP_LOW = 0.05
PROB_CLAMP_HIGH = 0.95


def normalize_win_probabilities(
    home_prob: float,
    away_prob: float,
) -> tuple[float, float]:
    """Clamp and re-normalize home/away win probabilities to sum to 1."""
    home = min(max(home_prob, PROB_CLAMP_LOW), PROB_CLAMP_HIGH)
    away = min(max(away_prob, PROB_CLAMP_LOW), PROB_CLAMP_HIGH)
    total = home + away
    if total <= 0:
        return 0.5, 0.5
    return home / total, away / total


def implied_win_probabilities(home_runs: float, away_runs: float) -> tuple[float, float]:
    """Convert predicted run totals into normalized implied win probabilities."""
    total = home_runs + away_runs
    if total <= 0:
        return 0.5, 0.5
    return normalize_win_probabilities(home_runs / total, away_runs / total)


def winner_win_probability_pct(
    home_runs: float,
    away_runs: float,
    *,
    home_wins: bool,
) -> float:
    """Return the predicted winner's implied win probability as a percentage."""
    home_prob, away_prob = implied_win_probabilities(home_runs, away_runs)
    return (home_prob if home_wins else away_prob) * 100.0


def _temporal_train_test_split(
    training_df: pd.DataFrame,
    test_size: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earliest games train, latest games test."""
    if "game_date" not in training_df.columns:
        raise ValueError("Training data must include game_date for temporal split.")

    sort_keys = ["game_date"]
    if "game_id" in training_df.columns:
        sort_keys.append("game_id")

    ordered = training_df.sort_values(sort_keys).reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("Need at least 2 games for train/test split.")

    split_at = max(1, int(len(ordered) * (1 - test_size)))
    split_at = min(split_at, len(ordered) - 1)

    return ordered.iloc[:split_at], ordered.iloc[split_at:]


def train_model(training_df: pd.DataFrame) -> dict:
    """Train two Random Forest regressors for home and away runs."""
    if training_df.empty:
        raise ValueError("Training dataset is empty. Try different seasons or settings.")

    required_targets = set(TARGET_COLUMNS)
    missing = required_targets - set(training_df.columns)
    if missing:
        raise ValueError(f"Training data missing target columns: {', '.join(sorted(missing))}")

    missing_features = set(FEATURE_COLUMNS) - set(training_df.columns)
    if missing_features:
        raise ValueError(
            f"Training data missing feature columns: {', '.join(sorted(missing_features))}"
        )

    train_df, test_df = _temporal_train_test_split(training_df)
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_home_train = train_df["home_team_runs"]
    y_home_test = test_df["home_team_runs"]
    y_away_train = train_df["away_team_runs"]
    y_away_test = test_df["away_team_runs"]

    home_model = RandomForestRegressor(n_estimators=100, random_state=42)
    away_model = RandomForestRegressor(n_estimators=100, random_state=42)

    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    home_pred = home_model.predict(X_test)
    away_pred = away_model.predict(X_test)

    home_mae = mean_absolute_error(y_home_test, home_pred)
    away_mae = mean_absolute_error(y_away_test, away_pred)
    home_rmse = mean_squared_error(y_home_test, home_pred) ** 0.5
    away_rmse = mean_squared_error(y_away_test, away_pred) ** 0.5

    actual_home_wins = (y_home_test.values > y_away_test.values).astype(int)
    predicted_home_wins = (home_pred > away_pred).astype(int)
    win_accuracy = (actual_home_wins == predicted_home_wins).mean()

    models = {"home_runs": home_model, "away_runs": away_model}

    return {
        "model": models,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "home_rmse": home_rmse,
        "away_rmse": away_rmse,
        "win_accuracy": win_accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_games": len(training_df),
    }


def predict_matchup(
    models: dict[str, RandomForestRegressor],
    features_df: pd.DataFrame,
) -> tuple[float, float, int]:
    """Return (predicted_home_runs, predicted_away_runs, home_win_prediction)."""
    home_runs = float(models["home_runs"].predict(features_df)[0])
    away_runs = float(models["away_runs"].predict(features_df)[0])
    prediction = 1 if home_runs > away_runs else 0
    return home_runs, away_runs, prediction


def save_model(models: dict[str, RandomForestRegressor]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with HOME_RUNS_MODEL_PATH.open("wb") as handle:
        pickle.dump(models["home_runs"], handle)
    with AWAY_RUNS_MODEL_PATH.open("wb") as handle:
        pickle.dump(models["away_runs"], handle)


def _load_pickle_pair(
    first_path: Path,
    second_path: Path,
    first_key: str,
    second_key: str,
) -> dict[str, RandomForestRegressor] | None:
    if not first_path.exists() or not second_path.exists():
        return None

    try:
        with first_path.open("rb") as handle:
            first_model = pickle.load(handle)
        with second_path.open("rb") as handle:
            second_model = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError):
        return None

    return {first_key: first_model, second_key: second_model}


def load_model() -> dict[str, RandomForestRegressor] | None:
    return _load_pickle_pair(
        HOME_RUNS_MODEL_PATH,
        AWAY_RUNS_MODEL_PATH,
        "home_runs",
        "away_runs",
    )


def train_f5_model(training_df: pd.DataFrame) -> dict:
    """Train two dedicated Random Forest regressors for First 5 Inning (F5) runs."""
    if training_df.empty:
        raise ValueError("F5 training dataset is empty. Try different seasons or settings.")

    required_targets = set(F5_TARGET_COLUMNS)
    missing = required_targets - set(training_df.columns)
    if missing:
        raise ValueError(f"F5 training data missing target columns: {', '.join(sorted(missing))}")

    missing_features = set(F5_FEATURE_COLUMNS) - set(training_df.columns)
    if missing_features:
        raise ValueError(
            f"F5 training data missing feature columns: {', '.join(sorted(missing_features))}"
        )

    train_df, test_df = _temporal_train_test_split(training_df)
    X_train = train_df[F5_FEATURE_COLUMNS]
    X_test = test_df[F5_FEATURE_COLUMNS]
    y_home_train = train_df["home_f5_runs"]
    y_home_test = test_df["home_f5_runs"]
    y_away_train = train_df["away_f5_runs"]
    y_away_test = test_df["away_f5_runs"]

    rf_home_f5 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_away_f5 = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_home_f5.fit(X_train, y_home_train)
    rf_away_f5.fit(X_train, y_away_train)

    home_pred = rf_home_f5.predict(X_test)
    away_pred = rf_away_f5.predict(X_test)

    home_mae = mean_absolute_error(y_home_test, home_pred)
    away_mae = mean_absolute_error(y_away_test, away_pred)
    home_rmse = mean_squared_error(y_home_test, home_pred) ** 0.5
    away_rmse = mean_squared_error(y_away_test, away_pred) ** 0.5

    actual_home_wins = (y_home_test.values > y_away_test.values).astype(int)
    predicted_home_wins = (home_pred > away_pred).astype(int)
    win_accuracy = (actual_home_wins == predicted_home_wins).mean()

    models = {"home_f5": rf_home_f5, "away_f5": rf_away_f5}

    return {
        "model": models,
        "home_mae": home_mae,
        "away_mae": away_mae,
        "home_rmse": home_rmse,
        "away_rmse": away_rmse,
        "win_accuracy": win_accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_games": len(training_df),
    }


def predict_f5_matchup(
    models: dict[str, RandomForestRegressor],
    features_df: pd.DataFrame,
) -> tuple[float, float, int]:
    """Return (predicted_home_f5_runs, predicted_away_f5_runs, home_f5_win_prediction)."""
    home_runs = float(models["home_f5"].predict(features_df)[0])
    away_runs = float(models["away_f5"].predict(features_df)[0])
    prediction = 1 if home_runs > away_runs else 0
    return home_runs, away_runs, prediction


def save_f5_model(models: dict[str, RandomForestRegressor]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with HOME_F5_MODEL_PATH.open("wb") as handle:
        pickle.dump(models["home_f5"], handle)
    with AWAY_F5_MODEL_PATH.open("wb") as handle:
        pickle.dump(models["away_f5"], handle)


def load_f5_model() -> dict[str, RandomForestRegressor] | None:
    return _load_pickle_pair(
        HOME_F5_MODEL_PATH,
        AWAY_F5_MODEL_PATH,
        "home_f5",
        "away_f5",
    )
