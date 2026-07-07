from datetime import date
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import baseball_data
import model
from config import DEFAULT_PREDICTION_SEASONS, DEFAULT_ROLLING_WINDOW

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

st.set_page_config(page_title="Baseball Predictor", page_icon="⚾", layout="wide")

FULL_GAME = "Full Game"
FIVE_INNING = "First 5 Innings (F5)"
DATA_CACHE_VERSION = getattr(baseball_data, "DATA_CACHE_VERSION", "basic_shell_v1")

st.title("Baseball Predictor")
st.caption("Random Forest run predictions using real MLB Stats API data")

LIMITATIONS = (
    "Early-season games are skipped when teams have fewer than 5 prior games. "
    "OBP uses full-season team stats as a proxy. Weather (temperature, wind speed) "
    "comes from Open-Meteo daily averages at each home ballpark. "
    "Two regressors predict full-game runs scored; the winner is whichever team is "
    "predicted to score more. The First 5 Innings (F5) market uses a completely "
    "separate pair of models trained on end-of-5th-inning scores, rolling F5 scoring "
    "averages, and season team ERA as a starting-pitcher-quality proxy."
)
st.info(LIMITATIONS)


@st.cache_data(ttl=86400)
def cached_fetch_season_games(seasons: tuple[int, ...]) -> pd.DataFrame:
    return baseball_data.fetch_season_games(list(seasons))


@st.cache_data(ttl=86400)
def cached_get_team_obp_map(season: int) -> dict[int, float]:
    return baseball_data.get_team_obp_map(season)


@st.cache_data(ttl=86400)
def cached_get_team_era_map(season: int) -> dict[int, float]:
    return baseball_data.get_team_era_map(season)


@st.cache_data(ttl=86400)
def cached_build_training_dataset(
    games_df: pd.DataFrame,
    window: int,
    seasons: tuple[int, ...],
    cache_version: str,
) -> pd.DataFrame:
    obp_maps = {season: cached_get_team_obp_map(season) for season in seasons}
    return baseball_data.build_training_dataset(games_df, window=window, obp_maps=obp_maps)


@st.cache_data(ttl=86400)
def cached_build_f5_training_dataset(
    games_df: pd.DataFrame,
    window: int,
    seasons: tuple[int, ...],
    cache_version: str,
) -> pd.DataFrame:
    obp_maps = {season: cached_get_team_obp_map(season) for season in seasons}
    era_maps = {season: cached_get_team_era_map(season) for season in seasons}
    return baseball_data.build_f5_training_dataset(
        games_df, window=window, obp_maps=obp_maps, era_maps=era_maps
    )


@st.cache_data(ttl=3600)
def cached_games_for_prediction(seasons: tuple[int, ...]) -> pd.DataFrame:
    return baseball_data.games_for_prediction(list(seasons))


@st.cache_data(ttl=3600)
def cached_f5_prediction_games(seasons: tuple[int, ...]) -> pd.DataFrame:
    games_df = baseball_data.games_for_prediction(list(seasons))
    if games_df.empty:
        return games_df
    return baseball_data.attach_f5_runs(games_df)


@st.cache_data(ttl=3600)
def cached_todays_games() -> list[dict]:
    return baseball_data.fetch_todays_games()


@st.cache_data(ttl=86400)
def cached_mlb_teams() -> list[dict]:
    return baseball_data.get_mlb_teams()


def display_prediction_summary(
    home_name: str,
    away_name: str,
    home_runs: float,
    away_runs: float,
    prediction: int,
    features: pd.DataFrame | None = None,
    market: str = FULL_GAME,
    game_pk: int | None = None,
) -> None:
    """Render market-specific prediction cards for full game or F5."""
    spread = home_runs - away_runs
    total = home_runs + away_runs
    home_wins = prediction == 1
    winner = home_name if home_wins else away_name
    win_prob = model.winner_win_probability_pct(home_runs, away_runs, home_wins=home_wins)
    feature_row = features.iloc[0] if features is not None and not features.empty else None

    if market == FULL_GAME:
        st.subheader("Full Game Predicted Runs")
        score_col1, score_col2 = st.columns(2)
        score_col1.metric(f"{home_name} (Home)", f"{home_runs:.1f} runs")
        score_col2.metric(f"{away_name} (Away)", f"{away_runs:.1f} runs")

        card1, card2, card3 = st.columns(3)
        with card1:
            st.metric(
                label="Predicted Winner & Win Probability",
                value=winner,
                delta=f"{win_prob:.1f}% implied win probability",
                delta_color="off",
            )
        with card2:
            spread_label = (
                "Home favored" if spread > 0 else "Away favored" if spread < 0 else "Pick 'em"
            )
            st.metric(
                label="Run Differential / Spread",
                value=f"{spread:+.1f}",
                delta=f"{spread_label} ({home_name} − {away_name})",
                delta_color="off",
            )
        with card3:
            st.metric(
                label="Predicted Over/Under Total",
                value=f"{total:.1f} runs",
                delta="Combined full-game score",
                delta_color="off",
            )

    elif market == FIVE_INNING:
        st.subheader("F5 Predicted Runs")
        score_col1, score_col2 = st.columns(2)
        score_col1.metric(f"{home_name} (Home)", f"{home_runs:.1f} F5 runs")
        score_col2.metric(f"{away_name} (Away)", f"{away_runs:.1f} F5 runs")

        card1, card2, card3 = st.columns(3)
        with card1:
            st.metric(
                label="F5 Predicted Winner & Win Probability",
                value=winner,
                delta=f"{win_prob:.1f}% implied F5 win probability",
                delta_color="off",
            )
        with card2:
            spread_label = (
                "Home favored" if spread > 0 else "Away favored" if spread < 0 else "Pick 'em"
            )
            st.metric(
                label="F5 Run Differential / Spread",
                value=f"{spread:+.1f}",
                delta=f"{spread_label} through 5 innings",
                delta_color="off",
            )
        with card3:
            st.metric(
                label="F5 Over/Under Total",
                value=f"{total:.1f} F5 runs",
                delta="Combined first-5-inning score",
                delta_color="off",
            )

        if game_pk is not None:
            pitchers = baseball_data.get_starting_pitchers(game_pk)
            st.markdown(
                f"**Starting Pitcher Matchup:** {pitchers['away_pitcher']} (away) vs "
                f"{pitchers['home_pitcher']} (home)"
            )

        if feature_row is not None and {"home_team_era", "away_team_era"}.issubset(features.columns):
            st.markdown("**Starting Pitcher Quality (Season Team ERA Proxy)**")
            era_col1, era_col2 = st.columns(2)
            era_col1.metric(f"{home_name} team ERA", f"{feature_row['home_team_era']:.2f}")
            era_col2.metric(f"{away_name} team ERA", f"{feature_row['away_team_era']:.2f}")

        if feature_row is not None and {
            "home_f5_avg_runs_scored",
            "home_f5_avg_runs_allowed",
            "away_f5_avg_runs_scored",
            "away_f5_avg_runs_allowed",
        }.issubset(features.columns):
            st.markdown("**F5 Rolling Lines**")
            f5_col1, f5_col2, f5_col3, f5_col4 = st.columns(4)
            f5_col1.metric(
                f"{home_name} F5 avg scored",
                f"{feature_row['home_f5_avg_runs_scored']:.2f}",
            )
            f5_col2.metric(
                f"{home_name} F5 avg allowed",
                f"{feature_row['home_f5_avg_runs_allowed']:.2f}",
            )
            f5_col3.metric(
                f"{away_name} F5 avg scored",
                f"{feature_row['away_f5_avg_runs_scored']:.2f}",
            )
            f5_col4.metric(
                f"{away_name} F5 avg allowed",
                f"{feature_row['away_f5_avg_runs_allowed']:.2f}",
            )

    if feature_row is not None and {"temperature", "wind_speed"}.issubset(features.columns):
        weather_col1, weather_col2 = st.columns(2)
        weather_col1.metric("Temperature (°F)", f"{feature_row['temperature']:.1f}")
        weather_col2.metric("Wind speed (mph)", f"{feature_row['wind_speed']:.1f}")


if "trained_model" not in st.session_state:
    st.session_state.trained_model = model.load_model()
    st.session_state.training_result = None
if "trained_f5_model" not in st.session_state:
    st.session_state.trained_f5_model = model.load_f5_model()
    st.session_state.training_f5_result = None


with st.sidebar:
    st.header("Settings")
    default_seasons = DEFAULT_PREDICTION_SEASONS
    seasons = st.multiselect(
        "Training seasons",
        options=list(range(2018, date.today().year + 1)),
        default=default_seasons,
    )
    window = st.slider("Rolling window (games)", min_value=5, max_value=20, value=DEFAULT_ROLLING_WINDOW)
    train_clicked = st.button("Train model", type="primary", width="stretch")
    force_retrain_clicked = st.button(
        "Force Clear Cache & Retrain",
        width="stretch",
        help="Clears Streamlit cached dataframes and rebuilds the training data from scratch.",
    )
    st.caption(
        "Training builds both the Full Game and First 5 Innings (F5) models. "
        "F5 training fetches end-of-5th-inning linescores and may take longer on first run."
    )

market = st.radio(
    "Market",
    options=[FULL_GAME, FIVE_INNING],
    horizontal=True,
    help="Switch between full-game predictions and the dedicated First 5 Innings (F5) model.",
)
is_f5 = market == FIVE_INNING

if force_retrain_clicked:
    st.cache_data.clear()
    for key in (
        "prediction_games",
        "f5_prediction_games",
        "training_result",
        "training_f5_result",
    ):
        st.session_state.pop(key, None)
    st.info("Cleared cached training data. Rebuilding everything now...")

if train_clicked or force_retrain_clicked:
    if len(seasons) < 1:
        st.error("Select at least one training season.")
    else:
        seasons_tuple = tuple(sorted(seasons))
        with st.spinner("Fetching MLB data and training Full Game model..."):
            try:
                games_df = cached_fetch_season_games(seasons_tuple)
                if games_df.empty:
                    st.error("No completed games found for the selected seasons.")
                else:
                    training_df = cached_build_training_dataset(
                        games_df,
                        window,
                        seasons_tuple,
                        DATA_CACHE_VERSION,
                    )
                    if training_df.empty:
                        st.error(
                            "Could not build training features. Try different seasons or a smaller rolling window."
                        )
                    else:
                        result = model.train_model(training_df)
                        st.session_state.trained_model = result["model"]
                        st.session_state.training_result = result
                        st.session_state.prediction_games = cached_games_for_prediction(
                            seasons_tuple
                        )
                        model.save_model(result["model"])
                        st.success("Full Game model trained successfully.")
            except Exception as exc:
                st.error(str(exc))

        with st.spinner(
            "Fetching first-5-inning linescores and training F5 model (this can take a while)..."
        ):
            try:
                games_df = cached_fetch_season_games(seasons_tuple)
                if not games_df.empty:
                    f5_training_df = cached_build_f5_training_dataset(
                        games_df,
                        window,
                        seasons_tuple,
                        DATA_CACHE_VERSION,
                    )
                    if f5_training_df.empty:
                        st.warning(
                            "Could not build F5 training features. Try different seasons or a smaller rolling window."
                        )
                    else:
                        f5_result = model.train_f5_model(f5_training_df)
                        st.session_state.trained_f5_model = f5_result["model"]
                        st.session_state.training_f5_result = f5_result
                        st.session_state.f5_prediction_games = cached_f5_prediction_games(
                            seasons_tuple
                        )
                        model.save_f5_model(f5_result["model"])
                        st.success("First 5 Innings (F5) model trained successfully.")
            except Exception as exc:
                st.error(f"F5 model: {exc}")

if is_f5:
    trained_model = st.session_state.get("trained_f5_model")
    result = st.session_state.get("training_f5_result")
    predict_fn = model.predict_f5_matchup
    build_row_fn = baseball_data.build_f5_prediction_row
    history_key = "f5_prediction_games"
    history_loader = cached_f5_prediction_games
else:
    trained_model = st.session_state.get("trained_model")
    result = st.session_state.get("training_result")
    predict_fn = model.predict_matchup
    build_row_fn = baseball_data.build_prediction_row
    history_key = "prediction_games"
    history_loader = cached_games_for_prediction


def get_prediction_history(seasons_tuple: tuple[int, ...]) -> pd.DataFrame | None:
    """Load (and cache in session) the game history for the active market."""
    if history_key not in st.session_state:
        st.session_state[history_key] = history_loader(seasons_tuple)
    return st.session_state.get(history_key)


st.header(f"Model status — {market}")

if result:
    st.metric("Derived win accuracy", f"{result['win_accuracy'] * 100:.2f}%")
    col1, col2, col3, col4 = st.columns(4)
    home_label = "Home F5 runs MAE" if is_f5 else "Home runs MAE"
    away_label = "Away F5 runs MAE" if is_f5 else "Away runs MAE"
    col1.metric(home_label, f"{result['home_mae']:.2f}")
    col2.metric(away_label, f"{result['away_mae']:.2f}")
    col3.metric("Training games", result["train_size"])
    col4.metric("Test games", result["test_size"])
    st.caption(f"Home RMSE: {result['home_rmse']:.2f} | Away RMSE: {result['away_rmse']:.2f}")
    st.caption(f"Total games used: {result['total_games']}")
elif trained_model:
    st.warning(f"Loaded a saved {market} model from disk. Retrain to refresh metrics.")
else:
    st.warning(f"No {market} model trained yet. Use the sidebar to train on MLB data.")

st.divider()
st.header(f"Predict a matchup — {market}")

teams = cached_mlb_teams()
team_names = [team["name"] for team in teams]
name_to_id = {team["name"]: team["id"] for team in teams}

col_home, col_away = st.columns(2)
with col_home:
    home_team = st.selectbox("Home team", team_names, index=0 if team_names else None)
with col_away:
    away_default = 1 if len(team_names) > 1 else 0
    away_team = st.selectbox("Away team", team_names, index=away_default if team_names else None)

predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    if not trained_model:
        st.error(f"Train the {market} model first.")
    elif home_team == away_team:
        st.error("Home and away teams must be different.")
    else:
        seasons_tuple = tuple(sorted(seasons)) if seasons else tuple(DEFAULT_PREDICTION_SEASONS)
        spinner_msg = (
            "Loading recent F5 game history..." if is_f5 else "Loading recent game history..."
        )
        if history_key not in st.session_state:
            with st.spinner(spinner_msg):
                try:
                    st.session_state[history_key] = history_loader(seasons_tuple)
                except Exception as exc:
                    st.error(str(exc))
                    st.session_state[history_key] = None

        games_history = st.session_state.get(history_key)
        if games_history is not None and not games_history.empty:
            try:
                features = build_row_fn(
                    name_to_id[home_team],
                    name_to_id[away_team],
                    date.today(),
                    games_history,
                    window=window,
                )
                home_runs, away_runs, prediction = predict_fn(trained_model, features)

                display_prediction_summary(
                    home_team,
                    away_team,
                    home_runs,
                    away_runs,
                    prediction,
                    features=features,
                    market=market,
                )
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.header(f"Today's games — {market}")

if st.button("Load today's games"):
    st.session_state.show_today = True

if st.session_state.get("show_today"):
    try:
        todays_games = cached_todays_games()
        if not todays_games:
            st.write("No MLB games scheduled for today.")
        elif not trained_model:
            st.warning(f"Train the {market} model first to see predictions.")
            for game in todays_games:
                st.write(game["summary"])
        else:
            seasons_tuple = tuple(sorted(seasons)) if seasons else tuple(DEFAULT_PREDICTION_SEASONS)
            games_history = get_prediction_history(seasons_tuple)

            for game in todays_games:
                with st.container(border=True):
                    st.write(f"**{game['away_name']} @ {game['home_name']}**")
                    st.caption(game["summary"])
                    if games_history is not None and not games_history.empty:
                        try:
                            row_kwargs = {
                                "window": window,
                                "venue_id": game.get("venue_id"),
                            }

                            features = build_row_fn(
                                game["home_id"],
                                game["away_id"],
                                date.today(),
                                games_history,
                                **row_kwargs,
                            )
                            home_runs, away_runs, prediction = predict_fn(
                                trained_model, features
                            )
                            display_prediction_summary(
                                game["home_name"],
                                game["away_name"],
                                home_runs,
                                away_runs,
                                prediction,
                                features=features,
                                market=market,
                                game_pk=game.get("game_id"),
                            )
                        except Exception as exc:
                            logger.warning(
                                "Could not build features for game ID %s: %s",
                                game.get("game_id"),
                                exc,
                            )
                            st.warning(
                                f"Could not build features for this matchup "
                                f"(game ID {game.get('game_id')}). "
                                "Check API connectivity or rolling game history."
                            )
    except Exception as exc:
        st.error(str(exc))
