from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import statsapi
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

from weather import attach_weather_features, get_game_weather

MIN_PRIOR_GAMES = 5

DATA_CACHE_VERSION = "basic_shell_v1"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
F5_CACHE_FILE = CACHE_DIR / "f5_linescore_cache.json"

FEATURE_COLUMNS = [
    "home_avg_runs_scored",
    "home_avg_runs_allowed",
    "home_team_obp",
    "away_avg_runs_scored",
    "away_avg_runs_allowed",
    "away_team_obp",
    "temperature",
    "wind_speed",
]

TARGET_COLUMNS = [
    "home_team_runs",
    "away_team_runs",
]

# Dedicated First 5 Innings (F5) feature set: early-game scoring trends
# (rolling F5 run averages), team offensive quality (OBP), pitching quality
# proxy (season team ERA, a stand-in for starting pitcher strength), and weather.
F5_FEATURE_COLUMNS = [
    "home_f5_avg_runs_scored",
    "home_f5_avg_runs_allowed",
    "home_team_obp",
    "home_team_era",
    "away_f5_avg_runs_scored",
    "away_f5_avg_runs_allowed",
    "away_team_obp",
    "away_team_era",
    "temperature",
    "wind_speed",
]

F5_TARGET_COLUMNS = [
    "home_f5_runs",
    "away_f5_runs",
]

SEASON_DATE_RANGES = {
    2023: ("03/30/2023", "10/01/2023"),
    2024: ("03/28/2024", "10/01/2024"),
    2025: ("03/27/2025", "10/01/2025"),
    2026: ("03/26/2026", "10/01/2026"),
}


def season_opener_date(season: int) -> date:
    """Return the regular-season opener for a given year."""
    start_date, _ = _season_range(season)
    return datetime.strptime(start_date, "%m/%d/%Y").date()


def get_mlb_teams() -> list[dict]:
    """Return active MLB teams with id and name."""
    teams = statsapi.get("teams", {"sportId": 1, "season": date.today().year})
    return sorted(
        [
            {"id": team["id"], "name": team["name"]}
            for team in teams.get("teams", [])
            if team.get("sport", {}).get("id") == 1
        ],
        key=lambda team: team["name"],
    )


def _season_range(season: int) -> tuple[str, str]:
    if season in SEASON_DATE_RANGES:
        return SEASON_DATE_RANGES[season]
    return (f"03/01/{season}", f"10/15/{season}")


def get_starting_pitchers(game_pk: int) -> dict[str, str]:
    """Fetch starting pitcher names for both teams."""
    info = get_starting_pitcher_info(game_pk)
    return {
        "home_pitcher": info["home_pitcher_name"],
        "away_pitcher": info["away_pitcher_name"],
    }


def get_starting_pitcher_info(game_pk: int) -> dict[str, str | int | None]:
    """Fetch starting pitcher names and MLBAM ids for both teams."""
    info: dict[str, str | int | None] = {
        "home_pitcher_id": None,
        "away_pitcher_id": None,
        "home_pitcher_name": "Unknown",
        "away_pitcher_name": "Unknown",
    }

    try:
        boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        response = requests.get(boxscore_url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            teams_data = data.get("teams", {})

            for side in ["home", "away"]:
                team_info = teams_data.get(side, {})
                pitcher_ids = team_info.get("pitchers", [])

                if pitcher_ids:
                    first_pitcher_id = int(pitcher_ids[0])
                    player_name = (
                        team_info.get("players", {})
                        .get(f"ID{first_pitcher_id}", {})
                        .get("person", {})
                        .get("fullName")
                    )
                    if player_name:
                        info[f"{side}_pitcher_id"] = first_pitcher_id
                        info[f"{side}_pitcher_name"] = player_name

        if info["home_pitcher_name"] == "Unknown" and info["away_pitcher_name"] == "Unknown":
            schedule_url = (
                "https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&gamePk={game_pk}&hydrate=probablePitcher"
            )
            sched_response = requests.get(schedule_url, timeout=5)

            if sched_response.status_code == 200:
                sched_data = sched_response.json()
                dates = sched_data.get("dates", [])
                if dates:
                    games = dates[0].get("games", [])
                    if games:
                        game_info = games[0]
                        teams = game_info.get("teams", {})
                        for side in ("home", "away"):
                            probable = teams.get(side, {}).get("probablePitcher") or {}
                            name = probable.get("fullName", "Unknown")
                            pitcher_id = probable.get("id")
                            info[f"{side}_pitcher_name"] = name
                            if pitcher_id is not None:
                                info[f"{side}_pitcher_id"] = int(pitcher_id)

    except requests.RequestException:
        return info

    return info


def _is_retriable_schedule_error(exc: BaseException) -> bool:
    """Return True for MLB schedule errors worth retrying."""
    if isinstance(exc, requests.exceptions.Timeout):
        return True
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in {429, 503, 504}
    message = str(exc).lower()
    return "503" in message or "504" in message or "timeout" in message or "timed out" in message


def _fetch_schedule_range(start_date: str, end_date: str) -> list[dict]:
    """Fetch MLB schedule entries for a date range with basic retry handling."""
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            games = statsapi.schedule(
                start_date=start_date,
                end_date=end_date,
                sportId=1,
            )
            return games or []
        except Exception as exc:
            last_exc = exc
            if attempt < 3 and _is_retriable_schedule_error(exc):
                logger.warning(
                    "Schedule fetch failed for %s-%s (attempt %s/3): %s",
                    start_date,
                    end_date,
                    attempt,
                    exc,
                )
                time.sleep(2 * attempt)
                continue
            raise RuntimeError(
                f"Failed to fetch schedule for {start_date} to {end_date}. "
                "Check your internet connection."
            ) from exc
    raise RuntimeError(
        f"Failed to fetch schedule for {start_date} to {end_date}."
    ) from last_exc


def fetch_season_games(seasons: list[int]) -> pd.DataFrame:
    """Fetch completed regular-season MLB games for the given seasons."""
    rows: list[dict] = []

    for season in seasons:
        start_date, end_date = _season_range(season)
        if season >= date.today().year:
            end_date = min(
                datetime.strptime(end_date, "%m/%d/%Y").date(),
                date.today(),
            ).strftime("%m/%d/%Y")

        start_dt = datetime.strptime(start_date, "%m/%d/%Y").date()
        end_dt = datetime.strptime(end_date, "%m/%d/%Y").date()
        if season >= date.today().year:
            games: list[dict] = []
            chunk_start = start_dt
            while chunk_start <= end_dt:
                if chunk_start.month == 12:
                    next_month = date(chunk_start.year + 1, 1, 1)
                else:
                    next_month = date(chunk_start.year, chunk_start.month + 1, 1)
                chunk_end = min(end_dt, next_month - timedelta(days=1))
                games.extend(
                    _fetch_schedule_range(
                        chunk_start.strftime("%m/%d/%Y"),
                        chunk_end.strftime("%m/%d/%Y"),
                    )
                )
                chunk_start = next_month
        else:
            games = _fetch_schedule_range(start_date, end_date)

        if not games:
            continue

        for game in games:
            if game.get("game_type") != "R":
                continue
            if game.get("status") != "Final":
                continue

            home_score = game.get("home_score")
            away_score = game.get("away_score")
            if home_score is None or away_score is None:
                continue

            rows.append(
                {
                    "game_id": game["game_id"],
                    "game_date": pd.to_datetime(game["game_date"]),
                    "season": season,
                    "home_id": game["home_id"],
                    "away_id": game["away_id"],
                    "home_name": game["home_name"],
                    "away_name": game["away_name"],
                    "venue_id": game.get("venue_id"),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "home_win": int(home_score > away_score),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def _load_f5_cache() -> dict[str, list[int] | None]:
    if not F5_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(F5_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_f5_cache(cache: dict[str, list[int] | None]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    F5_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def fetch_f5_runs(
    game_id: int,
    cache: dict[str, list[int] | None] | None = None,
) -> tuple[int, int] | None:
    """Return (home_f5_runs, away_f5_runs): runs through the end of the 5th inning.

    Uses the MLB Stats API linescore endpoint and sums the per-inning runs for
    innings 1-5. Returns None for games that did not reach 5 full innings or
    when linescore data is unavailable.
    """
    key = str(game_id)
    if cache is not None and key in cache:
        value = cache[key]
        return (value[0], value[1]) if value else None

    try:
        linescore = statsapi.get("game_linescore", {"gamePk": game_id})
    except Exception:
        return None

    innings = linescore.get("innings", [])
    home_f5 = 0
    away_f5 = 0
    counted = 0
    for inning in innings:
        num = inning.get("num")
        if num is None or num > 5:
            continue
        home_runs = inning.get("home", {}).get("runs")
        away_runs = inning.get("away", {}).get("runs")
        home_f5 += int(home_runs) if home_runs is not None else 0
        away_f5 += int(away_runs) if away_runs is not None else 0
        counted += 1

    if counted < 5:
        # Rain-shortened or in-progress game; no reliable F5 score.
        if cache is not None:
            cache[key] = None
        return None

    if cache is not None:
        cache[key] = [home_f5, away_f5]
    return home_f5, away_f5


def attach_f5_runs(games_df: pd.DataFrame) -> pd.DataFrame:
    """Add home_f5_runs / away_f5_runs columns by fetching linescores (cached)."""
    if games_df.empty:
        return games_df
    if {"home_f5_runs", "away_f5_runs"}.issubset(games_df.columns):
        return games_df

    cache = _load_f5_cache()
    initial_size = len(cache)

    home_f5_runs: list[int | None] = []
    away_f5_runs: list[int | None] = []
    for game in games_df.itertuples(index=False):
        result = fetch_f5_runs(game.game_id, cache=cache)
        if result is None:
            home_f5_runs.append(None)
            away_f5_runs.append(None)
        else:
            home_f5_runs.append(result[0])
            away_f5_runs.append(result[1])

    if len(cache) != initial_size:
        _save_f5_cache(cache)

    enriched = games_df.copy()
    enriched["home_f5_runs"] = home_f5_runs
    enriched["away_f5_runs"] = away_f5_runs
    return enriched


def build_team_game_logs(games_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Build per-team chronological game logs with runs scored and allowed.

    When F5 columns are present, also records per-game runs scored/allowed
    through the end of the 5th inning so F5 rolling averages can be computed.
    """
    has_f5 = {"home_f5_runs", "away_f5_runs"}.issubset(games_df.columns)
    logs: dict[int, list[dict]] = {}

    for row in games_df.itertuples(index=False):
        home_f5 = getattr(row, "home_f5_runs", None) if has_f5 else None
        away_f5 = getattr(row, "away_f5_runs", None) if has_f5 else None
        home_entry = {
            "game_date": row.game_date,
            "season": row.season,
            "game_id": row.game_id,
            "runs_scored": row.home_score,
            "runs_allowed": row.away_score,
            "f5_runs_scored": home_f5,
            "f5_runs_allowed": away_f5,
            "is_home": True,
            "opponent_id": row.away_id,
        }
        away_entry = {
            "game_date": row.game_date,
            "season": row.season,
            "game_id": row.game_id,
            "runs_scored": row.away_score,
            "runs_allowed": row.home_score,
            "f5_runs_scored": away_f5,
            "f5_runs_allowed": home_f5,
            "is_home": False,
            "opponent_id": row.home_id,
        }
        logs.setdefault(row.home_id, []).append(home_entry)
        logs.setdefault(row.away_id, []).append(away_entry)

    return {
        team_id: pd.DataFrame(entries).sort_values(["game_date", "game_id"]).reset_index(drop=True)
        for team_id, entries in logs.items()
    }


def _rolling_stats(log: pd.DataFrame, game_date: pd.Timestamp, window: int) -> tuple[float, float, int] | None:
    """Compute rolling average runs scored/allowed using games strictly before game_date."""
    prior = log[log["game_date"] < game_date].tail(window)
    if len(prior) < MIN_PRIOR_GAMES:
        return None
    return (
        float(prior["runs_scored"].mean()),
        float(prior["runs_allowed"].mean()),
        len(prior),
    )


def _rolling_f5_stats(
    log: pd.DataFrame, game_date: pd.Timestamp, window: int
) -> tuple[float, float] | None:
    """Compute rolling average F5 runs scored/allowed using games strictly before game_date."""
    if "f5_runs_scored" not in log.columns:
        return None
    prior = log[log["game_date"] < game_date].tail(window)
    prior = prior.dropna(subset=["f5_runs_scored", "f5_runs_allowed"])
    if len(prior) < MIN_PRIOR_GAMES:
        return None
    return (
        float(prior["f5_runs_scored"].mean()),
        float(prior["f5_runs_allowed"].mean()),
    )


def get_team_obp_map(season: int) -> dict[int, float]:
    """Fetch season-level team OBP for all MLB teams."""
    obp_map: dict[int, float] = {}

    try:
        teams = statsapi.get("teams", {"sportId": 1, "season": season})
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch teams for OBP lookup ({season}).") from exc

    for team in teams.get("teams", []):
        team_id = team["id"]
        try:
            stats = statsapi.get(
                "team_stats",
                {
                    "teamId": team_id,
                    "season": season,
                    "group": "hitting",
                    "stats": "season",
                    "sportIds": 1,
                },
            )
        except Exception:
            continue

        splits = stats.get("stats", [])
        if not splits:
            continue

        stat_block = splits[0].get("splits", [])
        if not stat_block:
            continue

        obp_value = stat_block[0].get("stat", {}).get("obp")
        if obp_value is not None:
            obp_map[team_id] = float(obp_value)

    return obp_map


def get_team_era_map(season: int) -> dict[int, float]:
    """Fetch season-level team ERA for all MLB teams (pitching-strength proxy)."""
    era_map: dict[int, float] = {}

    try:
        teams = statsapi.get("teams", {"sportId": 1, "season": season})
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch teams for ERA lookup ({season}).") from exc

    for team in teams.get("teams", []):
        team_id = team["id"]
        try:
            stats = statsapi.get(
                "team_stats",
                {
                    "teamId": team_id,
                    "season": season,
                    "group": "pitching",
                    "stats": "season",
                    "sportIds": 1,
                },
            )
        except Exception:
            continue

        splits = stats.get("stats", [])
        if not splits:
            continue

        stat_block = splits[0].get("splits", [])
        if not stat_block:
            continue

        era_value = stat_block[0].get("stat", {}).get("era")
        if era_value is not None:
            try:
                era_map[team_id] = float(era_value)
            except (TypeError, ValueError):
                continue

    return era_map


def compute_rolling_features(
    games_df: pd.DataFrame,
    window: int = 10,
    obp_maps: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Build training features with rolling run averages, season OBP, and run targets."""
    if games_df.empty:
        return pd.DataFrame()

    team_logs = build_team_game_logs(games_df)
    if obp_maps is None:
        obp_maps = {season: get_team_obp_map(season) for season in sorted(games_df["season"].unique())}

    rows: list[dict] = []

    for game in games_df.itertuples(index=False):
        home_log = team_logs.get(game.home_id)
        away_log = team_logs.get(game.away_id)
        if home_log is None or away_log is None:
            continue

        home_stats = _rolling_stats(home_log, game.game_date, window)
        away_stats = _rolling_stats(away_log, game.game_date, window)
        if home_stats is None or away_stats is None:
            continue

        season_obp = obp_maps.get(game.season, {})
        home_obp = season_obp.get(game.home_id)
        away_obp = season_obp.get(game.away_id)
        if home_obp is None or away_obp is None:
            continue

        home_runs_scored, home_runs_allowed, _ = home_stats
        away_runs_scored, away_runs_allowed, _ = away_stats

        rows.append(
            {
                "game_id": game.game_id,
                "game_date": game.game_date,
                "season": game.season,
                "home_id": game.home_id,
                "away_id": game.away_id,
                "home_name": game.home_name,
                "away_name": game.away_name,
                "venue_id": getattr(game, "venue_id", None),
                "home_avg_runs_scored": home_runs_scored,
                "home_avg_runs_allowed": home_runs_allowed,
                "home_team_obp": home_obp,
                "away_avg_runs_scored": away_runs_scored,
                "away_avg_runs_allowed": away_runs_allowed,
                "away_team_obp": away_obp,
                "home_team_runs": int(game.home_score),
                "away_team_runs": int(game.away_score),
                "home_score": int(game.home_score),
                "away_score": int(game.away_score),
                "home_win": int(game.home_win),
            }
        )

    return pd.DataFrame(rows)


def build_training_dataset(
    games_df: pd.DataFrame,
    window: int = 10,
    obp_maps: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Return a training dataset with features, weather, and raw run targets."""
    dataset = compute_rolling_features(games_df, window=window, obp_maps=obp_maps)
    if dataset.empty:
        return dataset

    dataset = attach_weather_features(dataset, games_df)
    if dataset.empty:
        return dataset

    missing_targets = [column for column in TARGET_COLUMNS if column not in dataset.columns]
    if missing_targets:
        raise ValueError(f"Training dataset missing target columns: {', '.join(missing_targets)}")

    return dataset


def compute_f5_features(
    games_df: pd.DataFrame,
    window: int = 10,
    obp_maps: dict[int, dict[int, float]] | None = None,
    era_maps: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Build F5 training features with rolling F5 run averages and F5 run targets."""
    if games_df.empty:
        return pd.DataFrame()

    games_df = attach_f5_runs(games_df)
    games_df = games_df.dropna(subset=["home_f5_runs", "away_f5_runs"]).reset_index(drop=True)
    if games_df.empty:
        return pd.DataFrame()

    team_logs = build_team_game_logs(games_df)
    seasons = sorted(games_df["season"].unique())
    if obp_maps is None:
        obp_maps = {season: get_team_obp_map(season) for season in seasons}
    if era_maps is None:
        era_maps = {season: get_team_era_map(season) for season in seasons}

    rows: list[dict] = []

    for game in games_df.itertuples(index=False):
        home_log = team_logs.get(game.home_id)
        away_log = team_logs.get(game.away_id)
        if home_log is None or away_log is None:
            continue

        home_f5 = _rolling_f5_stats(home_log, game.game_date, window)
        away_f5 = _rolling_f5_stats(away_log, game.game_date, window)
        if home_f5 is None or away_f5 is None:
            continue

        season_obp = obp_maps.get(game.season, {})
        season_era = era_maps.get(game.season, {})
        home_obp = season_obp.get(game.home_id)
        away_obp = season_obp.get(game.away_id)
        home_era = season_era.get(game.home_id)
        away_era = season_era.get(game.away_id)
        if None in (home_obp, away_obp, home_era, away_era):
            continue

        home_f5_scored, home_f5_allowed = home_f5
        away_f5_scored, away_f5_allowed = away_f5

        rows.append(
            {
                "game_id": game.game_id,
                "game_date": game.game_date,
                "season": game.season,
                "home_id": game.home_id,
                "away_id": game.away_id,
                "home_name": game.home_name,
                "away_name": game.away_name,
                "venue_id": getattr(game, "venue_id", None),
                "home_f5_avg_runs_scored": home_f5_scored,
                "home_f5_avg_runs_allowed": home_f5_allowed,
                "home_team_obp": home_obp,
                "home_team_era": home_era,
                "away_f5_avg_runs_scored": away_f5_scored,
                "away_f5_avg_runs_allowed": away_f5_allowed,
                "away_team_obp": away_obp,
                "away_team_era": away_era,
                "home_f5_runs": int(game.home_f5_runs),
                "away_f5_runs": int(game.away_f5_runs),
            }
        )

    return pd.DataFrame(rows)


def build_f5_training_dataset(
    games_df: pd.DataFrame,
    window: int = 10,
    obp_maps: dict[int, dict[int, float]] | None = None,
    era_maps: dict[int, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """Return an F5 training dataset with features, weather, and F5 run targets."""
    dataset = compute_f5_features(
        games_df, window=window, obp_maps=obp_maps, era_maps=era_maps
    )
    if dataset.empty:
        return dataset

    dataset = attach_weather_features(dataset, games_df)
    if dataset.empty:
        return dataset

    missing_targets = [column for column in F5_TARGET_COLUMNS if column not in dataset.columns]
    if missing_targets:
        raise ValueError(
            f"F5 training dataset missing target columns: {', '.join(missing_targets)}"
        )

    return dataset


def build_f5_prediction_row(
    home_id: int,
    away_id: int,
    as_of_date: date | None,
    games_df: pd.DataFrame,
    window: int = 10,
    venue_id: int | None = None,
) -> pd.DataFrame:
    """Build a single F5 feature row for an upcoming matchup."""
    if as_of_date is None:
        as_of_date = date.today()

    as_of_ts = pd.Timestamp(as_of_date)
    games_df = attach_f5_runs(games_df)
    prior_games = games_df[games_df["game_date"] < as_of_ts]
    prior_games = prior_games.dropna(subset=["home_f5_runs", "away_f5_runs"])
    if prior_games.empty:
        raise ValueError("No historical F5 games available before the selected date.")

    team_logs = build_team_game_logs(prior_games)
    home_log = team_logs.get(home_id)
    away_log = team_logs.get(away_id)
    if home_log is None or away_log is None:
        raise ValueError("One or both teams have no F5 game history in the loaded dataset.")

    home_f5 = _rolling_f5_stats(home_log, as_of_ts, window)
    away_f5 = _rolling_f5_stats(away_log, as_of_ts, window)
    if home_f5 is None or away_f5 is None:
        raise ValueError(
            f"Not enough prior F5 games (need at least {MIN_PRIOR_GAMES}) for rolling averages."
        )

    season = as_of_date.year
    obp_map = get_team_obp_map(season)
    era_map = get_team_era_map(season)
    home_obp = obp_map.get(home_id)
    away_obp = obp_map.get(away_id)
    if home_obp is None or away_obp is None:
        raise ValueError(f"Could not load OBP stats for season {season}.")
    home_era = era_map.get(home_id)
    away_era = era_map.get(away_id)
    if home_era is None or away_era is None:
        raise ValueError(f"Could not load ERA stats for season {season}.")

    weather = get_game_weather(home_id, as_of_date, venue_id=venue_id)
    if weather is None:
        raise ValueError("Could not load weather data for this matchup from Open-Meteo.")

    home_f5_scored, home_f5_allowed = home_f5
    away_f5_scored, away_f5_allowed = away_f5

    return pd.DataFrame(
        [
            {
                "home_f5_avg_runs_scored": home_f5_scored,
                "home_f5_avg_runs_allowed": home_f5_allowed,
                "home_team_obp": home_obp,
                "home_team_era": home_era,
                "away_f5_avg_runs_scored": away_f5_scored,
                "away_f5_avg_runs_allowed": away_f5_allowed,
                "away_team_obp": away_obp,
                "away_team_era": away_era,
                "temperature": weather["temperature"],
                "wind_speed": weather["wind_speed"],
            }
        ]
    )


def build_prediction_row(
    home_id: int,
    away_id: int,
    as_of_date: date | None,
    games_df: pd.DataFrame,
    window: int = 10,
    venue_id: int | None = None,
    game_id: int | None = None,
) -> pd.DataFrame:
    """Build a single feature row for an upcoming matchup."""
    if as_of_date is None:
        as_of_date = date.today()

    as_of_ts = pd.Timestamp(as_of_date)
    prior_games = games_df[games_df["game_date"] < as_of_ts]
    if prior_games.empty:
        raise ValueError("No historical games available before the selected date.")

    team_logs = build_team_game_logs(prior_games)
    home_log = team_logs.get(home_id)
    away_log = team_logs.get(away_id)
    if home_log is None or away_log is None:
        raise ValueError("One or both teams have no game history in the loaded dataset.")

    home_stats = _rolling_stats(home_log, as_of_ts, window)
    away_stats = _rolling_stats(away_log, as_of_ts, window)
    if home_stats is None or away_stats is None:
        raise ValueError(
            f"Not enough prior games (need at least {MIN_PRIOR_GAMES}) for rolling averages."
        )

    season = as_of_date.year
    obp_map = get_team_obp_map(season)
    home_obp = obp_map.get(home_id)
    away_obp = obp_map.get(away_id)
    if home_obp is None or away_obp is None:
        raise ValueError(f"Could not load OBP stats for season {season}.")

    home_runs_scored, home_runs_allowed, _ = home_stats
    away_runs_scored, away_runs_allowed, _ = away_stats

    weather = get_game_weather(home_id, as_of_date, venue_id=venue_id)
    if weather is None:
        raise ValueError("Could not load weather data for this matchup from Open-Meteo.")

    return pd.DataFrame(
        [
            {
                "home_avg_runs_scored": home_runs_scored,
                "home_avg_runs_allowed": home_runs_allowed,
                "home_team_obp": home_obp,
                "away_avg_runs_scored": away_runs_scored,
                "away_avg_runs_allowed": away_runs_allowed,
                "away_team_obp": away_obp,
                "temperature": weather["temperature"],
                "wind_speed": weather["wind_speed"],
            }
        ]
    )


def fetch_games_for_date(game_date: date | None = None) -> list[dict]:
    """Return MLB schedule entries for a given date (defaults to today)."""
    if game_date is None:
        game_date = date.today()
    date_str = game_date.strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(date=date_str, sportId=1)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch games for {game_date.isoformat()}. Check your internet connection."
        ) from exc

    if not games:
        return []

    return [
        {
            "game_id": game["game_id"],
            "status": game.get("status", ""),
            "home_id": game["home_id"],
            "away_id": game["away_id"],
            "home_name": game["home_name"],
            "away_name": game["away_name"],
            "venue_id": game.get("venue_id"),
            "summary": game.get("summary", ""),
        }
        for game in games
        if game.get("game_type") == "R"
    ]


def fetch_todays_games() -> list[dict]:
    """Return today's MLB schedule entries."""
    return fetch_games_for_date(date.today())


def games_for_prediction(seasons: list[int]) -> pd.DataFrame:
    """Fetch games through today for rolling stats used in live predictions."""
    all_seasons = sorted(set(seasons + [date.today().year]))
    games_df = fetch_season_games(all_seasons)

    if games_df.empty:
        return games_df

    today = pd.Timestamp(date.today())
    extra_rows: list[dict] = []

    current_season = date.today().year
    if current_season not in all_seasons:
        start_date, _ = _season_range(current_season)
        try:
            recent = statsapi.schedule(
                start_date=start_date,
                end_date=today.strftime("%m/%d/%Y"),
                sportId=1,
            )
        except Exception:
            recent = None

        if recent:
            existing_ids = set(games_df["game_id"])
            for game in recent:
                if game.get("game_type") != "R" or game.get("status") != "Final":
                    continue
                if game["game_id"] in existing_ids:
                    continue
                home_score = game.get("home_score")
                away_score = game.get("away_score")
                if home_score is None or away_score is None:
                    continue
                extra_rows.append(
                    {
                        "game_id": game["game_id"],
                        "game_date": pd.to_datetime(game["game_date"]),
                        "season": current_season,
                        "home_id": game["home_id"],
                        "away_id": game["away_id"],
                        "home_name": game["home_name"],
                        "away_name": game["away_name"],
                        "venue_id": game.get("venue_id"),
                        "home_score": int(home_score),
                        "away_score": int(away_score),
                        "home_win": int(home_score > away_score),
                    }
                )

    if extra_rows:
        games_df = pd.concat([games_df, pd.DataFrame(extra_rows)], ignore_index=True)
        games_df = games_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    return games_df
