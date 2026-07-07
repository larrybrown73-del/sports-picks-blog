"""Open-Meteo weather fetching and caching for game-day features."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from venues import get_stadium_coords

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
WEATHER_CACHE_FILE = CACHE_DIR / "weather_cache.json"


def _load_cache() -> dict[str, dict[str, float]]:
    if not WEATHER_CACHE_FILE.exists():
        return {}
    try:
        return json.loads(WEATHER_CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, dict[str, float]]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    WEATHER_CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")


def _cache_key(lat: float, lon: float, day: date) -> str:
    return f"{lat:.4f},{lon:.4f}:{day.isoformat()}"


def _fetch_daily_weather(
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
    use_forecast: bool = False,
) -> dict[str, dict[str, float]]:
    """Fetch daily mean temperature (F) and wind speed (mph) for a date range."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,wind_speed_10m_mean",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "auto",
    }
    url = FORECAST_URL if use_forecast else ARCHIVE_URL
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    times = daily.get("time", [])
    temperatures = daily.get("temperature_2m_mean", [])
    wind_speeds = daily.get("wind_speed_10m_mean", [])

    results: dict[str, dict[str, float]] = {}
    for day_str, temp, wind in zip(times, temperatures, wind_speeds):
        if temp is None or wind is None:
            continue
        results[day_str] = {"temperature": float(temp), "wind_speed": float(wind)}
    return results


def prefetch_weather_for_games(games_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Batch-fetch and cache weather for all games in the dataframe."""
    if games_df.empty:
        return _load_cache()

    cache = _load_cache()
    if "venue_id" not in games_df.columns:
        games_df = games_df.copy()
        games_df["venue_id"] = None

    location_dates: dict[tuple[float, float], set[date]] = {}
    for game in games_df.itertuples(index=False):
        game_day = pd.Timestamp(game.game_date).date()
        venue_id = getattr(game, "venue_id", None)
        coords = get_stadium_coords(game.home_id, venue_id)
        if coords is None:
            continue
        location_dates.setdefault(coords, set()).add(game_day)

    today = date.today()
    for (lat, lon), days in location_dates.items():
        missing_days = [
            day
            for day in sorted(days)
            if _cache_key(lat, lon, day) not in cache
        ]
        if not missing_days:
            continue

        historical_days = [day for day in missing_days if day < today]
        forecast_days = [day for day in missing_days if day >= today]

        for chunk_start, chunk_end, use_forecast in _date_chunks(
            historical_days, forecast_days
        ):
            try:
                fetched = _fetch_daily_weather(
                    lat, lon, chunk_start, chunk_end, use_forecast=use_forecast
                )
            except requests.RequestException:
                continue

            for day_str, values in fetched.items():
                day = date.fromisoformat(day_str)
                cache[_cache_key(lat, lon, day)] = values

    _save_cache(cache)
    return cache


def _date_chunks(
    historical_days: list[date],
    forecast_days: list[date],
) -> list[tuple[date, date, bool]]:
    chunks: list[tuple[date, date, bool]] = []
    if historical_days:
        chunks.append((min(historical_days), max(historical_days), False))
    if forecast_days:
        chunks.append((min(forecast_days), max(forecast_days), True))
    return chunks


def get_game_weather(
    home_team_id: int,
    game_date: date,
    venue_id: int | None = None,
    cache: dict[str, dict[str, float]] | None = None,
) -> dict[str, float] | None:
    """Return temperature and wind_speed for a game at the home stadium."""
    coords = get_stadium_coords(home_team_id, venue_id)
    if coords is None:
        return None

    lat, lon = coords
    if cache is None:
        cache = _load_cache()

    key = _cache_key(lat, lon, game_date)
    if key in cache:
        return cache[key]

    today = date.today()
    use_forecast = game_date >= today
    try:
        fetched = _fetch_daily_weather(lat, lon, game_date, game_date, use_forecast=use_forecast)
    except requests.RequestException:
        return None

    values = fetched.get(game_date.isoformat())
    if values is not None:
        cache[key] = values
        _save_cache(cache)
    return values


def attach_weather_features(
    features_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add temperature and wind_speed columns using Open-Meteo data."""
    if features_df.empty:
        return features_df

    if "venue_id" not in games_df.columns:
        games_df = games_df.copy()
        games_df["venue_id"] = None

    weather_cache = prefetch_weather_for_games(games_df)
    game_lookup = (
        games_df.drop_duplicates(subset="game_id", keep="first")
        .set_index("game_id")[["home_id", "game_date", "venue_id"]]
    )

    temperatures: list[float | None] = []
    wind_speeds: list[float | None] = []

    for row in features_df.itertuples(index=False):
        if row.game_id not in game_lookup.index:
            temperatures.append(None)
            wind_speeds.append(None)
            continue

        game_info = game_lookup.loc[row.game_id]
        # Defensive: if duplicate game_ids slipped through, .loc returns a
        # DataFrame; collapse to the first row so we always have scalars.
        if isinstance(game_info, pd.DataFrame):
            game_info = game_info.iloc[0]

        game_day = pd.Timestamp(game_info["game_date"]).date()
        venue_id = game_info["venue_id"]
        if pd.isna(venue_id):
            venue_id = None
        else:
            venue_id = int(venue_id)

        weather = get_game_weather(
            int(game_info["home_id"]),
            game_day,
            venue_id=venue_id,
            cache=weather_cache,
        )
        if weather is None:
            temperatures.append(None)
            wind_speeds.append(None)
        else:
            temperatures.append(weather["temperature"])
            wind_speeds.append(weather["wind_speed"])

    enriched = features_df.copy()
    enriched["temperature"] = temperatures
    enriched["wind_speed"] = wind_speeds
    enriched = enriched.dropna(subset=["temperature", "wind_speed"])
    return enriched.reset_index(drop=True)
