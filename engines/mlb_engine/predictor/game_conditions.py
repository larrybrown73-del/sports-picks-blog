"""MLB game-day weather parsing and run-environment multipliers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import statsapi

from config import (
    RUN_ENV_BOOST_MULTIPLIER,
    RUN_ENV_NEUTRAL_MULTIPLIER,
    RUN_ENV_REDUCE_MULTIPLIER,
    WEATHER_COLD_TEMP_F,
    WEATHER_HOT_TEMP_F,
)
from data_health import safe_feature_fetch

logger = logging.getLogger(__name__)

_WIND_SPEED_RE = re.compile(r"(\d+)\s*mph", re.IGNORECASE)
_WIND_OUT_RE = re.compile(r"\bout\b", re.IGNORECASE)
_WIND_IN_RE = re.compile(r"\bin\b", re.IGNORECASE)


@dataclass(frozen=True)
class GameConditions:
    temperature_f: int | None
    wind_raw: str
    wind_speed_mph: float | None
    wind_direction: str
    run_env_multiplier: float
    display_temp: str
    display_wind: str


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_wind(wind_raw: str) -> tuple[float | None, str]:
    if not wind_raw or not str(wind_raw).strip():
        return None, "unknown"

    text = str(wind_raw).strip()
    speed_match = _WIND_SPEED_RE.search(text)
    speed = float(speed_match.group(1)) if speed_match else None

    if _WIND_OUT_RE.search(text):
        direction = "out"
    elif _WIND_IN_RE.search(text):
        direction = "in"
    else:
        direction = "neutral"

    return speed, direction


def _compute_run_env_multiplier(temperature_f: int | None, wind_direction: str) -> float:
    hot = temperature_f is not None and temperature_f > WEATHER_HOT_TEMP_F
    cold = temperature_f is not None and temperature_f < WEATHER_COLD_TEMP_F
    wind_out = wind_direction == "out"
    wind_in = wind_direction == "in"

    # If hot/out and cold/in both trigger, prefer the boost (offensive environment).
    if hot or wind_out:
        return RUN_ENV_BOOST_MULTIPLIER
    if cold or wind_in:
        return RUN_ENV_REDUCE_MULTIPLIER
    return RUN_ENV_NEUTRAL_MULTIPLIER


def _neutral_conditions() -> GameConditions:
    return GameConditions(
        temperature_f=None,
        wind_raw="",
        wind_speed_mph=None,
        wind_direction="unknown",
        run_env_multiplier=RUN_ENV_NEUTRAL_MULTIPLIER,
        display_temp="Unknown",
        display_wind="Unknown",
    )


def fetch_game_conditions(game_id: int) -> GameConditions:
    """Fetch pre-game weather from MLB Stats API and derive run-environment multiplier."""

    def _fetch() -> GameConditions:
        payload = statsapi.get("game", {"gamePk": game_id, "hydrate": "weather,venue"})
        weather = payload.get("gameData", {}).get("weather") or {}
        temp_f = _safe_int(weather.get("temp"))
        wind_raw = str(weather.get("wind") or "").strip()
        wind_speed, wind_direction = _parse_wind(wind_raw)
        multiplier = _compute_run_env_multiplier(temp_f, wind_direction)
        display_temp = f"{temp_f}F" if temp_f is not None else "Unknown"
        display_wind = wind_raw if wind_raw else "Unknown"
        return GameConditions(
            temperature_f=temp_f,
            wind_raw=wind_raw,
            wind_speed_mph=wind_speed,
            wind_direction=wind_direction,
            run_env_multiplier=multiplier,
            display_temp=display_temp,
            display_wind=display_wind,
        )

    return safe_feature_fetch(
        "mlb_game_weather",
        _fetch,
        fallback=_neutral_conditions(),
    )


def apply_run_environment(
    home_runs: float,
    away_runs: float,
    multiplier: float,
) -> tuple[float, float]:
    """Scale both predicted runs by the same run-environment multiplier."""
    if multiplier == 1.0:
        return home_runs, away_runs
    return home_runs * multiplier, away_runs * multiplier
