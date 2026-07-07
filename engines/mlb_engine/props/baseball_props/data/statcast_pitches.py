from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Literal

import pandas as pd

from baseball_props.data.statcast_feed import _fetch_statcast_pitcher_season_frame, _season_year
from baseball_props.data.data_health import safe_feature_slice
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

PlayerRole = Literal["pitcher", "batter"]

PITCH_TYPE_NAMES: dict[str, str] = {
    "FF": "4-SEAM FASTBALL",
    "FA": "FASTBALL",
    "FT": "2-SEAM FASTBALL",
    "SI": "SINKER",
    "FC": "CUTTER",
    "SL": "SLIDER",
    "ST": "SWEEPER",
    "SV": "SLURVE",
    "CU": "CURVEBALL",
    "KC": "KNUCKLE CURVE",
    "CS": "SLOW CURVE",
    "CH": "CHANGEUP",
    "FS": "SPLITTER",
    "FO": "FORKBALL",
    "SC": "SCREWBALL",
    "KN": "KNUCKLEBALL",
    "EP": "EEPHUS",
    "PO": "PITCHOUT",
    "UN": "UNKNOWN",
    "IN": "INTENT BALL",
}

# Baseball Savant Illustrator-style palette keyed by full pitch name.
PITCH_TYPE_COLORS: dict[str, str] = {
    "4-SEAM FASTBALL": "#FF007D",
    "FASTBALL": "#FF007D",
    "2-SEAM FASTBALL": "#FE9F01",
    "SINKER": "#FE9F01",
    "CUTTER": "#C6011F",
    "SLIDER": "#FFE176",
    "SWEEPER": "#823ACE",
    "SLURVE": "#FFE176",
    "CURVEBALL": "#00D18F",
    "KNUCKLE CURVE": "#00D18F",
    "SLOW CURVE": "#00D18F",
    "CHANGEUP": "#97D0FF",
    "SPLITTER": "#97D0FF",
    "FORKBALL": "#97D0FF",
    "SCREWBALL": "#97D0FF",
    "KNUCKLEBALL": "#867A7A",
    "EEPHUS": "#867A7A",
    "PITCHOUT": "#867A7A",
    "UNKNOWN": "#BBBBBB",
    "INTENT BALL": "#BBBBBB",
}

PITCH_LOCATION_CAP = 400
_REQUIRED_COLS = ("pitch_type", "plate_x", "plate_z")


def pitch_type_full_name(code: str) -> str:
    """Map Statcast pitch abbreviation to Savant-style full name."""
    key = str(code or "").strip().upper()
    if not key:
        return "UNKNOWN"
    return PITCH_TYPE_NAMES.get(key, key)


def pitch_type_color(full_name: str) -> str:
    """Return Savant hex color for a pitch type full name."""
    return PITCH_TYPE_COLORS.get(full_name, PITCH_TYPE_COLORS.get(full_name.upper(), "#BBBBBB"))


def _is_numeric_mlbam_id(player_id: str) -> bool:
    text = str(player_id).strip()
    return text.isdigit()


def _season_window(season: int | None) -> tuple[str, str]:
    year = season if season is not None else _season_year()
    start = date(year, 3, 1).isoformat()
    end = date.today().isoformat()
    return start, end


@lru_cache(maxsize=256)
def _fetch_batter_season_frame_cached(player_id: str, start: str, end: str) -> pd.DataFrame:
    try:
        from pybaseball import statcast_batter

        sc = statcast_batter(start, end, int(player_id))
        if sc is None or sc.empty:
            return pd.DataFrame()
        return sc
    except Exception as exc:
        logger.debug("Statcast batter pitch fetch failed for %s: %s", player_id, exc)
        return pd.DataFrame()


def fetch_pitcher_pitches(player_id: str, season: int | None = None) -> pd.DataFrame:
    """Fetch season-to-date pitch-by-pitch Statcast log for a pitcher."""
    if not _is_numeric_mlbam_id(player_id):
        return pd.DataFrame()
    return _fetch_statcast_pitcher_season_frame(str(player_id))


def fetch_batter_pitches(player_id: str, season: int | None = None) -> pd.DataFrame:
    """Fetch season-to-date pitch-by-pitch Statcast log for a batter."""
    if not _is_numeric_mlbam_id(player_id):
        return pd.DataFrame()
    start, end = _season_window(season)
    return _fetch_batter_season_frame_cached(str(player_id), start, end)


def clean_pitch_locations(df: pd.DataFrame, *, cap: int = PITCH_LOCATION_CAP) -> pd.DataFrame:
    """
    Extract plate location columns; drop missing values; add full pitch names.

    Returns empty DataFrame when input is empty or required columns are absent.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[*_REQUIRED_COLS, "pitch_type_name"])

    missing = [col for col in _REQUIRED_COLS if col not in df.columns]
    if missing:
        logger.debug("Pitch location frame missing columns: %s", missing)
        return pd.DataFrame(columns=[*_REQUIRED_COLS, "pitch_type_name"])

    out = df[list(_REQUIRED_COLS)].copy()
    if "game_date" in df.columns:
        out["game_date"] = df["game_date"]

    out = out.dropna(subset=["pitch_type"])
    out["pitch_type"] = out["pitch_type"].astype(str).str.strip().str.upper()
    out = out[out["pitch_type"].ne("") & out["pitch_type"].ne("NAN") & out["pitch_type"].ne("NONE")]
    out["plate_x"] = pd.to_numeric(out["plate_x"], errors="coerce")
    out["plate_z"] = pd.to_numeric(out["plate_z"], errors="coerce")
    out = out.dropna(subset=["plate_x", "plate_z"])

    if out.empty:
        return pd.DataFrame(columns=[*_REQUIRED_COLS, "pitch_type_name"])

    if "game_date" in out.columns:
        out = out.sort_values("game_date", ascending=False)

    if cap > 0 and len(out) > cap:
        out = out.head(cap)

    out["pitch_type_name"] = out["pitch_type"].map(pitch_type_full_name)
    return out[[*_REQUIRED_COLS, "pitch_type_name"]]


def pitch_locations_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize cleaned pitch locations for canvas JSON export."""
    cleaned = clean_pitch_locations(df)
    if cleaned.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in cleaned.iterrows():
        records.append(
            {
                "pitch_type": str(row["pitch_type"]),
                "pitch_type_name": str(row["pitch_type_name"]),
                "plate_x": round(float(row["plate_x"]), 3),
                "plate_z": round(float(row["plate_z"]), 3),
            }
        )
    return records


@lru_cache(maxsize=512)
def get_pitch_locations_for_player(
    player_id: str,
    role: PlayerRole,
    data_health_key: int = 0,
) -> tuple[dict[str, Any], ...]:
    """Cached pitch location records for export (immutable tuple for lru_cache)."""
    del data_health_key  # cache-bust hook when health reporting is enabled per-run

    def _fetch() -> tuple[dict[str, Any], ...]:
        if role == "pitcher":
            raw = fetch_pitcher_pitches(player_id)
        else:
            raw = fetch_batter_pitches(player_id)
        return tuple(pitch_locations_to_records(raw))

    records = safe_feature_slice(
        f"statcast_pitch_locations_{role}",
        _fetch,
        default=(),
        report=None,
        empty_check=lambda value: len(value) == 0,
    )
    return records if isinstance(records, tuple) else tuple(records)
