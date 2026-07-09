from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from baseball_props.config import LEAGUE_AVG, REGRESSION_PA_STABILIZATION
from baseball_props.data.data_health import DataHealthReport
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

RATE_COL_MAP = {
    "woba": "wOBA",
    "iso": "ISO",
    "k_pct": "K%",
    "bb_pct": "BB%",
    "wrc_plus": "wRC+",
}


def _safe_rate(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def _season_year() -> int:
    today = date.today()
    return today.year if today.month >= 3 else today.year - 1


def _fetch_season_batting() -> pd.DataFrame:
    try:
        from pybaseball import batting_stats

        year = _season_year()
        df = batting_stats(year, qual=1)
        if df is None or df.empty:
            raise ValueError("empty batting_stats")
        df = df.copy()
        df["mlbam_id"] = df["IDfg"].astype(str) if "IDfg" not in df.columns else df.get("IDfg")
        if "IDfg" in df.columns:
            df["player_id_lookup"] = df["Name"].astype(str).str.lower()
        return df
    except Exception as exc:
        logger.warning("pybaseball batting_stats unavailable: %s", exc)
        return pd.DataFrame()


def _player_row_from_season(season_df: pd.DataFrame, player_id: str) -> dict[str, float]:
    """Map MLBAM id to season rate dict; fall back to league averages."""
    defaults = {
        "season_woba": LEAGUE_AVG["woba"],
        "season_iso": LEAGUE_AVG["iso"],
        "season_k_pct": LEAGUE_AVG["k_pct"],
        "season_bb_pct": LEAGUE_AVG["bb_pct"],
        "season_wrc_plus": LEAGUE_AVG["wrc_plus"],
        "season_hard_hit_pct": LEAGUE_AVG["hard_hit_pct"],
        "season_pa": REGRESSION_PA_STABILIZATION,
    }
    if season_df.empty:
        return defaults

    # batting_stats uses FanGraphs ID; match by attempting MLBAM via pybaseball lookup
    try:
        from pybaseball import playerid_reverse_lookup

        lookup = playerid_reverse_lookup([int(player_id)], key_type="mlbam")
        if lookup.empty:
            logger.info("Season stat lookup: no FanGraphs id for MLBAM %s", player_id)
            return defaults
        fg_id = int(lookup.iloc[0]["key_fangraphs"])
        row = season_df[season_df["IDfg"] == fg_id]
        if row.empty:
            logger.info("Season stat lookup: no batting row for MLBAM %s", player_id)
            return defaults
        r = row.iloc[0]
        pa = float(r.get("PA", 1) or 1)
        if "Hard%" in r.index and pd.notna(r.get("Hard%")):
            hard_val = float(r.get("Hard%"))
            # FanGraphs Hard% is 0-100 scale
            hard_pct = hard_val / 100.0 if hard_val > 1 else hard_val
        else:
            hard_pct = defaults["season_hard_hit_pct"]
        return {
            "season_woba": float(r.get("wOBA", defaults["season_woba"])),
            "season_iso": float(r.get("ISO", defaults["season_iso"])),
            "season_k_pct": _safe_rate(float(r.get("SO", 0)), pa),
            "season_bb_pct": _safe_rate(float(r.get("BB", 0)), pa),
            "season_wrc_plus": float(r.get("wRC+", defaults["season_wrc_plus"])),
            "season_hard_hit_pct": hard_pct,
            "season_pa": pa,
        }
    except Exception as exc:
        logger.debug("Season stat lookup failed for %s: %s", player_id, exc)
        return defaults


def _statcast_pa_frame(sc: pd.DataFrame) -> pd.DataFrame:
    """Keep terminal pitch rows (one per plate appearance) for rate stats."""
    if sc is None or sc.empty:
        return sc
    if "events" not in sc.columns:
        return sc
    events = sc["events"].astype(str).str.strip()
    pa = sc.loc[sc["events"].notna() & events.ne("") & events.ne("nan")]
    return pa if not pa.empty else sc


def _rates_from_statcast_frame(sc: pd.DataFrame) -> dict[str, float]:
    """Compute rate stats from a statcast event slice (pitch rows filtered to PAs)."""
    if sc is None or sc.empty:
        return {}
    sc = _statcast_pa_frame(sc)
    pa = len(sc)
    if pa == 0:
        return {}
    events = sc["events"].fillna("")
    k = int(events.eq("strikeout").sum())
    bb = int(events.isin(["walk", "hit_by_pitch"]).sum())
    hits = int(events.isin(["single", "double", "triple", "home_run"]).sum())
    tb = (
        int(events.eq("single").sum())
        + 2 * int(events.eq("double").sum())
        + 3 * int(events.eq("triple").sum())
        + 4 * int(events.eq("home_run").sum())
    )
    if "woba_value" in sc.columns:
        woba = _safe_rate(float(sc["woba_value"].fillna(0).sum()), pa)
    else:
        woba = _safe_rate(hits + bb, pa) * 0.32
    iso = _safe_rate(tb - hits, pa)
    hard = sc.get("launch_speed", pd.Series(dtype=float))
    hard_hit = float((hard >= 95).mean()) if len(hard.dropna()) else LEAGUE_AVG["hard_hit_pct"]
    wrc = (woba / LEAGUE_AVG["woba"]) * 100.0 if LEAGUE_AVG["woba"] else LEAGUE_AVG["wrc_plus"]
    return {
        "woba": woba,
        "iso": iso,
        "k_pct": _safe_rate(k, pa),
        "bb_pct": _safe_rate(bb, pa),
        "wrc_plus": wrc,
        "hard_hit_pct": hard_hit,
    }


def _fetch_statcast_season_frame(player_id: str) -> pd.DataFrame:
    """Season-to-date statcast pitch log for a batter (single Savant pull)."""
    try:
        from pybaseball import statcast_batter

        year = _season_year()
        start = date(year, 3, 1)
        end = date.today()
        sc = statcast_batter(start.isoformat(), end.isoformat(), int(player_id))
        if sc is None or sc.empty:
            return pd.DataFrame()
        return sc
    except Exception as exc:
        logger.debug("Statcast season fetch failed for %s: %s", player_id, exc)
        return pd.DataFrame()


def _season_dict_from_statcast_rates(rates: dict[str, float], pa_count: int) -> dict[str, float]:
    """Map statcast rate dict into season baseline fields."""
    return {
        "season_woba": rates.get("woba", LEAGUE_AVG["woba"]),
        "season_iso": rates.get("iso", LEAGUE_AVG["iso"]),
        "season_k_pct": rates.get("k_pct", LEAGUE_AVG["k_pct"]),
        "season_bb_pct": rates.get("bb_pct", LEAGUE_AVG["bb_pct"]),
        "season_wrc_plus": rates.get("wrc_plus", LEAGUE_AVG["wrc_plus"]),
        "season_hard_hit_pct": rates.get("hard_hit_pct", LEAGUE_AVG["hard_hit_pct"]),
        "season_pa": float(max(pa_count, 1)),
    }


def _rolling_from_statcast(player_id: str, days: int) -> dict[str, float]:
    """Compute rolling rates from statcast batted-ball events."""
    sc = _fetch_statcast_season_frame(player_id)
    if sc.empty:
        return {}
    end = date.today()
    sc = sc[sc["game_date"] >= (end - timedelta(days=days)).isoformat()]
    return _rates_from_statcast_frame(sc)


def _rolling_from_statcast_frame(sc: pd.DataFrame, days: int) -> dict[str, float]:
    """Compute rolling rates from an already-fetched statcast frame."""
    if sc.empty:
        return {}
    end = date.today()
    window = sc[sc["game_date"] >= (end - timedelta(days=days)).isoformat()]
    return _rates_from_statcast_frame(window)


def _xbh_rate_from_statcast_frame(sc: pd.DataFrame) -> float | None:
    """Extra-base hit rate (double, triple, HR per PA) from a Statcast batter frame."""
    if sc is None or sc.empty:
        return None
    pa_frame = _statcast_pa_frame(sc)
    pa = len(pa_frame)
    if pa == 0:
        return None
    events = pa_frame["events"].fillna("")
    xbh = int(
        events.isin(["double", "triple", "home_run"]).sum()
    )
    return float(xbh / pa)


def _xbh_profile_from_statcast_frame(sc: pd.DataFrame) -> dict[str, float] | None:
    """XBH rate and singles/hits ratio from a Statcast batter frame."""
    if sc is None or sc.empty:
        return None
    pa_frame = _statcast_pa_frame(sc)
    pa = len(pa_frame)
    if pa == 0:
        return None
    events = pa_frame["events"].fillna("")
    singles = int(events.eq("single").sum())
    hits = int(events.isin(["single", "double", "triple", "home_run"]).sum())
    xbh = int(events.isin(["double", "triple", "home_run"]).sum())
    singles_ratio = float(singles / hits) if hits > 0 else 0.0
    return {
        "xbh_rate": float(xbh / pa),
        "singles_ratio": singles_ratio,
        "pa": float(pa),
    }


def compute_xbh_profile(
    player_id: str,
    games: int | None = None,
    *,
    statcast_frame: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """
    Rolling XBH profile over the last N distinct games.

    Returns xbh_rate (XBH/PA), singles_ratio (singles/hits), and pa count.
    """
    from baseball_props.config import TB_XBH_ROLLING_GAMES

    window_games = games if games is not None else TB_XBH_ROLLING_GAMES
    pid = str(player_id).strip()
    if not pid.isdigit():
        return None

    sc = statcast_frame if statcast_frame is not None else _fetch_statcast_season_frame(pid)
    if sc.empty or "game_date" not in sc.columns:
        return None

    pa_frame = _statcast_pa_frame(sc)
    if pa_frame.empty:
        return None

    game_dates = (
        pa_frame[["game_date"]]
        .drop_duplicates()
        .sort_values("game_date", ascending=True)
    )
    if game_dates.empty:
        return None

    recent_dates = set(game_dates.tail(window_games)["game_date"].astype(str))
    recent = pa_frame[pa_frame["game_date"].astype(str).isin(recent_dates)]
    return _xbh_profile_from_statcast_frame(recent)


def _contact_profile_from_statcast_frame(sc: pd.DataFrame) -> dict[str, float] | None:
    """Rolling K%, contact%, and BABIP from a Statcast batter frame."""
    if sc is None or sc.empty:
        return None
    pa_frame = _statcast_pa_frame(sc)
    pa = len(pa_frame)
    if pa == 0:
        return None
    events = pa_frame["events"].fillna("").astype(str)
    k = int(events.eq("strikeout").sum())
    bb = int(events.isin(["walk", "hit_by_pitch"]).sum())
    hr = int(events.eq("home_run").sum())
    hits = int(events.isin(["single", "double", "triple", "home_run"]).sum())
    ab = max(pa - bb, 1)
    babip_denom = max(ab - k - hr, 1)
    babip = float((hits - hr) / babip_denom)

    contact_pct: float | None = None
    if "description" in sc.columns:
        desc = sc["description"].astype(str)
        swings = desc.str.contains("swinging", case=False, na=False)
        whiffs = desc.str.contains("swinging_strike", case=False, na=False)
        swing_count = int(swings.sum())
        if swing_count > 0:
            contact_pct = float((swing_count - int(whiffs.sum())) / swing_count)
    if contact_pct is None:
        contact_pct = float((pa - k) / max(pa - bb, 1))

    return {
        "k_pct": float(k / pa),
        "contact_pct": contact_pct,
        "babip": babip,
        "pa": float(pa),
    }


def compute_contact_profile(
    player_id: str,
    games: int | None = None,
    *,
    statcast_frame: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """
    Rolling contact profile over the last N distinct games.

    Returns k_pct, contact_pct, babip, and pa count for hits prop guardrails.
    """
    from baseball_props.config import HITS_CONTACT_ROLLING_GAMES

    window_games = games if games is not None else HITS_CONTACT_ROLLING_GAMES
    pid = str(player_id).strip()
    if not pid.isdigit():
        return None

    sc = statcast_frame if statcast_frame is not None else _fetch_statcast_season_frame(pid)
    if sc.empty or "game_date" not in sc.columns:
        return None

    pa_frame = _statcast_pa_frame(sc)
    if pa_frame.empty:
        return None

    game_dates = (
        pa_frame[["game_date"]]
        .drop_duplicates()
        .sort_values("game_date", ascending=True)
    )
    if game_dates.empty:
        return None

    recent_dates = set(game_dates.tail(window_games)["game_date"].astype(str))
    recent_pa = pa_frame[pa_frame["game_date"].astype(str).isin(recent_dates)]
    recent_full = sc[sc["game_date"].astype(str).isin(recent_dates)]
    return _contact_profile_from_statcast_frame(recent_full if not recent_full.empty else recent_pa)


def consecutive_hit_games(
    player_id: str,
    *,
    statcast_frame: pd.DataFrame | None = None,
) -> int:
    """
    Count consecutive games (most recent backward) with at least one hit.
    """
    pid = str(player_id).strip()
    if not pid.isdigit():
        return 0

    sc = statcast_frame if statcast_frame is not None else _fetch_statcast_season_frame(pid)
    if sc.empty or "game_date" not in sc.columns:
        return 0

    pa_frame = _statcast_pa_frame(sc)
    if pa_frame.empty:
        return 0

    events = pa_frame.get("events")
    if events is None:
        return 0

    hit_events = {"single", "double", "triple", "home_run"}
    hits_by_game = (
        pa_frame.assign(_hit=events.astype(str).isin(hit_events))
        .groupby("game_date", sort=True)["_hit"]
        .any()
    )
    if hits_by_game.empty:
        return 0

    streak = 0
    for had_hit in reversed(hits_by_game.tolist()):
        if had_hit:
            streak += 1
        else:
            break
    return streak


def apply_hits_momentum_multipliers(
    adjusted_proj: float,
    player_id: str,
    adjustments: dict[str, float],
    warnings: list[str],
) -> float:
    """Apply hot-hand projection boosts before probability / EV evaluation."""
    from baseball_props.config import (
        HITS_RECENT_CONTACT_GAMES,
        HITS_RECENT_CONTACT_PCT_MIN,
        HITS_STREAK_MIN_GAMES,
        PLAYER_HIT_STREAK_BONUS,
        RECENT_CONTACT_BONUS,
    )

    proj = float(adjusted_proj)

    hit_streak = consecutive_hit_games(player_id)
    adjustments["hit_streak_games"] = float(hit_streak)
    if hit_streak >= HITS_STREAK_MIN_GAMES:
        proj *= PLAYER_HIT_STREAK_BONUS
        adjustments["hit_streak_bonus"] = PLAYER_HIT_STREAK_BONUS

    recent_contact = compute_contact_profile(
        player_id,
        games=HITS_RECENT_CONTACT_GAMES,
    )
    if recent_contact:
        recent_pct = recent_contact.get("contact_pct")
        adjustments["recent_contact_pct"] = (
            float(recent_pct) if recent_pct is not None else float("nan")
        )
        if recent_pct is not None and recent_pct > HITS_RECENT_CONTACT_PCT_MIN:
            proj *= RECENT_CONTACT_BONUS
            adjustments["recent_contact_bonus"] = RECENT_CONTACT_BONUS
    else:
        warnings.append(
            "Missing-Data Warning: recent_contact_profile — momentum spike skipped"
        )

    return proj


def rolling_xbh_rate_last_n_games(
    player_id: str,
    n: int | None = None,
) -> float | None:
    """
    XBH frequency over the last N distinct games in the season Statcast log.

    Returns None when the player has no numeric MLBAM id or insufficient data.
    """
    from baseball_props.config import TB_XBH_ROLLING_GAMES

    window_games = n if n is not None else TB_XBH_ROLLING_GAMES
    pid = str(player_id).strip()
    if not pid.isdigit():
        return None

    sc = _fetch_statcast_season_frame(pid)
    if sc.empty or "game_date" not in sc.columns:
        return None

    pa_frame = _statcast_pa_frame(sc)
    if pa_frame.empty:
        return None

    game_dates = (
        pa_frame[["game_date"]]
        .drop_duplicates()
        .sort_values("game_date", ascending=True)
    )
    if game_dates.empty:
        return None

    recent_dates = set(game_dates.tail(window_games)["game_date"].astype(str))
    recent = pa_frame[pa_frame["game_date"].astype(str).isin(recent_dates)]
    return _xbh_rate_from_statcast_frame(recent)


def _split_row_from_rates(
    player_id: str,
    split: str,
    rates: dict[str, float],
) -> dict[str, object]:
    return {
        "player_id": player_id,
        "split": split,
        "woba": rates.get("woba", LEAGUE_AVG["woba"]),
        "iso": rates.get("iso", LEAGUE_AVG["iso"]),
        "k_pct": rates.get("k_pct", LEAGUE_AVG["k_pct"]),
        "bb_pct": rates.get("bb_pct", LEAGUE_AVG["bb_pct"]),
        "wrc_plus": rates.get("wrc_plus", LEAGUE_AVG["wrc_plus"]),
        "hard_hit_pct": rates.get("hard_hit_pct", LEAGUE_AVG["hard_hit_pct"]),
    }


def _platoon_from_statcast_frame(sc: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Platoon rates via statcast pitcher hand from a season frame."""
    if sc.empty or "p_throws" not in sc.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for hand, split in (("L", "vs_lhp"), ("R", "vs_rhp")):
        slice_df = sc[sc["p_throws"] == hand]
        rates = _rates_from_statcast_frame(slice_df)
        if rates:
            out[split] = rates
    return out


def _platoon_from_statcast(player_id: str) -> dict[str, dict[str, float]]:
    """Season-to-date platoon rates via statcast pitcher hand."""
    sc = _fetch_statcast_season_frame(player_id)
    return _platoon_from_statcast_frame(sc)


def _platoon_from_fangraphs(fg_id: int, year: int) -> dict[str, dict[str, float]]:
    """Try FanGraphs platoon split export via pybaseball."""
    try:
        from pybaseball import get_splits

        splits_df = get_splits(fg_id, year=year, split_type="plato")
        if splits_df is None or splits_df.empty:
            return {}
        out: dict[str, dict[str, float]] = {}
        label_map = {
            "vs lhp": "vs_lhp",
            "vs rhp": "vs_rhp",
            "vs lhp (platoon)": "vs_lhp",
            "vs rhp (platoon)": "vs_rhp",
        }
        for _, row in splits_df.iterrows():
            label = str(row.iloc[0]).strip().lower()
            split_key = label_map.get(label)
            if split_key is None:
                continue
            pa = float(row.get("PA", 0) or 0)
            if pa <= 0:
                continue
            hard = row.get("Hard%")
            hard_pct = LEAGUE_AVG["hard_hit_pct"]
            if hard is not None and pd.notna(hard):
                hard_val = float(hard)
                hard_pct = hard_val / 100.0 if hard_val > 1 else hard_val
            out[split_key] = {
                "woba": float(row.get("wOBA", LEAGUE_AVG["woba"])),
                "iso": float(row.get("ISO", LEAGUE_AVG["iso"])),
                "k_pct": _safe_rate(float(row.get("SO", 0)), pa),
                "bb_pct": _safe_rate(float(row.get("BB", 0)), pa),
                "wrc_plus": float(row.get("wRC+", LEAGUE_AVG["wrc_plus"])),
                "hard_hit_pct": hard_pct,
            }
        return out
    except Exception as exc:
        logger.debug("FanGraphs platoon failed for fg_id %s: %s", fg_id, exc)
        return {}


def _regressed_platoon_split(season: dict[str, float], split: str) -> dict[str, float]:
    """Small handedness regression when real splits unavailable."""
    lhp_bump = 1.04 if split == "vs_lhp" else 0.97
    return {
        "woba": season["season_woba"] * lhp_bump,
        "iso": season["season_iso"] * lhp_bump,
        "k_pct": min(max(season["season_k_pct"] / lhp_bump, 0.05), 0.45),
        "bb_pct": min(max(season["season_bb_pct"] * lhp_bump, 0.02), 0.25),
        "wrc_plus": season["season_wrc_plus"] * lhp_bump,
        "hard_hit_pct": min(max(season["season_hard_hit_pct"] * lhp_bump, 0.15), 0.65),
    }


def _fetch_platoon_splits(
    player_id: str,
    season: dict[str, float],
    fg_id: int | None = None,
    *,
    statcast_frame: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Build vs_lhp / vs_rhp split rows from real data with fallbacks."""
    platoon: dict[str, dict[str, float]] = {}
    if fg_id is not None:
        platoon = _platoon_from_fangraphs(fg_id, _season_year())
    if len(platoon) < 2:
        if statcast_frame is not None and not statcast_frame.empty:
            platoon = {**platoon, **_platoon_from_statcast_frame(statcast_frame)}
        else:
            platoon = {**platoon, **_platoon_from_statcast(player_id)}

    rows: list[dict[str, object]] = []
    for split in ("vs_lhp", "vs_rhp"):
        if split in platoon:
            rows.append(_split_row_from_rates(player_id, split, platoon[split]))
        else:
            rows.append(
                _split_row_from_rates(player_id, split, _regressed_platoon_split(season, split))
            )
    return rows


def _lookup_fangraphs_id(player_id: str) -> int | None:
    try:
        from pybaseball import playerid_reverse_lookup

        lookup = playerid_reverse_lookup([int(player_id)], key_type="mlbam")
        if lookup.empty:
            return None
        return int(lookup.iloc[0]["key_fangraphs"])
    except Exception:
        return None


def build_player_baselines_and_splits(
    player_ids: list[str],
    *,
    data_health: DataHealthReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build player_baselines and matchup_splits for live MLBAM ids."""
    report = data_health or DataHealthReport()
    season_df = _fetch_season_batting()
    fangraphs_available = not season_df.empty
    if not fangraphs_available:
        logger.warning(
            "FanGraphs season batting unavailable; using Statcast season-to-date for hitter rates"
        )

    baseline_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for pid in player_ids:
        try:
            statcast_frame = _fetch_statcast_season_frame(pid)
        except Exception as exc:
            report.record_missing(
                f"statcast_season_{pid}",
                detail=f"error ({exc}); using league-average baseline",
            )
            statcast_frame = pd.DataFrame()

        try:
            season = _player_row_from_season(season_df, pid)
        except Exception as exc:
            report.record_missing(
                f"season_batting_{pid}",
                detail=f"error ({exc}); using league-average baseline",
            )
            season = _player_row_from_season(pd.DataFrame(), pid)
        if not fangraphs_available and not statcast_frame.empty:
            season_rates = _rates_from_statcast_frame(statcast_frame)
            if season_rates:
                pa_count = len(_statcast_pa_frame(statcast_frame))
                season = _season_dict_from_statcast_rates(season_rates, pa_count)

        fg_id = _lookup_fangraphs_id(pid)
        roll14 = _rolling_from_statcast_frame(statcast_frame, 14) if not statcast_frame.empty else {}
        roll30 = _rolling_from_statcast_frame(statcast_frame, 30) if not statcast_frame.empty else {}

        def _roll(key: str, window: dict[str, float], season_key: str) -> float:
            if key in window:
                return window[key]
            return float(season[season_key])

        baseline_rows.append(
            {
                "player_id": pid,
                "season_woba": season["season_woba"],
                "roll14_woba": _roll("woba", roll14, "season_woba"),
                "roll30_woba": _roll("woba", roll30, "season_woba"),
                "season_iso": season["season_iso"],
                "roll14_iso": _roll("iso", roll14, "season_iso"),
                "roll30_iso": _roll("iso", roll30, "season_iso"),
                "season_k_pct": season["season_k_pct"],
                "roll14_k_pct": _roll("k_pct", roll14, "season_k_pct"),
                "roll30_k_pct": _roll("k_pct", roll30, "season_k_pct"),
                "season_bb_pct": season["season_bb_pct"],
                "roll14_bb_pct": _roll("bb_pct", roll14, "season_bb_pct"),
                "roll30_bb_pct": _roll("bb_pct", roll30, "season_bb_pct"),
                "season_wrc_plus": season["season_wrc_plus"],
                "roll14_wrc_plus": _roll("wrc_plus", roll14, "season_wrc_plus"),
                "roll30_wrc_plus": _roll("wrc_plus", roll30, "season_wrc_plus"),
                "season_hard_hit_pct": season["season_hard_hit_pct"],
                "roll14_hard_hit_pct": _roll("hard_hit_pct", roll14, "season_hard_hit_pct"),
                "roll30_hard_hit_pct": _roll("hard_hit_pct", roll30, "season_hard_hit_pct"),
                "season_pa": season.get("season_pa", REGRESSION_PA_STABILIZATION),
            }
        )

        split_rows.extend(
            _fetch_platoon_splits(pid, season, fg_id, statcast_frame=statcast_frame)
        )

    return pd.DataFrame(baseline_rows), pd.DataFrame(split_rows)


def _resolve_pitcher_name(
    sp_id: str,
    names: dict[str, str],
    fg_name: str | None = None,
) -> str:
    if sp_id in names and names[sp_id]:
        return names[sp_id]
    if fg_name:
        return fg_name
    return f"SP {sp_id}"


def _normalize_mlbam_id(sp_id: str | int) -> str:
    text = str(sp_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _parse_innings_pitched(ip_value: str | float | int) -> float:
    text = str(ip_value)
    if "." not in text:
        return float(text)
    whole, frac = text.split(".", 1)
    return float(whole) + float(frac[:1] or 0) / 3.0


def _fallback_pitcher_bundle(*, is_starter: bool) -> dict[str, float | bool]:
    from baseball_props.config import (
        FALLBACK_RELIEF_OUTS,
        FALLBACK_STARTER_OUTS,
        LEAGUE_AVG,
        LEAGUE_PITCHES_PER_OUT,
    )

    outs = FALLBACK_STARTER_OUTS if is_starter else FALLBACK_RELIEF_OUTS
    return {
        "avg_outs_last5": outs,
        "pitch_efficiency": LEAGUE_PITCHES_PER_OUT,
        "gs": 1.0 if is_starter else 0.0,
        "is_true_starter": is_starter,
        "sp_k_pct": LEAGUE_AVG["k_pct"],
        "sp_bb_pct": LEAGUE_AVG["bb_pct"],
        "avg_bf_per_start": 25.0 if is_starter else 6.0,
    }


def _pitcher_bundle_from_rates(
    *,
    gs: float,
    ip: float,
    tbf: float,
    so: float,
    bb: float,
    pitches: float,
) -> dict[str, float | bool]:
    from baseball_props.config import (
        FALLBACK_STARTER_OUTS,
        LEAGUE_AVG,
        LEAGUE_PITCHES_PER_OUT,
        LEAGUE_STARTER_OUTS,
        MAX_PITCH_EFFICIENCY,
        MAX_PROJ_OUTS,
        MIN_PITCH_EFFICIENCY,
    )

    league_bf = 25.0
    avg_outs = LEAGUE_STARTER_OUTS
    efficiency = LEAGUE_PITCHES_PER_OUT
    sp_k_pct = LEAGUE_AVG["k_pct"]
    sp_bb_pct = LEAGUE_AVG["bb_pct"]
    avg_bf_per_start = league_bf
    is_true_starter = False
    if gs > 0 and ip > 0:
        is_true_starter = True
        outs_per_start = (ip / gs) * 3
        if outs_per_start <= 0 or outs_per_start > MAX_PROJ_OUTS:
            logger.warning(
                "Implausible outs/start %.1f (IP=%.1f GS=%.0f); using fallback %.1f",
                outs_per_start,
                ip,
                gs,
                FALLBACK_STARTER_OUTS,
            )
            avg_outs = FALLBACK_STARTER_OUTS
        else:
            avg_outs = round(outs_per_start, 2)
        if outs_per_start > 0 and pitches > 0:
            raw_eff = (pitches / gs) / outs_per_start
            efficiency = round(
                max(MIN_PITCH_EFFICIENCY, min(raw_eff, MAX_PITCH_EFFICIENCY)), 2
            )
        if tbf > 0:
            sp_k_pct = _safe_rate(so, tbf)
            sp_bb_pct = _safe_rate(bb, tbf)
            avg_bf_per_start = round(tbf / gs, 2)
    return {
        "avg_outs_last5": avg_outs,
        "pitch_efficiency": efficiency,
        "gs": gs,
        "is_true_starter": is_true_starter,
        "sp_k_pct": round(sp_k_pct, 4),
        "sp_bb_pct": round(sp_bb_pct, 4),
        "avg_bf_per_start": avg_bf_per_start,
    }


_PITCHING_STATS_CACHE: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
_FANGraphS_PITCHING_UNAVAILABLE: set[int] = set()


def _get_pitching_stats_frames(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if year in _PITCHING_STATS_CACHE:
        return _PITCHING_STATS_CACHE[year]
    qual1 = pd.DataFrame()
    qual0 = pd.DataFrame()
    try:
        from pybaseball import pitching_stats

        qual1 = pitching_stats(year, qual=1)
        qual0 = pitching_stats(year, qual=0)
    except Exception as exc:
        logger.warning("FanGraphs pitching_stats unavailable for %s: %s", year, exc)
        _FANGraphS_PITCHING_UNAVAILABLE.add(year)
    if qual1.empty and qual0.empty:
        _FANGraphS_PITCHING_UNAVAILABLE.add(year)
    _PITCHING_STATS_CACHE[year] = (qual1, qual0)
    return qual1, qual0


def fangraphs_pitching_available(year: int) -> bool:
    qual1, qual0 = _get_pitching_stats_frames(year)
    return not qual1.empty or not qual0.empty


def _fetch_mlb_pitcher_season_stats(mlbam_id: str, season: int) -> dict[str, float] | None:
    try:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{mlbam_id}/stats",
            params={"stats": "season", "group": "pitching", "season": season},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        stats = payload.get("stats") or []
        if not stats:
            return None
        splits = stats[0].get("splits") or []
        if not splits:
            return None
        stat = splits[0].get("stat") or {}
        gs = float(stat.get("gamesStarted", 0) or 0)
        ip = _parse_innings_pitched(stat.get("inningsPitched", 0) or 0)
        return {
            "gs": gs,
            "ip": ip,
            "tbf": float(stat.get("battersFaced", 0) or 0),
            "so": float(stat.get("strikeOuts", 0) or 0),
            "bb": float(stat.get("baseOnBalls", 0) or 0),
            "pitches": float(stat.get("numberOfPitches", ip * 15) or ip * 15),
        }
    except Exception as exc:
        logger.debug("MLB API pitcher stats failed for %s: %s", mlbam_id, exc)
        return None


def _resolve_pitcher_stats_bundle(
    sp_id_norm: str,
    *,
    ps_qual1: pd.DataFrame,
    ps_qual0: pd.DataFrame,
    season: int,
) -> tuple[dict[str, float | bool], str, str | None]:
    fg_name: str | None = None
    try:
        from pybaseball import playerid_reverse_lookup

        lookup = playerid_reverse_lookup([int(sp_id_norm)], key_type="mlbam")
        if not lookup.empty:
            fg_id = int(lookup.iloc[0]["key_fangraphs"])
            row = pd.DataFrame()
            source = "fangraphs_missing"
            if not ps_qual1.empty:
                row = ps_qual1[ps_qual1["IDfg"] == fg_id]
                if not row.empty:
                    source = "fangraphs_qual1"
            if row.empty and not ps_qual0.empty:
                row = ps_qual0[ps_qual0["IDfg"] == fg_id]
                if not row.empty:
                    source = "fangraphs_qual0"
            if not row.empty:
                r = row.iloc[0]
                fg_name = str(r.get("Name", "")).strip() or None
                bundle = _pitcher_bundle_from_rates(
                    gs=float(r.get("GS", 0) or 0),
                    ip=float(r.get("IP", 0) or 0),
                    tbf=float(r.get("TBF", 0) or 0),
                    so=float(r.get("SO", 0) or 0),
                    bb=float(r.get("BB", 0) or 0),
                    pitches=float(r.get("Pitches", 0) or 0),
                )
                if bundle["gs"] > 0:
                    return bundle, source, fg_name
    except Exception as exc:
        logger.debug("FanGraphs lookup failed for pitcher %s: %s", sp_id_norm, exc)

    mlb = _fetch_mlb_pitcher_season_stats(sp_id_norm, season)
    if mlb and mlb["gs"] > 0:
        bundle = _pitcher_bundle_from_rates(
            gs=mlb["gs"],
            ip=mlb["ip"],
            tbf=mlb["tbf"],
            so=mlb["so"],
            bb=mlb["bb"],
            pitches=mlb["pitches"],
        )
        return bundle, "mlb_api", fg_name

    return {}, "unresolved", fg_name


def _average_pitcher_bundles(bundles: list[dict[str, float | bool]]) -> dict[str, float | bool]:
    if not bundles:
        return _fallback_pitcher_bundle(is_starter=True)
    keys = ("avg_outs_last5", "pitch_efficiency", "sp_k_pct", "sp_bb_pct", "avg_bf_per_start")
    averaged = {key: round(sum(float(b[key]) for b in bundles) / len(bundles), 4) for key in keys}
    averaged["gs"] = max(float(b["gs"]) for b in bundles)
    averaged["is_true_starter"] = averaged["gs"] > 0
    if averaged["gs"] <= 0:
        averaged.update(_fallback_pitcher_bundle(is_starter=False))
    return averaged


def _build_sp_team_map(slate_games: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for _, game in slate_games.iterrows():
        home_sp = _normalize_mlbam_id(game.get("sp_home_id", ""))
        away_sp = _normalize_mlbam_id(game.get("sp_away_id", ""))
        if home_sp and home_sp != "nan":
            mapping[home_sp] = str(game["home_team_id"])
        if away_sp and away_sp != "nan":
            mapping[away_sp] = str(game["away_team_id"])
    return mapping


def build_pitcher_tendencies(
    slate_games: pd.DataFrame,
    pitcher_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build pitcher tendency rows for probable starters (GS > 0 mask)."""
    names = {_normalize_mlbam_id(k): v for k, v in (pitcher_names or {}).items()}
    sp_ids: set[str] = set()
    for col in ("sp_home_id", "sp_away_id"):
        for raw in slate_games[col].astype(str).tolist():
            norm = _normalize_mlbam_id(raw)
            if norm and norm != "nan":
                sp_ids.add(norm)

    season = _season_year()
    ps_qual1, ps_qual0 = _get_pitching_stats_frames(season)
    resolved: dict[str, dict[str, Any]] = {}
    slate_bundles: list[dict[str, float | bool]] = []

    for sp_id_norm in sorted(sp_ids):
        bundle, source, fg_name = _resolve_pitcher_stats_bundle(
            sp_id_norm,
            ps_qual1=ps_qual1,
            ps_qual0=ps_qual0,
            season=season,
        )
        resolved[sp_id_norm] = {
            "bundle": bundle,
            "source": source,
            "fg_name": fg_name,
        }
        if bundle and bundle.get("gs", 0) > 0:
            slate_bundles.append(bundle)

    rows: list[dict[str, object]] = []
    for sp_id_norm in sorted(sp_ids):
        entry = resolved[sp_id_norm]
        bundle = entry["bundle"]
        source = entry["source"]
        fg_name = entry["fg_name"]
        if not bundle:
            bundle = _fallback_pitcher_bundle(is_starter=True)
            source = "starter_default"
        elif float(bundle.get("gs", 0)) <= 0:
            bundle = _fallback_pitcher_bundle(is_starter=False)
            source = "relief_default"

        rows.append(
            {
                "pitcher_id": sp_id_norm,
                "pitcher_name": _resolve_pitcher_name(sp_id_norm, names, fg_name),
                "avg_outs_last5": bundle["avg_outs_last5"],
                "pitch_efficiency": bundle["pitch_efficiency"],
                "gs": bundle["gs"],
                "is_true_starter": bundle["is_true_starter"],
                "sp_k_pct": bundle["sp_k_pct"],
                "sp_bb_pct": bundle["sp_bb_pct"],
                "avg_bf_per_start": bundle["avg_bf_per_start"],
                "stats_source": source,
            }
        )

    return pd.DataFrame(rows)


def build_team_pitching_stub(team_ids: list[str]) -> pd.DataFrame:
    """Neutral team pitching frame using league averages."""
    rows: list[dict[str, object]] = []
    for team_id in team_ids:
        for role in ("sp", "bullpen"):
            rows.append(
                {
                    "team_id": team_id,
                    "role": role,
                    "woba_allowed": LEAGUE_AVG["woba"],
                    "iso_allowed": LEAGUE_AVG["iso"],
                    "k_pct": LEAGUE_AVG["k_pct"],
                    "bb_pct": LEAGUE_AVG["bb_pct"],
                }
            )
    return pd.DataFrame(rows)


def build_park_weather_stub(park_ids: list[str]) -> pd.DataFrame:
    """Park/weather stub for live slates using static scoring factors."""
    from baseball_props.environment.parks import get_park_scoring_factor

    rows = [
        {
            "park_id": pid,
            "park_factor_runs": get_park_scoring_factor(str(pid)),
            "park_factor_hr": 1.0,
            "temp_f": 72.0,
            "wind_mph": 5.0,
            "wind_dir": "calm",
        }
        for pid in park_ids
    ]
    return pd.DataFrame(rows)


def _fetch_statcast_pitcher_season_frame(pitcher_id: str) -> pd.DataFrame:
    """Season-to-date statcast pitch log for a pitcher."""
    try:
        from pybaseball import statcast_pitcher

        year = _season_year()
        start = date(year, 3, 1)
        end = date.today()
        sc = statcast_pitcher(start.isoformat(), end.isoformat(), int(pitcher_id))
        if sc is None or sc.empty:
            return pd.DataFrame()
        return sc
    except Exception as exc:
        logger.debug("Statcast pitcher fetch failed for %s: %s", pitcher_id, exc)
        return pd.DataFrame()


def _pitcher_platoon_from_statcast_frame(sc: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pitcher allowed rates split by batter stand (L/R)."""
    if sc is None or sc.empty or "stand" not in sc.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for stand, split in (("L", "vs_lhb"), ("R", "vs_rhb")):
        slice_df = sc[sc["stand"].astype(str).str.upper() == stand]
        rates = _rates_from_statcast_frame(slice_df)
        if rates:
            pa = len(_statcast_pa_frame(slice_df))
            rates["bf"] = float(pa)
            out[split] = rates
    return out


def _pitcher_platoon_from_fangraphs(fg_id: int, year: int) -> dict[str, dict[str, float]]:
    """FanGraphs pitcher platoon splits (vs LHB / vs RHB)."""
    try:
        from pybaseball import get_splits

        splits_df = get_splits(fg_id, year=year, split_type="plato")
        if splits_df is None or splits_df.empty:
            return {}
        out: dict[str, dict[str, float]] = {}
        label_map = {
            "vs lhb": "vs_lhb",
            "vs rhb": "vs_rhb",
            "vs lhb (platoon)": "vs_lhb",
            "vs rhb (platoon)": "vs_rhb",
        }
        for _, row in splits_df.iterrows():
            label = str(row.iloc[0]).strip().lower()
            split_key = label_map.get(label)
            if split_key is None:
                continue
            bf = float(row.get("PA", row.get("TBF", 0)) or 0)
            if bf <= 0:
                continue
            out[split_key] = {
                "woba_allowed": float(row.get("wOBA", LEAGUE_AVG["woba"])),
                "iso_allowed": float(row.get("ISO", LEAGUE_AVG["iso"])),
                "k_pct": _safe_rate(float(row.get("SO", 0)), bf),
                "bb_pct": _safe_rate(float(row.get("BB", 0)), bf),
                "bf": bf,
            }
        return out
    except Exception as exc:
        logger.debug("FanGraphs pitcher platoon failed for fg_id %s: %s", fg_id, exc)
        return {}


def _regressed_pitcher_platoon_split(season_woba: float, split: str) -> dict[str, float]:
    """Small handedness regression for pitcher allowed rates when splits missing."""
    bump = 1.03 if split == "vs_lhb" else 0.97
    woba = season_woba * bump
    return {
        "woba_allowed": woba,
        "iso_allowed": LEAGUE_AVG["iso"] * bump,
        "k_pct": min(max(LEAGUE_AVG["k_pct"] / bump, 0.05), 0.45),
        "bb_pct": min(max(LEAGUE_AVG["bb_pct"] * bump, 0.02), 0.25),
        "bf": 0.0,
    }


def _pitcher_split_row(
    pitcher_id: str,
    split: str,
    rates: dict[str, float],
) -> dict[str, object]:
    return {
        "pitcher_id": pitcher_id,
        "split": split,
        "woba_allowed": rates.get("woba_allowed", rates.get("woba", LEAGUE_AVG["woba"])),
        "iso_allowed": rates.get("iso_allowed", rates.get("iso", LEAGUE_AVG["iso"])),
        "k_pct": rates.get("k_pct", LEAGUE_AVG["k_pct"]),
        "bb_pct": rates.get("bb_pct", LEAGUE_AVG["bb_pct"]),
        "bf": float(rates.get("bf", 0.0)),
    }


def _fetch_pitcher_platoon_splits(
    pitcher_id: str,
    *,
    fg_id: int | None = None,
    statcast_frame: pd.DataFrame | None = None,
    season_woba: float | None = None,
) -> list[dict[str, object]]:
    """Build vs_lhb / vs_rhb rows for a pitcher."""
    platoon: dict[str, dict[str, float]] = {}
    if fg_id is not None:
        platoon = _pitcher_platoon_from_fangraphs(fg_id, _season_year())
    if len(platoon) < 2:
        if statcast_frame is not None and not statcast_frame.empty:
            platoon = {**platoon, **_pitcher_platoon_from_statcast_frame(statcast_frame)}
        elif statcast_frame is None:
            platoon = {
                **platoon,
                **_pitcher_platoon_from_statcast_frame(
                    _fetch_statcast_pitcher_season_frame(pitcher_id)
                ),
            }

    base_woba = season_woba if season_woba is not None else LEAGUE_AVG["woba"]
    rows: list[dict[str, object]] = []
    for split in ("vs_lhb", "vs_rhb"):
        if split in platoon:
            rows.append(_pitcher_split_row(pitcher_id, split, platoon[split]))
        else:
            rows.append(
                _pitcher_split_row(
                    pitcher_id,
                    split,
                    _regressed_pitcher_platoon_split(base_woba, split),
                )
            )
    return rows


def build_pitcher_platoon_splits(sp_ids: list[str]) -> pd.DataFrame:
    """Build pitcher platoon allowed-rate splits for probable starters."""
    if not sp_ids:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "split",
                "woba_allowed",
                "iso_allowed",
                "k_pct",
                "bb_pct",
                "bf",
            ]
        )

    season = _season_year()
    qual1, qual0 = _get_pitching_stats_frames(season)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for raw_id in sp_ids:
        pid = _normalize_mlbam_id(raw_id)
        if not pid or pid == "nan" or pid in seen:
            continue
        seen.add(pid)

        fg_id = None
        season_woba: float | None = None
        try:
            from pybaseball import playerid_reverse_lookup

            lookup = playerid_reverse_lookup([int(pid)], key_type="mlbam")
            if not lookup.empty:
                fg_id = int(lookup.iloc[0]["key_fangraphs"])
        except Exception:
            fg_id = None

        bundle, _, _ = _resolve_pitcher_stats_bundle(
            pid, ps_qual1=qual1, ps_qual0=qual0, season=season
        )
        if bundle:
            season_woba = LEAGUE_AVG["woba"] * (
                1.0 + (float(bundle.get("sp_k_pct", LEAGUE_AVG["k_pct"])) - LEAGUE_AVG["k_pct"])
            )

        statcast_frame = _fetch_statcast_pitcher_season_frame(pid)
        rows.extend(
            _fetch_pitcher_platoon_splits(
                pid,
                fg_id=fg_id,
                statcast_frame=statcast_frame,
                season_woba=season_woba,
            )
        )

    return pd.DataFrame(rows)
