from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from baseball_props.config import (
    LEAGUE_AVG,
    LEAGUE_HITS_PER_GAME,
    LEAGUE_HITS_PER_PA,
    LEAGUE_SLG,
    LEAGUE_TB_PER_GAME,
    REGRESSION_PA_STABILIZATION,
    TB_PER_SLG_PA,
    TB_PER_WOBA_PA,
)
from baseball_props.core.adjustments import calculate_projected_rate
from baseball_props.data.injuries import injury_rust_multiplier, lookup_injury
from baseball_props.environment.parks import get_park_scoring_factor
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


def wrc_plus_to_woba(wrc_plus: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Convert wRC+ (100 = league avg) to wOBA scale using league anchor."""
    arr = np.asarray(wrc_plus, dtype=float)
    result = (arr / 100.0) * LEAGUE_AVG["woba"]
    if isinstance(wrc_plus, pd.Series):
        return pd.Series(result, index=wrc_plus.index)
    if isinstance(wrc_plus, float):
        return float(result.reshape(-1)[0])
    return result


def platoon_hitting_rate(df: pd.DataFrame) -> pd.Series:
    """
    Batter rate for today's SP handedness: platoon wRC+ (preferred) or platoon wOBA.

    Uses `matchup_wrc_plus` / `matchup_woba` from split merge, not season/global averages.
    """
    if "matchup_wrc_plus" in df.columns and df["matchup_wrc_plus"].notna().any():
        wrc = df["matchup_wrc_plus"].fillna(LEAGUE_AVG["wrc_plus"])
        return wrc_plus_to_woba(wrc)
    if "wrc_plus_split" in df.columns and df["wrc_plus_split"].notna().any():
        wrc = df["wrc_plus_split"].fillna(LEAGUE_AVG["wrc_plus"])
        return wrc_plus_to_woba(wrc)
    if "matchup_woba" in df.columns:
        return df["matchup_woba"].fillna(LEAGUE_AVG["woba"])
    if "effective_woba" in df.columns:
        logger.warning("No platoon split columns; falling back to effective_woba")
        return df["effective_woba"].fillna(LEAGUE_AVG["woba"])
    return pd.Series(LEAGUE_AVG["woba"], index=df.index)


def platoon_iso(df: pd.DataFrame) -> pd.Series:
    """Platoon ISO for today's SP handedness (extra-base power input)."""
    if "matchup_iso" in df.columns:
        return df["matchup_iso"].fillna(LEAGUE_AVG["iso"])
    if "effective_iso" in df.columns:
        return df["effective_iso"].fillna(LEAGUE_AVG["iso"])
    return pd.Series(LEAGUE_AVG["iso"], index=df.index)


def marcels_regress(
    player_stat: pd.Series,
    league_stat: float,
    sample_pa: pd.Series,
    *,
    stabilization_pa: float = REGRESSION_PA_STABILIZATION,
) -> pd.Series:
    """
    Marcels-style regression blend toward league average when PA is below stabilization.

    Regressed = (PA / stabilization) * Player_Stat + (1 - PA / stabilization) * League_Avg
    At PA >= stabilization, player stat is used in full.
    """
    pa = sample_pa.fillna(0).clip(lower=0)
    weight = (pa / stabilization_pa).clip(upper=1.0)
    return weight * player_stat + (1.0 - weight) * league_stat


def regress_rate_to_mean(
    player_rate: pd.Series,
    league_rate: float,
    sample_pa: pd.Series,
    *,
    full_weight_pa: float = REGRESSION_PA_STABILIZATION,
) -> pd.Series:
    """Backward-compatible alias for marcels_regress."""
    return marcels_regress(
        player_rate,
        league_rate,
        sample_pa,
        stabilization_pa=full_weight_pa,
    )


def _as_series(value: pd.Series | np.ndarray | float, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, np.ndarray):
        return pd.Series(value, index=index)
    return pd.Series(float(value), index=index)


def project_batter_total_bases(
    projected: pd.DataFrame,
    *,
    injury_lookup: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Situational total-bases projection for each batter-game row.

    1. Platoon split rate (wRC+ or wOBA vs today's SP hand)
    2. Opponent SP odds-ratio adjustment on contact (wOBA)
    3. Marcels regression (300 PA) on wOBA, ISO, and slugging proxy
    4. Slugging-weighted skill TB with heavy ISO penalty for low samples
    5. Final Marcels blend on skill TB when season_pa < 300:
       regressed = (season_pa/300)*skill_tb + (1 - season_pa/300)*LEAGUE_TB_PER_GAME
       At 300+ PA the player skill TB is used in full (weight capped at 1.0).
    6. Home park scoring factor
    """
    df = projected.copy()
    required = {"proj_pa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"project_batter_total_bases missing columns: {sorted(missing)}")

    platoon_rate = platoon_hitting_rate(df)
    platoon_iso_rate = platoon_iso(df)
    platoon_slg = platoon_rate + platoon_iso_rate

    opp_col = "opp_sp_woba_allowed"
    if opp_col not in df.columns:
        df[opp_col] = LEAGUE_AVG["woba"]
    adjusted_woba = _as_series(
        calculate_projected_rate(
            platoon_rate,
            df[opp_col].fillna(LEAGUE_AVG["woba"]),
            LEAGUE_AVG["woba"],
        ),
        df.index,
    )

    sample_pa = (
        df["season_pa"]
        if "season_pa" in df.columns
        else pd.Series(REGRESSION_PA_STABILIZATION, index=df.index)
    )

    regressed_woba = marcels_regress(adjusted_woba, LEAGUE_AVG["woba"], sample_pa)
    regressed_iso = marcels_regress(platoon_iso_rate, LEAGUE_AVG["iso"], sample_pa)
    regressed_slg = marcels_regress(platoon_slg, LEAGUE_SLG, sample_pa)

    contact_tb = regressed_woba * df["proj_pa"] * TB_PER_WOBA_PA
    power_ratio = (regressed_iso / LEAGUE_AVG["iso"]).clip(lower=0.5, upper=2.0)
    skill_tb_woba_iso = contact_tb * power_ratio
    skill_tb_slg = regressed_slg * df["proj_pa"] * TB_PER_SLG_PA
    skill_tb = (skill_tb_woba_iso + skill_tb_slg) / 2.0

    regressed_game_tb = marcels_regress(skill_tb, LEAGUE_TB_PER_GAME, sample_pa)

    park_ids = df["park_id"] if "park_id" in df.columns else pd.Series("", index=df.index)
    park_factor = park_ids.map(get_park_scoring_factor).fillna(1.0)

    df["platoon_woba"] = platoon_rate
    df["platoon_iso"] = platoon_iso_rate
    df["regressed_woba"] = regressed_woba
    df["regressed_iso"] = regressed_iso
    df["regressed_slg"] = regressed_slg
    df["skill_tb"] = skill_tb.round(3)
    df["regressed_game_tb"] = regressed_game_tb.round(3)
    df["park_tb_factor"] = park_factor
    df["proj_total_bases"] = (regressed_game_tb * park_factor).round(3)

    if injury_lookup:
        if "player_name" in df.columns:
            injury_records = df["player_name"].map(
                lambda name: lookup_injury(str(name), injury_lookup)
            )
            df["injury_multiplier"] = injury_records.map(injury_rust_multiplier)
            df["injury_status"] = injury_records.map(
                lambda rec: rec.get("status") if rec else None
            )
        else:
            logger.warning("No player_name column; skipping injury rust adjustments")
            df["injury_multiplier"] = 1.0
            df["injury_status"] = None
        df["proj_total_bases"] = (
            df["proj_total_bases"] * df["injury_multiplier"]
        ).round(3)
        adjusted = int((df["injury_multiplier"] != 1.0).sum())
        if adjusted:
            logger.info(
                "Applied injury rust multiplier to %d player-games (IL zeroed or returning rust)",
                adjusted,
            )

    logger.info(
        "Projected situational total bases for %d player-games (park factor range %.2f–%.2f)",
        len(df),
        park_factor.min(),
        park_factor.max(),
    )
    return df


def project_batter_hits(
    projected: pd.DataFrame,
    *,
    injury_lookup: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Situational hits projection for each batter-game row.

    Scales regressed contact quality (wOBA vs SP) to hits/PA, regresses toward league
    average for low-PA samples, then multiplies by projected PA and park factor.
    """
    df = projected.copy()
    if "proj_pa" not in df.columns:
        raise ValueError("project_batter_hits missing column: proj_pa")

    platoon_rate = platoon_hitting_rate(df)
    opp_col = "opp_sp_woba_allowed"
    if opp_col not in df.columns:
        df[opp_col] = LEAGUE_AVG["woba"]
    adjusted_woba = _as_series(
        calculate_projected_rate(
            platoon_rate,
            df[opp_col].fillna(LEAGUE_AVG["woba"]),
            LEAGUE_AVG["woba"],
        ),
        df.index,
    )

    sample_pa = (
        df["season_pa"]
        if "season_pa" in df.columns
        else pd.Series(REGRESSION_PA_STABILIZATION, index=df.index)
    )

    regressed_woba = marcels_regress(adjusted_woba, LEAGUE_AVG["woba"], sample_pa)
    woba_scale = (regressed_woba / LEAGUE_AVG["woba"]).clip(lower=0.5, upper=1.8)
    hits_per_pa = LEAGUE_HITS_PER_PA * woba_scale

    if "proj_k_pct" in df.columns:
        contact_rate = (1.0 - df["proj_k_pct"].fillna(LEAGUE_AVG["k_pct"])).clip(lower=0.35)
        league_contact = 1.0 - LEAGUE_AVG["k_pct"]
        hits_per_pa = hits_per_pa * (contact_rate / league_contact)

    skill_hits = hits_per_pa * df["proj_pa"]
    regressed_game_hits = marcels_regress(skill_hits, LEAGUE_HITS_PER_GAME, sample_pa)

    park_ids = df["park_id"] if "park_id" in df.columns else pd.Series("", index=df.index)
    park_factor = park_ids.map(get_park_scoring_factor).fillna(1.0)

    df["hits_per_pa"] = hits_per_pa.round(4)
    df["skill_hits"] = skill_hits.round(3)
    df["regressed_game_hits"] = regressed_game_hits.round(3)
    df["proj_hits"] = (regressed_game_hits * park_factor).round(3)

    if injury_lookup and "player_name" in df.columns:
        injury_records = df["player_name"].map(
            lambda name: lookup_injury(str(name), injury_lookup)
        )
        if "injury_multiplier" not in df.columns:
            df["injury_multiplier"] = injury_records.map(injury_rust_multiplier)
        df["proj_hits"] = (df["proj_hits"] * df["injury_multiplier"]).round(3)
    elif "injury_multiplier" in df.columns:
        df["proj_hits"] = (df["proj_hits"] * df["injury_multiplier"]).round(3)

    logger.info(
        "Projected situational hits for %d player-games (mean %.2f hits)",
        len(df),
        float(df["proj_hits"].mean()),
    )
    return df
