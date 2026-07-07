from __future__ import annotations

import pandas as pd

from baseball_props.config import (
    FALLBACK_RELIEF_OUTS,
    FALLBACK_STARTER_OUTS,
    LEAGUE_AVG,
    LEAGUE_PITCHES_PER_OUT,
    LEAGUE_STARTER_OUTS,
    MAX_PITCH_EFFICIENCY,
    MAX_PROJ_OUTS,
    MIN_PITCH_EFFICIENCY,
    PITCHES_PER_STRIKEOUT,
    PITCHES_PER_WALK,
)
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_LEAGUE_STARTER_BF: float = 25.0


def _baseline_outs_for_role(gs: float) -> float:
    """Realistic outs when FanGraphs/stats feeds fail."""
    return FALLBACK_STARTER_OUTS if gs > 0 else FALLBACK_RELIEF_OUTS


def _clamp_pitch_efficiency(efficiency: float) -> float:
    if not pd.notna(efficiency) or efficiency <= 0:
        return LEAGUE_PITCHES_PER_OUT
    return float(max(MIN_PITCH_EFFICIENCY, min(efficiency, MAX_PITCH_EFFICIENCY)))


def _sanitize_avg_outs(avg_outs: float, gs: float) -> float:
    if not pd.notna(avg_outs) or avg_outs <= 0:
        return _baseline_outs_for_role(gs)
    capped = float(min(avg_outs, MAX_PROJ_OUTS))
    if gs <= 0:
        return min(capped, FALLBACK_RELIEF_OUTS)
    if capped > MAX_PROJ_OUTS * 0.95:
        logger.warning(
            "Capping implausible avg_outs_last5 %.1f to starter fallback %.1f",
            avg_outs,
            FALLBACK_STARTER_OUTS,
        )
        return FALLBACK_STARTER_OUTS
    return capped


def _compute_proj_outs(avg_outs: float, efficiency: float, gs: float) -> float:
    raw_avg = float(avg_outs)
    sanitized = _sanitize_avg_outs(avg_outs, gs)
    if gs <= 0:
        return min(FALLBACK_RELIEF_OUTS, MAX_PROJ_OUTS)
    used_starter_fallback = sanitized == FALLBACK_STARTER_OUTS and (
        not pd.notna(raw_avg) or raw_avg <= 0 or raw_avg > MAX_PROJ_OUTS * 0.95
    )
    if used_starter_fallback:
        return min(FALLBACK_STARTER_OUTS, MAX_PROJ_OUTS)
    efficiency = _clamp_pitch_efficiency(efficiency)
    eff_adj = LEAGUE_PITCHES_PER_OUT / efficiency
    return min(sanitized * eff_adj, MAX_PROJ_OUTS)


def _apply_final_outs_guardrails(result: pd.DataFrame) -> pd.DataFrame:
    """Hard ceiling on projected outs before downstream tables."""
    if result.empty or "proj_outs" not in result.columns:
        return result
    over = result["proj_outs"] > MAX_PROJ_OUTS
    if over.any():
        logger.warning(
            "Capping %d pitcher(s) with proj_outs above %.1f",
            int(over.sum()),
            MAX_PROJ_OUTS,
        )
    result = result.copy()
    result["proj_outs"] = result["proj_outs"].clip(upper=MAX_PROJ_OUTS)
    return result


def _hook_floor_pitches(proj_outs: float) -> float:
    from baseball_props.config import HOOK_FLOOR_MAX_PITCHES, HOOK_FLOOR_MIN_PITCHES

    innings = proj_outs / 3.0
    span = 6.5 - 5.0
    return HOOK_FLOOR_MIN_PITCHES + (innings - 5.0) / span * (
        HOOK_FLOOR_MAX_PITCHES - HOOK_FLOOR_MIN_PITCHES
    )


def _apply_hook_floor(
    raw_pitches: float,
    proj_outs: float,
    avg_outs_last5: float,
    gs: float,
) -> float:
    from baseball_props.config import (
        HOOK_FLOOR_MAX_PITCHES,
        HOOK_FLOOR_MIN_OUTS,
        MARQUEE_STARTER_OUTS,
    )

    if (
        gs > 0
        and avg_outs_last5 >= MARQUEE_STARTER_OUTS
        and proj_outs >= HOOK_FLOOR_MIN_OUTS
    ):
        floor = _hook_floor_pitches(proj_outs)
        raw_pitches = max(raw_pitches, floor)
        raw_pitches = min(raw_pitches, HOOK_FLOOR_MAX_PITCHES)
    return raw_pitches


def _opposing_team_pa(
    game_id: str,
    pitcher_team_id: str,
    game: pd.Series,
    projected_batters: pd.DataFrame | None,
) -> float:
    if projected_batters is not None and not projected_batters.empty:
        if "proj_pa" not in projected_batters.columns:
            raise ValueError("projected_batters must include proj_pa column")
        home = str(game["home_team_id"])
        away = str(game["away_team_id"])
        opp_team = away if pitcher_team_id == home else home
        mask = (projected_batters["game_id"] == game_id) & (
            projected_batters["team_id"] == opp_team
        )
        return float(projected_batters.loc[mask, "proj_pa"].sum())
    return _LEAGUE_STARTER_BF


def _resolve_proj_bf(
    proj_outs: float,
    avg_outs: float,
    avg_bf_per_start: float | None,
    opp_pa: float,
) -> float:
    safe_avg_outs = max(float(avg_outs) if pd.notna(avg_outs) else 0.0, FALLBACK_RELIEF_OUTS)
    if avg_bf_per_start and safe_avg_outs > 0:
        return avg_bf_per_start * (proj_outs / safe_avg_outs)
    if opp_pa > 0:
        return opp_pa * (proj_outs / LEAGUE_STARTER_OUTS)
    return _LEAGUE_STARTER_BF * (proj_outs / LEAGUE_STARTER_OUTS)


def _sp_discipline_rates(
    tendency_row: pd.Series | None,
    pitcher_team_id: str,
    team_pitching: pd.DataFrame | None,
) -> tuple[float, float, float | None]:
    if tendency_row is not None:
        if "sp_k_pct" in tendency_row.index and pd.notna(tendency_row.get("sp_k_pct")):
            k_pct = float(tendency_row["sp_k_pct"])
            bb_pct = float(tendency_row.get("sp_bb_pct", LEAGUE_AVG["bb_pct"]))
            avg_bf = (
                float(tendency_row["avg_bf_per_start"])
                if "avg_bf_per_start" in tendency_row.index
                and pd.notna(tendency_row.get("avg_bf_per_start"))
                else None
            )
            return k_pct, bb_pct, avg_bf
    if team_pitching is not None and not team_pitching.empty:
        sp_rows = team_pitching[
            (team_pitching["team_id"] == pitcher_team_id)
            & (team_pitching["role"] == "sp")
        ]
        if not sp_rows.empty:
            row = sp_rows.iloc[0]
            return float(row["k_pct"]), float(row["bb_pct"]), None
    return LEAGUE_AVG["k_pct"], LEAGUE_AVG["bb_pct"], None


def _project_pitch_count(
    proj_outs: float,
    proj_bf: float,
    sp_k_pct: float,
    sp_bb_pct: float,
    avg_outs_last5: float,
    gs: float,
) -> tuple[float, float, float, float, float]:
    proj_k = proj_bf * sp_k_pct
    proj_bb = proj_bf * sp_bb_pct
    contact_outs = max(proj_outs - proj_k, 0.0)
    raw_pitches = (
        contact_outs * LEAGUE_PITCHES_PER_OUT
        + proj_bb * PITCHES_PER_WALK
        + proj_k * PITCHES_PER_STRIKEOUT
    )
    proj_pitches = _apply_hook_floor(raw_pitches, proj_outs, avg_outs_last5, gs)
    return proj_bf, proj_k, proj_bb, contact_outs, proj_pitches


def project_pitcher_outs_and_pitches(
    games: pd.DataFrame,
    pitcher_tendencies: pd.DataFrame,
    projected_batters: pd.DataFrame | None = None,
    team_pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Project starting-pitcher outs and pitch counts.

    Outs use manager tendency (avg_outs_last5) adjusted by pitch efficiency vs
    league average (5.3 pitches/out). Pitch count uses mutually exclusive outcome
    buckets: contact outs, walks, and strikeouts derived from projected BF.
    """
    rows: list[dict[str, object]] = []

    for _, game in games.iterrows():
        game_id = game["game_id"]
        for side, sp_col in [("home", "sp_home_id"), ("away", "sp_away_id")]:
            sp_id = str(game[sp_col]).strip()
            if sp_id.endswith(".0"):
                sp_id = sp_id[:-2]
            team_id = str(game[f"{side}_team_id"])
            tend_ids = pitcher_tendencies["pitcher_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            tend = pitcher_tendencies[tend_ids == sp_id]
            tendency_row = tend.iloc[0] if not tend.empty else None

            if tendency_row is None:
                logger.warning("No tendency data for SP %s; defaulting to league starter", sp_id)
                avg_outs = FALLBACK_STARTER_OUTS
                efficiency = LEAGUE_PITCHES_PER_OUT
                gs = 1.0
                pitcher_name = sp_id
            else:
                gs = (
                    float(tendency_row["gs"])
                    if "gs" in tendency_row.index and pd.notna(tendency_row.get("gs"))
                    else 1.0
                )
                raw_avg_outs = float(tendency_row["avg_outs_last5"])
                avg_outs = _sanitize_avg_outs(raw_avg_outs, gs)
                efficiency = _clamp_pitch_efficiency(
                    float(tendency_row["pitch_efficiency"])
                )
                pitcher_name = (
                    str(tendency_row["pitcher_name"])
                    if "pitcher_name" in tendency_row.index
                    and pd.notna(tendency_row.get("pitcher_name"))
                    else sp_id
                )

            raw_for_proj = raw_avg_outs if tendency_row is not None else avg_outs
            proj_outs = _compute_proj_outs(raw_for_proj, efficiency, gs)

            opp_pa = _opposing_team_pa(game_id, team_id, game, projected_batters)
            sp_k_pct, sp_bb_pct, avg_bf_per_start = _sp_discipline_rates(
                tendency_row, team_id, team_pitching
            )
            proj_bf = _resolve_proj_bf(proj_outs, avg_outs, avg_bf_per_start, opp_pa)
            proj_bf, proj_k, proj_bb, contact_outs, proj_pitches = _project_pitch_count(
                proj_outs, proj_bf, sp_k_pct, sp_bb_pct, avg_outs, gs
            )

            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "pitcher_id": sp_id,
                    "pitcher_name": pitcher_name,
                    "proj_outs": round(proj_outs, 2),
                    "proj_bf": round(proj_bf, 2),
                    "proj_k": round(proj_k, 2),
                    "proj_bb": round(proj_bb, 2),
                    "contact_outs": round(contact_outs, 2),
                    "proj_pitch_count": round(proj_pitches, 1),
                }
            )

    result = pd.DataFrame(rows)
    result = _apply_final_outs_guardrails(result)
    logger.info("Projected outs and pitch counts for %d starting pitchers", len(result))
    return result
