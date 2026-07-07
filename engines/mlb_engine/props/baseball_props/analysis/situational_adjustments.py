from __future__ import annotations

import pandas as pd


def apply_travel_rest_to_rates(
    df: pd.DataFrame,
    multiplier_col: str = "travel_rest_multiplier",
) -> pd.DataFrame:
    """Apply travel/rest multiplier to rate projections (placeholder: 1.0)."""
    out = df.copy()
    if multiplier_col not in out.columns:
        return out
    mult = out[multiplier_col].fillna(1.0)
    for col in ("proj_woba", "proj_iso"):
        if col in out.columns:
            out[col] = out[col] * mult
    return out


def apply_umpire_to_runs(
    df: pd.DataFrame,
    modifier_col: str = "run_environment_modifier",
) -> pd.DataFrame:
    """Apply umpire run-environment modifier (placeholder: 1.0)."""
    out = df.copy()
    if modifier_col not in out.columns:
        return out
    mult = out[modifier_col].fillna(1.0)
    if "park_factor_runs" in out.columns:
        out["park_factor_runs"] = out["park_factor_runs"] * mult
    return out


def apply_lineup_absence_penalty(
    df: pd.DataFrame,
    penalty_col: str = "offensive_penalty",
) -> pd.DataFrame:
    """Apply lineup absence offensive penalty (placeholder: 1.0)."""
    out = df.copy()
    if penalty_col not in out.columns:
        return out
    mult = out[penalty_col].fillna(1.0)
    for col in ("proj_woba", "proj_iso", "proj_total_bases", "proj_hits"):
        if col in out.columns:
            out[col] = out[col] * mult
    return out
