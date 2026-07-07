from __future__ import annotations

import re

import pandas as pd

from baseball_props.data.odds_props import is_plausible_prop_line
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


def normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[^a-z\s]", "", n)
    n = re.sub(r"\s(jr|sr|ii|iii|iv)$", "", n).strip()
    return n


def match_name(model_name: str, market_names: pd.Series) -> str | None:
    """Exact normalized name match only (no substring fallback)."""
    target = normalize_name(model_name)
    for candidate in market_names.dropna().unique():
        if normalize_name(str(candidate)) == target:
            return str(candidate)
    return None


def filter_plausible_market_lines(market_df: pd.DataFrame) -> pd.DataFrame:
    if market_df.empty:
        return market_df
    mask = market_df.apply(
        lambda row: is_plausible_prop_line(str(row["market"]), float(row["market_line"])),
        axis=1,
    )
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(
            "Filtered %d implausible prop market lines (likely game/team totals)",
            dropped,
        )
    return market_df.loc[mask].reset_index(drop=True)
