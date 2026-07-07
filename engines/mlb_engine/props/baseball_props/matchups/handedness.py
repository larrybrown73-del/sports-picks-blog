from __future__ import annotations

import pandas as pd

from baseball_props.types import PitcherHand, PitcherSplitKey

BATTER_TO_PITCHER_SPLIT: dict[str, PitcherSplitKey] = {
    "L": "vs_lhb",
    "R": "vs_rhb",
}


def resolve_active_batter_hand(
    bat_hand: str | None,
    opp_sp_hand: PitcherHand | str,
) -> str:
    """Resolve switch hitters to active plate side vs opposing SP hand."""
    hand = str(bat_hand or "R").upper()
    sp = str(opp_sp_hand).upper()
    if hand == "S":
        return "R" if sp == "L" else "L"
    if hand in {"L", "R"}:
        return hand
    return "R"


def batter_hand_to_pitcher_split(batter_hand: str) -> PitcherSplitKey:
    """Map active batter hand to pitcher platoon split key."""
    active = str(batter_hand).upper()
    return BATTER_TO_PITCHER_SPLIT.get(active, "vs_rhb")


def resolve_active_batter_hand_series(
    bat_hand: pd.Series,
    opp_sp_hand: pd.Series,
) -> pd.Series:
    """Vectorized active batter hand resolution."""
    bat = bat_hand.fillna("R").astype(str).str.upper()
    sp = opp_sp_hand.fillna("R").astype(str).str.upper()
    switch = bat.eq("S")
    active = bat.where(~switch, sp.map({"L": "R", "R": "L"}).fillna("R"))
    return active.where(active.isin(["L", "R"]), "R")
