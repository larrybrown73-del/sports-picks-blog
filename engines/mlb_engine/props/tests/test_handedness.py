from baseball_props.matchups.handedness import (
    batter_hand_to_pitcher_split,
    resolve_active_batter_hand,
    resolve_active_batter_hand_series,
)
import pandas as pd


def test_switch_hitter_bats_right_vs_lefty() -> None:
    assert resolve_active_batter_hand("S", "L") == "R"


def test_switch_hitter_bats_left_vs_righty() -> None:
    assert resolve_active_batter_hand("S", "R") == "L"


def test_batter_hand_to_pitcher_split() -> None:
    assert batter_hand_to_pitcher_split("L") == "vs_lhb"
    assert batter_hand_to_pitcher_split("R") == "vs_rhb"


def test_resolve_active_batter_hand_series() -> None:
    df = pd.DataFrame(
        {
            "bat_hand": ["S", "R", "L"],
            "opp_sp_hand": ["L", "R", "L"],
        }
    )
    active = resolve_active_batter_hand_series(df["bat_hand"], df["opp_sp_hand"])
    assert active.tolist() == ["R", "R", "L"]
