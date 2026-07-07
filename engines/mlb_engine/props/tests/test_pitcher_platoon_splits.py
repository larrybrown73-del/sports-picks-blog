import pandas as pd

from baseball_props.data.statcast_feed import (
    _pitcher_platoon_from_statcast_frame,
    _regressed_pitcher_platoon_split,
)


def test_pitcher_platoon_statcast_filters_by_batter_stand() -> None:
    sc = pd.DataFrame(
        {
            "events": ["single", "strikeout", "walk", "home_run"],
            "stand": ["L", "L", "R", "R"],
            "woba_value": [0.88, 0.0, 0.7, 1.24],
            "launch_speed": [95, 80, 92, 105],
        }
    )
    platoon = _pitcher_platoon_from_statcast_frame(sc)
    assert "vs_lhb" in platoon
    assert "vs_rhb" in platoon
    assert platoon["vs_lhb"]["bf"] == 2.0
    assert platoon["vs_rhb"]["bf"] == 2.0
    assert platoon["vs_lhb"]["woba"] > 0


def test_regressed_pitcher_platoon_split_differs_by_hand() -> None:
    lhb = _regressed_pitcher_platoon_split(0.320, "vs_lhb")
    rhb = _regressed_pitcher_platoon_split(0.320, "vs_rhb")
    assert lhb["woba_allowed"] != rhb["woba_allowed"]
