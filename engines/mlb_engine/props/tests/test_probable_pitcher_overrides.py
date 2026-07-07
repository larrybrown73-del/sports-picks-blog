import pandas as pd

from baseball_props.data.probable_pitcher_overrides import apply_probable_pitcher_overrides


def test_apply_littell_override_for_pit_wsh() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "822716",
                "mlb_game_pk": 822716,
                "home_team_id": "WSH",
                "away_team_id": "PIT",
                "sp_home_id": "687223",
                "sp_home_hand": "R",
                "sp_away_id": "677952",
                "sp_away_hand": "R",
            }
        ]
    )
    names = {"687223": "Carson Palmquist", "677952": "Braxton Ashcraft"}
    updated, updated_names = apply_probable_pitcher_overrides(
        games,
        names,
        slate_date="2026-07-04",
    )
    assert str(updated.iloc[0]["sp_home_id"]) == "641793"
    assert updated_names["641793"] == "Zack Littell"
    assert str(updated.iloc[0]["sp_away_id"]) == "677952"
