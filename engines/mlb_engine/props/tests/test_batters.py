import pandas as pd

from baseball_props.config import LEAGUE_AVG_IMPLIED_RUNS, LEAGUE_PA_PER_RUN, SLOT_PA_WEIGHTS
from baseball_props.opportunity.batters import project_batter_pa


def test_project_batter_pa_falls_back_when_vegas_missing() -> None:
    lineups = pd.DataFrame(
        [
            {
                "game_id": "822960",
                "team_id": "AZ",
                "lineup_slot": 1,
                "player_id": "P001",
                "player_name": "Test Player",
            }
        ]
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "822960",
                "home_team_id": "TB",
                "away_team_id": "AZ",
            }
        ]
    )
    vegas_totals = pd.DataFrame(
        [
            {
                "game_id": "other-game-id",
                "home_implied_runs": 5.0,
                "away_implied_runs": 4.0,
                "game_total": 9.0,
            }
        ]
    )

    result = project_batter_pa(lineups, vegas_totals, games)
    weight_sum = sum(SLOT_PA_WEIGHTS.values())
    expected_pa = (
        LEAGUE_AVG_IMPLIED_RUNS * LEAGUE_PA_PER_RUN * SLOT_PA_WEIGHTS[1] / weight_sum
    )

    assert len(result) == 1
    assert result.iloc[0]["proj_pa"] == expected_pa
