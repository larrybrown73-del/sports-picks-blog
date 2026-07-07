import pandas as pd

from baseball_props.analysis.game_splits import build_game_split_export
from baseball_props.data.statcast_pitches import (
    PITCH_TYPE_NAMES,
    clean_pitch_locations,
    pitch_locations_to_records,
    pitch_type_full_name,
)
from baseball_props.pipeline.slate_run import run_slate


def test_pitch_type_full_name() -> None:
    assert pitch_type_full_name("FF") == "4-SEAM FASTBALL"
    assert pitch_type_full_name("ST") == "SWEEPER"
    assert pitch_type_full_name("XYZ") == "XYZ"


def test_pitch_type_names_contains_common_codes() -> None:
    assert PITCH_TYPE_NAMES["SI"] == "SINKER"
    assert PITCH_TYPE_NAMES["CH"] == "CHANGEUP"


def test_clean_pitch_locations_drops_missing() -> None:
    df = pd.DataFrame(
        {
            "pitch_type": ["FF", "SL", None, "CH"],
            "plate_x": [-0.5, 0.2, 0.1, None],
            "plate_z": [2.5, 3.0, 2.0, 2.2],
        }
    )
    cleaned = clean_pitch_locations(df)
    assert len(cleaned) == 2
    assert "pitch_type_name" in cleaned.columns
    assert cleaned.iloc[0]["pitch_type_name"] == "4-SEAM FASTBALL"


def test_clean_pitch_locations_empty_input() -> None:
    assert clean_pitch_locations(pd.DataFrame()).empty
    assert clean_pitch_locations(
        pd.DataFrame({"pitch_type": ["FF"], "other": [1]})
    ).empty


def test_pitch_locations_to_records_rounds() -> None:
    df = pd.DataFrame(
        {
            "pitch_type": ["FF"],
            "plate_x": [-0.4567],
            "plate_z": [2.7891],
        }
    )
    records = pitch_locations_to_records(df)
    assert records == [
        {
            "pitch_type": "FF",
            "pitch_type_name": "4-SEAM FASTBALL",
            "plate_x": -0.457,
            "plate_z": 2.789,
        }
    ]


def test_game_split_export_includes_pitch_locations() -> None:
    result = run_slate(source="mock")
    games = build_game_split_export(result, include_pitch_locations=False)
    game = games[0]
    assert "pitch_locations" in game["away_sp"]
    assert game["away_sp"]["pitch_locations"] == []
    batter = game["away_lineup"][0]
    assert "player_id" in batter
    assert batter["pitch_locations"] == []
