from baseball_props.analysis.game_splits import build_game_split_export
from baseball_props.pipeline.slate_run import run_slate


def test_game_split_export_shape() -> None:
    result = run_slate(source="mock")
    games = build_game_split_export(result, include_pitch_locations=False)
    assert len(games) == 2
    game = games[0]
    assert "away_lineup" in game
    assert "home_lineup" in game
    assert len(game["away_lineup"]) == 9
    assert len(game["home_lineup"]) == 9

    batter = game["away_lineup"][0]
    assert "player_id" in batter
    assert "vs_lhp_woba" in batter
    assert "vs_rhp_woba" in batter
    assert "sp_vs_hand_woba" in batter
    assert "pitch_locations" in batter
    assert "ev_per_unit" in batter
    assert "confidence_tier" in batter
    assert "fractional_kelly_pct" in batter

    assert "vs_lhb" in game["away_sp"]
    assert "vs_rhb" in game["away_sp"]
    assert "pitch_locations" in game["away_sp"]
    assert "ev_per_unit" in game["away_sp"]

    assert "umpire" in game
    assert "travel_rest" in game
    assert "data_health_warnings" in game
