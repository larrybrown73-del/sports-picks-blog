from baseball_props.data.mlb_live import reconcile_lineup_with_active_roster


def test_reconcile_lineup_drops_inactive_players() -> None:
    lineup = [
        {"player_id": "1", "player_name": "Active Star", "lineup_slot": 1},
        {"player_id": "2", "player_name": "Departed Vet", "lineup_slot": 2},
        {"player_id": "3", "player_name": "Active Two", "lineup_slot": 3},
    ]
    roster = [
        {"player_id": "1", "player_name": "Active Star", "position": "CF"},
        {"player_id": "3", "player_name": "Active Two", "position": "SS"},
        {"player_id": "4", "player_name": "Callup", "position": "2B"},
        {"player_id": "5", "player_name": "Bench", "position": "LF"},
        {"player_id": "6", "player_name": "Bench2", "position": "RF"},
        {"player_id": "7", "player_name": "Bench3", "position": "C"},
        {"player_id": "8", "player_name": "Bench4", "position": "1B"},
        {"player_id": "9", "player_name": "Bench5", "position": "3B"},
        {"player_id": "10", "player_name": "Bench6", "position": "DH"},
    ]
    reconciled, removed = reconcile_lineup_with_active_roster(
        lineup, team_id=999, roster_hitters=roster
    )
    assert removed == 1
    assert all(row["player_name"] != "Departed Vet" for row in reconciled)
    assert len(reconciled) == 9
    assert reconciled[0]["player_name"] == "Active Star"


def test_reconcile_lineup_keeps_valid_previous_game_order() -> None:
    lineup = [
        {"player_id": str(i), "player_name": f"P{i}", "lineup_slot": i}
        for i in range(1, 10)
    ]
    roster = [
        {"player_id": str(i), "player_name": f"P{i}", "position": "UT"} for i in range(1, 10)
    ]
    reconciled, removed = reconcile_lineup_with_active_roster(
        lineup, team_id=999, roster_hitters=roster
    )
    assert removed == 0
    assert len(reconciled) == 9
    assert [row["player_id"] for row in reconciled] == [str(i) for i in range(1, 10)]
