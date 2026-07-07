from __future__ import annotations

import pandas as pd

from baseball_props.analysis.parlay_builder import PropLeg, build_diversified_tickets


def _sample_legs() -> list[PropLeg]:
    return [
        PropLeg("P1", "Player One", "G1", "batter_hits", 1.5, "Over", 10.0),
        PropLeg("P2", "Player Two", "G2", "batter_hits", 1.5, "Over", 9.0),
        PropLeg("P3", "Player Three", "G3", "batter_hits", 1.5, "Over", 8.0),
        PropLeg("P4", "Player Four", "G4", "batter_hits", 1.5, "Over", 7.0),
        PropLeg("P5", "Player Five", "G5", "batter_hits", 1.5, "Over", 6.0),
        PropLeg("P6", "Player Six", "G6", "batter_hits", 1.5, "Over", 5.0),
    ]


def test_same_player_leg_max_two_of_three_tickets() -> None:
    legs = _sample_legs()
    tickets = build_diversified_tickets(
        legs,
        ticket_count=3,
        legs_per_ticket=2,
        max_player_exposure=2,
    )
    assert len(tickets) >= 1
    p1_count = sum(1 for ticket in tickets for leg in ticket.legs if leg.player_id == "P1")
    assert p1_count <= 2
    assert p1_count < len(tickets)


def test_tickets_prefer_different_games() -> None:
    legs = _sample_legs()
    tickets = build_diversified_tickets(
        legs,
        ticket_count=3,
        legs_per_ticket=2,
        max_player_exposure=2,
    )
    for ticket in tickets:
        games = [leg.game_id for leg in ticket.legs]
        assert len(games) == len(set(games))


def test_build_from_batter_sheet_filters_pass_verdict() -> None:
    sheet = pd.DataFrame(
        [
            {
                "player_id": "P1",
                "player_name": "Play Player",
                "game_id": "G1",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 8.0,
                "verdict": "Play",
            },
            {
                "player_id": "P2",
                "player_name": "Pass Player",
                "game_id": "G2",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Pass",
                "edge_pct": 8.0,
                "verdict": "Pass",
            },
        ]
    )
    tickets = build_diversified_tickets(sheet, ticket_count=3, legs_per_ticket=1)
    all_players = {leg.player_id for ticket in tickets for leg in ticket.legs}
    assert "P1" in all_players or not tickets
    assert "P2" not in all_players
