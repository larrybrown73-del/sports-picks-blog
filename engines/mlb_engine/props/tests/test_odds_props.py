import pandas as pd

from baseball_props.data.odds_props import (
    _parse_prop_payload,
    consolidated_market_lines,
    consolidated_prop_quotes,
)


def _mixed_market_payload() -> dict:
    return {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Aaron Judge",
                                "price": -110,
                                "point": 1.5,
                            },
                            {
                                "name": "Under",
                                "description": "Aaron Judge",
                                "price": -110,
                                "point": 1.5,
                            },
                            {
                                "name": "Over",
                                "price": -110,
                                "point": 1.5,
                            },
                        ],
                    },
                    {
                        "key": "team_totals",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "New York Yankees",
                                "price": -110,
                                "point": 4.5,
                            },
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {
                                "name": "Over",
                                "price": -110,
                                "point": 8.5,
                            },
                        ],
                    },
                ],
            }
        ]
    }


def test_parse_prop_payload_skips_game_and_team_totals() -> None:
    event_id = "abc123event456789012345678901234"
    allowed = {"batter_total_bases", "pitcher_outs"}
    df = _parse_prop_payload(_mixed_market_payload(), event_id, allowed)

    assert len(df) == 2
    assert set(df["side"]) == {"Over", "Under"}
    assert (df["player_name"] == "Aaron Judge").all()
    assert (df["line"] == 1.5).all()
    assert (df["market"] == "batter_total_bases").all()


def test_parse_prop_payload_skips_missing_description() -> None:
    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "outcomes": [{"name": "Over", "price": -110, "point": 1.5}],
                    }
                ],
            }
        ]
    }
    df = _parse_prop_payload(payload, "abc123event456789012345678901234", {"batter_total_bases"})
    assert df.empty


def test_parse_prop_payload_skips_implausible_tb_line() -> None:
    payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Aaron Judge",
                                "price": -110,
                                "point": 4.5,
                            },
                        ],
                    }
                ],
            }
        ]
    }
    df = _parse_prop_payload(payload, "abc123event456789012345678901234", {"batter_total_bases"})
    assert df.empty


def test_consolidated_market_lines_median_across_books() -> None:
    props = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_total_bases",
                "side": "Over",
                "line": 1.5,
                "bookmaker": "draftkings",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_total_bases",
                "side": "Under",
                "line": 1.5,
                "bookmaker": "draftkings",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_total_bases",
                "side": "Over",
                "line": 2.5,
                "bookmaker": "fanduel",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_total_bases",
                "side": "Under",
                "line": 2.5,
                "bookmaker": "fanduel",
            },
        ]
    )
    lines = consolidated_market_lines(props)
    assert len(lines) == 1
    assert lines.iloc[0]["market_line"] == 2.0


def test_parse_event_payload_skips_team_totals_in_single_event() -> None:
    event_id = "abc123event456789012345678901234"
    allowed = {"batter_total_bases", "pitcher_outs"}
    df = _parse_prop_payload(_mixed_market_payload(), event_id, allowed)

    assert len(df) == 2
    assert (df["line"] == 1.5).all()
    assert (df["market"] == "batter_total_bases").all()
