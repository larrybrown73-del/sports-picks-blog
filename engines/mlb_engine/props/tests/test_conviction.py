import pandas as pd



from baseball_props.analysis.conviction import compute_conviction_plays





def test_conviction_uses_player_tb_line_not_team_total() -> None:

    game_id = "abc123event456789012345678901234"

    prop_lines = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "market": "batter_hits",

                "side": "Over",

                "line": 1.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "market": "batter_hits",

                "side": "Under",

                "line": 1.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

        ]

    )

    projected = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "proj_hits": 1.9,

                "proj_woba": 0.400,

                "proj_pa": 4.5,

            }

        ]

    )

    pitcher_outs = pd.DataFrame(columns=["game_id", "pitcher_name", "proj_outs"])



    result = compute_conviction_plays(projected, pitcher_outs, prop_lines, top_n=3)



    assert len(result) == 1

    assert result.iloc[0]["market_line"] == 1.5

    assert result.iloc[0]["market"] == "batter_hits"

    assert result.iloc[0]["market_line"] != 4.5

    assert "edge_pct" in result.columns





def test_conviction_exact_name_match_only() -> None:

    game_id = "abc123event456789012345678901234"

    prop_lines = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Mike Trout",

                "market": "batter_hits",

                "side": "Over",

                "line": 1.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

            {

                "game_id": game_id,

                "player_name": "Mike Trout",

                "market": "batter_hits",

                "side": "Under",

                "line": 1.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

        ]

    )

    projected = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Mike",

                "proj_woba": 0.350,

                "proj_pa": 4.0,

            }

        ]

    )

    pitcher_outs = pd.DataFrame(columns=["game_id", "pitcher_name", "proj_outs"])



    result = compute_conviction_plays(projected, pitcher_outs, prop_lines, top_n=3)

    assert result.empty





def test_conviction_rejects_leaked_game_total_lines() -> None:

    game_id = "abc123event456789012345678901234"

    prop_lines = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "market": "batter_hits",

                "side": "Over",

                "line": 4.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "market": "batter_hits",

                "side": "Under",

                "line": 4.5,

                "odds": -110,

                "bookmaker": "draftkings",

            },

        ]

    )

    projected = pd.DataFrame(

        [

            {

                "game_id": game_id,

                "player_name": "Aaron Judge",

                "proj_woba": 0.400,

                "proj_pa": 4.5,

            }

        ]

    )

    pitcher_outs = pd.DataFrame(columns=["game_id", "pitcher_name", "proj_outs"])



    result = compute_conviction_plays(projected, pitcher_outs, prop_lines, top_n=3)

    assert result.empty

