from __future__ import annotations

import pandas as pd

from baseball_props.config import LEAGUE_AVG
from baseball_props.types import SlateFrames

# NYY H001-H009, BOS H010-H018, LAD H019-H027, SF H028-H036
# Names reflect approximate 2026 rosters (mock is illustrative, not live lineups).
HITTER_NAMES: dict[str, str] = {
    "H001": "Aaron Judge",
    "H002": "Juan Soto",
    "H003": "Anthony Volpe",
    "H004": "Giancarlo Stanton",
    "H005": "Jazz Chisholm Jr.",
    "H006": "Ben Rice",
    "H007": "Austin Wells",
    "H008": "Trent Grisham",
    "H009": "Jasson Dominguez",
    "H010": "Triston Casas",
    "H011": "Trevor Story",
    "H012": "Tyler O'Neill",
    "H013": "Jarren Duran",
    "H014": "Masataka Yoshida",
    "H015": "Connor Wong",
    "H016": "Ceddanne Rafaela",
    "H017": "Wilyer Abreu",
    "H018": "David Hamilton",
    "H019": "Shohei Ohtani",
    "H020": "Mookie Betts",
    "H021": "Freddie Freeman",
    "H022": "Teoscar Hernandez",
    "H023": "Will Smith",
    "H024": "Gavin Lux",
    "H025": "Tommy Edman",
    "H026": "Andy Pages",
    "H027": "Miguel Rojas",
    "H028": "Heliot Ramos",
    "H029": "Wilmer Flores",
    "H030": "Matt Chapman",
    "H031": "Michael Conforto",
    "H032": "Patrick Bailey",
    "H033": "Jung Hoo Lee",
    "H034": "LaMonte Wade Jr.",
    "H035": "Casey Schmitt",
    "H036": "Tyler Fitzgerald",
}

PITCHER_NAMES: dict[str, str] = {
    "P101": "Brayan Bello",
    "P102": "Carlos Rodon",
    "P103": "Logan Webb",
    "P104": "Yoshinobu Yamamoto",
}

# Illustrative bat sides for mock lineups (L/R/S).
MOCK_BAT_HAND: dict[str, str] = {
    "H019": "S",
    "H025": "S",
    "H002": "L",
    "H014": "L",
    "H034": "L",
}


def build_mock_slate() -> SlateFrames:
    """Return mock DataFrames for a 2-game slate with intentional data gaps."""
    slate_games = pd.DataFrame(
        [
            {
                "game_id": "G001",
                "game_date": "2026-06-30",
                "home_team_id": "BOS",
                "away_team_id": "NYY",
                "park_id": "FEN",
                "sp_home_id": "P101",
                "sp_away_id": "P102",
                "sp_home_hand": "L",
                "sp_away_hand": "R",
                "mlb_game_pk": 777001,
            },
            {
                "game_id": "G002",
                "game_date": "2026-06-30",
                "home_team_id": "SF",
                "away_team_id": "LAD",
                "park_id": "ORC",
                "sp_home_id": "P103",
                "sp_away_id": "P104",
                "sp_home_hand": "R",
                "sp_away_hand": "L",
                "mlb_game_pk": 777002,
            },
        ]
    )

    # 18 hitters across two teams per game (9 per side)
    hitter_ids = [f"H{i:03d}" for i in range(1, 37)]
    lineups_rows: list[dict[str, object]] = []
    lineup_map = {
        "G001": [("NYY", hitter_ids[0:9]), ("BOS", hitter_ids[9:18])],
        "G002": [("LAD", hitter_ids[18:27]), ("SF", hitter_ids[27:36])],
    }
    for game_id, teams in lineup_map.items():
        for team_id, players in teams:
            for slot, player_id in enumerate(players, start=1):
                lineups_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team_id,
                        "lineup_slot": slot,
                        "player_id": player_id,
                        "player_name": HITTER_NAMES[player_id],
                        "bat_hand": MOCK_BAT_HAND.get(
                            player_id, "R" if int(player_id[1:]) % 2 else "L"
                        ),
                    }
                )

    lineups = pd.DataFrame(lineups_rows)

    # Baselines: H005 missing roll14_woba + roll14_hard_hit_pct; H012 missing roll14+roll30 k_pct
    baseline_rows: list[dict[str, object]] = []
    for i, pid in enumerate(hitter_ids):
        baseline_rows.append(
            {
                "player_id": pid,
                "season_woba": 0.310 + (i % 5) * 0.015,
                "roll14_woba": None if pid == "H005" else 0.305 + (i % 4) * 0.018,
                "roll30_woba": 0.308 + (i % 3) * 0.012,
                "season_iso": 0.140 + (i % 6) * 0.010,
                "roll14_iso": 0.135 + (i % 4) * 0.012,
                "roll30_iso": 0.138 + (i % 5) * 0.009,
                "season_k_pct": 0.200 + (i % 7) * 0.015,
                "roll14_k_pct": None if pid == "H012" else 0.195 + (i % 5) * 0.014,
                "roll30_k_pct": None if pid == "H012" else 0.198 + (i % 4) * 0.013,
                "season_bb_pct": 0.070 + (i % 4) * 0.008,
                "roll14_bb_pct": 0.068 + (i % 3) * 0.007,
                "roll30_bb_pct": 0.069 + (i % 5) * 0.006,
                "season_wrc_plus": 90 + (i % 8) * 5,
                "roll14_wrc_plus": 92 + (i % 7) * 4,
                "roll30_wrc_plus": 91 + (i % 6) * 4,
                "season_hard_hit_pct": 0.34 + (i % 6) * 0.02,
                "roll14_hard_hit_pct": None if pid == "H005" else 0.35 + (i % 5) * 0.018,
                "roll30_hard_hit_pct": 0.36 + (i % 4) * 0.015,
                "season_pa": 80 + (i % 10) * 45 if pid == "H005" else 320 + (i % 8) * 30,
            }
        )

    player_baselines = pd.DataFrame(baseline_rows)

    # Splits: H020 missing vs_rhp split entirely
    split_rows: list[dict[str, object]] = []
    for pid in hitter_ids:
        idx = int(pid[1:])
        if pid == "H020":
            split_rows.append(
                {
                    "player_id": pid,
                    "split": "vs_lhp",
                    "woba": 0.340,
                    "iso": 0.180,
                    "k_pct": 0.210,
                    "bb_pct": 0.095,
                    "wrc_plus": 128,
                    "hard_hit_pct": 0.48,
                }
            )
            continue
        for split in ("vs_lhp", "vs_rhp"):
            bump = 0.020 if split == "vs_lhp" else -0.010
            wrc_bump = 8 if split == "vs_lhp" else -5
            hh_bump = 0.03 if split == "vs_lhp" else -0.02
            split_rows.append(
                {
                    "player_id": pid,
                    "split": split,
                    "woba": 0.310 + bump + (idx % 5) * 0.005,
                    "iso": 0.145 + bump,
                    "k_pct": 0.220 - bump / 2,
                    "bb_pct": 0.085 + bump / 3,
                    "wrc_plus": 100 + wrc_bump + (idx % 5) * 3,
                    "hard_hit_pct": 0.38 + hh_bump + (idx % 4) * 0.01,
                }
            )

    matchup_splits = pd.DataFrame(split_rows)

    pitcher_platoon_rows: list[dict[str, object]] = []
    for pid, name in PITCHER_NAMES.items():
        for split, bump in (("vs_lhb", -0.012), ("vs_rhb", 0.010)):
            pitcher_platoon_rows.append(
                {
                    "pitcher_id": pid,
                    "split": split,
                    "woba_allowed": LEAGUE_AVG["woba"] + bump,
                    "iso_allowed": 0.145 + bump,
                    "k_pct": 0.240 - bump / 2,
                    "bb_pct": 0.075 + bump / 3,
                    "bf": 120.0,
                }
            )
    pitcher_platoon_splits = pd.DataFrame(pitcher_platoon_rows)

    team_pitching = pd.DataFrame(
        [
            {"team_id": "NYY", "role": "sp", "woba_allowed": 0.305, "iso_allowed": 0.138, "k_pct": 0.235, "bb_pct": 0.078},
            {"team_id": "NYY", "role": "bullpen", "woba_allowed": 0.318, "iso_allowed": 0.152, "k_pct": 0.245, "bb_pct": 0.092},
            {"team_id": "BOS", "role": "sp", "woba_allowed": 0.328, "iso_allowed": 0.160, "k_pct": 0.210, "bb_pct": 0.088},
            {"team_id": "BOS", "role": "bullpen", "woba_allowed": 0.322, "iso_allowed": 0.148, "k_pct": 0.228, "bb_pct": 0.085},
            {"team_id": "LAD", "role": "sp", "woba_allowed": 0.298, "iso_allowed": 0.130, "k_pct": 0.250, "bb_pct": 0.072},
            {"team_id": "LAD", "role": "bullpen", "woba_allowed": 0.310, "iso_allowed": 0.142, "k_pct": 0.238, "bb_pct": 0.080},
            {"team_id": "SF", "role": "sp", "woba_allowed": 0.315, "iso_allowed": 0.145, "k_pct": 0.225, "bb_pct": 0.082},
            {"team_id": "SF", "role": "bullpen", "woba_allowed": 0.320, "iso_allowed": 0.150, "k_pct": 0.230, "bb_pct": 0.086},
        ]
    )

    park_weather = pd.DataFrame(
        [
            {"park_id": "FEN", "park_factor_runs": 1.05, "park_factor_hr": 1.08, "temp_f": 78.0, "wind_mph": 12.0, "wind_dir": "out_to_lf"},
            {"park_id": "ORC", "park_factor_runs": 0.92, "park_factor_hr": 0.88, "temp_f": 62.0, "wind_mph": 8.0, "wind_dir": "in_from_cf"},
        ]
    )

    vegas_totals = pd.DataFrame(
        [
            {"game_id": "G001", "home_implied_runs": 4.6, "away_implied_runs": 4.2, "game_total": 8.8},
            {"game_id": "G002", "home_implied_runs": 3.8, "away_implied_runs": 4.5, "game_total": 8.3},
        ]
    )

    pitcher_tendencies = pd.DataFrame(
        [
            {"pitcher_id": "P101", "pitcher_name": PITCHER_NAMES["P101"], "avg_outs_last5": 16.2, "pitch_efficiency": 5.15, "gs": 14, "is_true_starter": True, "sp_k_pct": 0.210, "sp_bb_pct": 0.088, "avg_bf_per_start": 23.5},
            {"pitcher_id": "P102", "pitcher_name": PITCHER_NAMES["P102"], "avg_outs_last5": 17.8, "pitch_efficiency": 5.05, "gs": 16, "is_true_starter": True, "sp_k_pct": 0.265, "sp_bb_pct": 0.072, "avg_bf_per_start": 26.2},
            {"pitcher_id": "P103", "pitcher_name": PITCHER_NAMES["P103"], "avg_outs_last5": 15.5, "pitch_efficiency": 5.25, "gs": 15, "is_true_starter": True, "sp_k_pct": 0.228, "sp_bb_pct": 0.082, "avg_bf_per_start": 24.0},
            {"pitcher_id": "P104", "pitcher_name": PITCHER_NAMES["P104"], "avg_outs_last5": 18.1, "pitch_efficiency": 5.00, "gs": 17, "is_true_starter": True, "sp_k_pct": 0.278, "sp_bb_pct": 0.065, "avg_bf_per_start": 27.5},
        ]
    )

    return SlateFrames(
        slate_games=slate_games,
        player_baselines=player_baselines,
        matchup_splits=matchup_splits,
        pitcher_platoon_splits=pitcher_platoon_splits,
        team_pitching=team_pitching,
        park_weather=park_weather,
        vegas_totals=vegas_totals,
        lineups=lineups,
        pitcher_tendencies=pitcher_tendencies,
    )
