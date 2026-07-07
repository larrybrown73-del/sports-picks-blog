from __future__ import annotations

import pandas as pd
import pytest

from baseball_props.data.statcast_feed import (
    _fetch_platoon_splits,
    _pitcher_bundle_from_rates,
    _rates_from_statcast_frame,
    _regressed_platoon_split,
    _resolve_pitcher_name,
    build_pitcher_tendencies,
)
from baseball_props.analysis.pitcher_projection import project_pitcher_outs_and_pitches
from baseball_props.config import (
    FALLBACK_RELIEF_OUTS,
    FALLBACK_STARTER_OUTS,
    LEAGUE_PITCHES_PER_OUT,
    MAX_PROJ_OUTS,
)


def test_rates_from_statcast_frame_computes_discipline() -> None:
    sc = pd.DataFrame(
        {
            "events": ["single", "strikeout", "walk", "home_run"],
            "woba_value": [0.9, 0.0, 0.7, 2.0],
            "launch_speed": [98.0, None, 88.0, 101.0],
        }
    )
    rates = _rates_from_statcast_frame(sc)
    assert rates["k_pct"] == 0.25
    assert rates["bb_pct"] == 0.25
    assert rates["woba"] > 0
    assert rates["hard_hit_pct"] > 0


def test_rates_from_statcast_frame_ignores_non_terminal_pitches() -> None:
    """Pitch-level rows without terminal events must not dilute PA rates."""
    sc = pd.DataFrame(
        {
            "events": ["", "", "single", "", "home_run"],
            "woba_value": [0.0, 0.0, 0.9, 0.0, 2.0],
            "launch_speed": [90.0, 91.0, 98.0, 92.0, 101.0],
        }
    )
    rates = _rates_from_statcast_frame(sc)
    assert rates["woba"] == pytest.approx(1.45)
    assert rates["k_pct"] == 0.0


def test_platoon_splits_differ_by_hand() -> None:
    season = {
        "season_woba": 0.320,
        "season_iso": 0.180,
        "season_k_pct": 0.220,
        "season_bb_pct": 0.080,
        "season_wrc_plus": 110.0,
        "season_hard_hit_pct": 0.420,
    }
    rows = _fetch_platoon_splits("999001", season, fg_id=None)
    by_split = {row["split"]: row for row in rows}
    assert by_split["vs_lhp"]["woba"] != by_split["vs_rhp"]["woba"]
    assert by_split["vs_lhp"]["hard_hit_pct"] != by_split["vs_rhp"]["hard_hit_pct"]


def test_regressed_platoon_not_flat_league() -> None:
    season = {
        "season_woba": 0.340,
        "season_iso": 0.200,
        "season_k_pct": 0.210,
        "season_bb_pct": 0.090,
        "season_wrc_plus": 125.0,
        "season_hard_hit_pct": 0.450,
    }
    lhp = _regressed_platoon_split(season, "vs_lhp")
    rhp = _regressed_platoon_split(season, "vs_rhp")
    assert lhp["hard_hit_pct"] > rhp["hard_hit_pct"]


def test_resolve_pitcher_name_prefers_schedule_map() -> None:
    names = {"680694": "Carlos Rodon"}
    assert _resolve_pitcher_name("680694", names, "Rodon, Carlos") == "Carlos Rodon"
    assert _resolve_pitcher_name("680694", {}, "Carlos Rodon") == "Carlos Rodon"


def test_pitcher_bundle_caps_implausible_season_ip() -> None:
    bundle = _pitcher_bundle_from_rates(gs=1, ip=30.0, tbf=120, so=80, bb=20, pitches=400)
    assert bundle["avg_outs_last5"] == pytest.approx(FALLBACK_STARTER_OUTS)
    assert bundle["avg_outs_last5"] <= MAX_PROJ_OUTS


def test_build_pitcher_tendencies_uses_defaults_when_unresolved() -> None:
    slate = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "home_team_id": "NYM",
                "away_team_id": "PHI",
                "sp_home_id": "999888",
                "sp_away_id": "999777",
            }
        ]
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "baseball_props.data.statcast_feed._resolve_pitcher_stats_bundle",
            lambda *args, **kwargs: ({}, "unresolved", None),
        )
        tendencies = build_pitcher_tendencies(slate)
    assert tendencies.loc[0, "avg_outs_last5"] == pytest.approx(FALLBACK_STARTER_OUTS)


def test_fangraphs_403_corrupted_mlb_stats_caps_proj_outs() -> None:
    slate = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "home_team_id": "NYM",
                "away_team_id": "PHI",
                "sp_home_id": "111111",
                "sp_away_id": "222222",
            }
        ]
    )
    corrupted_mlb = {
        "gs": 1.0,
        "ip": 30.66,
        "tbf": 100.0,
        "so": 50.0,
        "bb": 10.0,
        "pitches": 400.0,
    }

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "baseball_props.data.statcast_feed._get_pitching_stats_frames",
            lambda year: (pd.DataFrame(), pd.DataFrame()),
        )
        mp.setattr(
            "baseball_props.data.statcast_feed._fetch_mlb_pitcher_season_stats",
            lambda sp_id, season: corrupted_mlb if sp_id in {"111111", "222222"} else None,
        )
        tendencies = build_pitcher_tendencies(slate)
        result = project_pitcher_outs_and_pitches(slate, tendencies)

    assert (tendencies["avg_outs_last5"] <= MAX_PROJ_OUTS).all()
    assert (result["proj_outs"] <= MAX_PROJ_OUTS).all()
    assert tendencies["pitch_efficiency"].iloc[0] == pytest.approx(LEAGUE_PITCHES_PER_OUT) or (
        tendencies["pitch_efficiency"].between(3.5, 8.0).all()
    )
