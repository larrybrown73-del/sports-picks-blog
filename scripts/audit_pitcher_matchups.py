#!/usr/bin/env python3
"""Audit pitcher matchup flags for a given slate date."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTOR = PROJECT_ROOT / "engines" / "mlb_engine" / "predictor"

for env_path in (PROJECT_ROOT / ".env.local", PREDICTOR / ".env"):
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

sys.path.insert(0, str(PREDICTOR))

from backtest import _memoize_obp_lookups, load_frozen_models  # noqa: E402
import baseball_data  # noqa: E402
import model  # noqa: E402
from baseball_data import fetch_games_for_date, get_starting_pitcher_info  # noqa: E402
from config import DEFAULT_ROLLING_WINDOW  # noqa: E402
from pitcher_matchup import (  # noqa: E402
    apply_pitcher_matchup_adjustments,
    fetch_pitcher_season_profile,
    fetch_team_offense_profile,
    is_ground_ball_pitcher,
    is_patient_lineup,
    is_power_pitcher,
    is_velo_struggler,
    pitcher_runs_allowed_scalar,
)


def main() -> None:
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    print(f"Pitcher matchup audit for {d.isoformat()}\n")

    _memoize_obp_lookups()
    history = baseball_data.games_for_prediction([d.year])
    models = load_frozen_models()
    games = fetch_games_for_date(d)
    print(f"Slate games: {len(games)}\n")

    for game in games:
        matchup_label = f"{game['away_name']} @ {game['home_name']}"
        print(f"=== {matchup_label} ===")
        info = get_starting_pitcher_info(int(game["game_id"]))
        print(
            f"  Starters: {info['away_pitcher_name']} (away) vs {info['home_pitcher_name']} (home)"
        )

        for side, pid_key, name_key, opp_id in (
            ("home", "home_pitcher_id", "home_pitcher_name", int(game["away_id"])),
            ("away", "away_pitcher_id", "away_pitcher_name", int(game["home_id"])),
        ):
            pid = info.get(pid_key)
            if pid is None:
                print(f"  [{side}] no pitcher id")
                continue
            profile = fetch_pitcher_season_profile(
                int(pid), pitcher_name=str(info.get(name_key) or ""), season=d.year
            )
            if profile is None:
                print(f"  [{side}] {info[name_key]}: season profile unavailable")
                continue
            scalar, tag = pitcher_runs_allowed_scalar(profile)
            offense = fetch_team_offense_profile(opp_id, season=d.year)
            flags = []
            if tag:
                flags.append(tag)
            if is_power_pitcher(profile) and is_velo_struggler(opp_id, d.year):
                flags.append("velo_dominance")
            if is_ground_ball_pitcher(profile) and is_patient_lineup(offense):
                flags.append("patient_lineup")
            print(
                f"  [{side}] {profile.pitcher_name}: "
                f"ERA={profile.season_era} WHIP={profile.season_whip} "
                f"BABIP={profile.season_babip} velo={profile.avg_fastball_velo} "
                f"GB%={profile.ground_ball_pct}"
            )
            print(f"         stability={scalar:.2f} flags={flags or ['none']}")

        features = baseball_data.build_prediction_row(
            game["home_id"],
            game["away_id"],
            d,
            history,
            window=DEFAULT_ROLLING_WINDOW,
            venue_id=game.get("venue_id"),
        )
        home_runs, away_runs, _ = model.predict_matchup(models, features)
        before = (home_runs, away_runs)
        result = apply_pitcher_matchup_adjustments(
            home_runs,
            away_runs,
            game_id=int(game["game_id"]),
            home_id=int(game["home_id"]),
            away_id=int(game["away_id"]),
            season=d.year,
        )
        print(
            f"  Runs: away {before[1]:.2f}->{result.away_runs:.2f}, "
            f"home {before[0]:.2f}->{result.home_runs:.2f}"
        )
        if result.tags:
            for tag in result.tags:
                print(f"    + {tag}")
        else:
            print("    + no matchup adjustments applied")
        print()


if __name__ == "__main__":
    main()
