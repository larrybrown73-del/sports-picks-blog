#!/usr/bin/env python3
"""Live diagnostic: pitcher matchup tags for remaining / in-progress slate games."""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
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

import statsapi  # noqa: E402
from backtest import _memoize_obp_lookups, load_frozen_models  # noqa: E402
import baseball_data  # noqa: E402
from baseball_data import get_starting_pitcher_info  # noqa: E402
from config import DEFAULT_ROLLING_WINDOW  # noqa: E402
from pitcher_matchup import (  # noqa: E402
    fetch_pitcher_season_profile,
    pitcher_runs_allowed_scalar,
)
from slate_evaluation import evaluate_game  # noqa: E402

FINAL_STATUSES = {
    "final",
    "game over",
    "completed early",
    "postponed",
    "cancelled",
    "canceled",
    "suspended",
}


def _is_active_game(status: str) -> bool:
    return status.strip().lower() not in FINAL_STATUSES


def _fetch_slate_with_status(game_date: date) -> list[dict]:
    games = statsapi.schedule(date=game_date.strftime("%m/%d/%Y"), sportId=1)
    rows: list[dict] = []
    for game in games or []:
        if game.get("game_type") != "R":
            continue
        rows.append(
            {
                "game_id": game["game_id"],
                "status": str(game.get("status", "Unknown")),
                "home_id": game["home_id"],
                "away_id": game["away_id"],
                "home_name": game["home_name"],
                "away_name": game["away_name"],
                "venue_id": game.get("venue_id"),
            }
        )
    return rows


def _classify_tags(tags: list[str]) -> dict[str, list[str]]:
    buckets = {
        "babip_luck": [],
        "regression": [],
        "velo_dominance": [],
        "patient_lineup": [],
        "other": [],
    }
    for tag in tags:
        lowered = tag.lower()
        if "babip_luck" in lowered:
            buckets["babip_luck"].append(tag)
        elif "regression_penalty" in lowered:
            buckets["regression"].append(tag)
        elif "velo_dominance" in lowered:
            buckets["velo_dominance"].append(tag)
        elif "patient_lineup" in lowered:
            buckets["patient_lineup"].append(tag)
        else:
            buckets["other"].append(tag)
    return buckets


def main() -> None:
    game_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    print(f"Live pitcher matchup diagnostic — {game_date.isoformat()}")
    print(f"Run time: {datetime.now().isoformat(timespec='seconds')}\n")

    _memoize_obp_lookups()
    history = baseball_data.games_for_prediction([game_date.year])
    models = load_frozen_models()

    slate = _fetch_slate_with_status(game_date)
    active = [g for g in slate if _is_active_game(g["status"])]
    final_count = len(slate) - len(active)

    print(f"Full slate: {len(slate)} games ({final_count} completed, {len(active)} remaining/in-progress)\n")
    if not active:
        print("No remaining games to evaluate.")
        return

    all_babip: list[str] = []
    all_regression: list[str] = []
    all_velo: list[str] = []
    all_patient: list[str] = []

    for game in active:
        matchup = f"{game['away_name']} @ {game['home_name']}"
        print("=" * 72)
        print(f"{matchup}  [{game['status']}]")

        pitcher_info = get_starting_pitcher_info(int(game["game_id"]))
        print(
            f"Starters: {pitcher_info['away_pitcher_name']} (away) | "
            f"{pitcher_info['home_pitcher_name']} (home)"
        )

        for side, pid_key, name_key in (
            ("home", "home_pitcher_id", "home_pitcher_name"),
            ("away", "away_pitcher_id", "away_pitcher_name"),
        ):
            pid = pitcher_info.get(pid_key)
            name = pitcher_info.get(name_key)
            if pid is None:
                print(f"  {side.upper()} SP {name}: no MLBAM id")
                continue
            profile = fetch_pitcher_season_profile(
                int(pid), pitcher_name=str(name or ""), season=game_date.year
            )
            if profile is None:
                print(f"  {side.upper()} SP {name}: season profile unavailable")
                continue
            scalar, stability = pitcher_runs_allowed_scalar(profile)
            stability_label = stability or "neutral"
            print(
                f"  {side.upper()} SP {profile.pitcher_name}: "
                f"ERA {profile.season_era} | WHIP {profile.season_whip} | "
                f"BABIP {profile.season_babip} | velo {profile.avg_fastball_velo} | "
                f"GB% {profile.ground_ball_pct} | stability x{scalar:.2f} ({stability_label})"
            )

        try:
            evaluated = evaluate_game(
                game,
                game_date,
                history,
                models,
                window=DEFAULT_ROLLING_WINDOW,
            )
        except Exception as exc:
            print(f"  Evaluation error: {exc}")
            print()
            continue

        print(
            f"Projected runs: {game['away_name']} {evaluated.away_runs:.2f} — "
            f"{game['home_name']} {evaluated.home_runs:.2f}"
        )
        print(
            f"Win prob: {game['away_name']} {evaluated.away_prob:.1%} | "
            f"{game['home_name']} {evaluated.home_prob:.1%}"
        )

        tags = evaluated.pitcher_matchup_tags
        if tags:
            print("pitcher_matchup_tags:")
            for tag in tags:
                print(f"  - {tag}")
        else:
            print("pitcher_matchup_tags: (none)")

        buckets = _classify_tags(tags)
        all_babip.extend(buckets["babip_luck"])
        all_regression.extend(buckets["regression"])
        all_velo.extend(buckets["velo_dominance"])
        all_patient.extend(buckets["patient_lineup"])
        print()

    print("=" * 72)
    print("SLATE-WIDE TRIGGER SUMMARY (remaining / in-progress only)")
    print("=" * 72)
    print(f"BABIP_LUCK_BONUS (0.95x runs allowed): {len(all_babip)}")
    for tag in all_babip:
        print(f"  + {tag}")
    print(f"\nREGRESSION_PENALTY (1.15x runs allowed): {len(all_regression)}")
    for tag in all_regression:
        print(f"  + {tag}")
    print(f"\nVELO_DOMINANCE_SCALAR (0.88x offense): {len(all_velo)}")
    for tag in all_velo:
        print(f"  + {tag}")
    print(f"\nPATIENT_LINEUP_ADVANTAGE (1.12x offense): {len(all_patient)}")
    for tag in all_patient:
        print(f"  + {tag}")


if __name__ == "__main__":
    main()
