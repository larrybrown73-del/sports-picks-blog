#!/usr/bin/env python3
"""Focused diagnostic for specific slate games and starting pitchers."""

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

import model  # noqa: E402
from backtest import _memoize_obp_lookups, load_frozen_models  # noqa: E402
import baseball_data  # noqa: E402
from baseball_data import get_starting_pitcher_info  # noqa: E402
from config import DEFAULT_ROLLING_WINDOW  # noqa: E402
from pitcher_matchup import apply_pitcher_matchup_adjustments  # noqa: E402
from slate_evaluation import evaluate_game  # noqa: E402

TARGETS = {
    823357: {
        "label": "MIL @ PIT",
        "pitchers": ("Brandon Sproat", "Braxton Ashcraft"),
    },
    823200: {
        "label": "COL @ SF",
        "pitchers": ("Tanner Gordon", "Robbie Ray"),
    },
}


def _bucket_tags(tags: list[str]) -> dict[str, list[str]]:
    buckets = {
        "starter_rest_hierarchy": [],
        "hitter_discipline": [],
        "pitcher_style": [],
        "bullpen": [],
        "other": [],
    }
    for tag in tags:
        lowered = tag.lower()
        if any(
            key in lowered
            for key in (
                "short_rest",
                "optimal_rest",
                "rust_penalty",
                "tier",
                "true_ace",
                "depth_chart",
                "innings_eater",
                "ace_synergy",
                "starter_scalar",
                "rotation_slot",
                "days_rest",
                "gold_glove",
                "poor_defense",
                "k_bb_pct",
            )
        ):
            buckets["starter_rest_hierarchy"].append(tag)
        elif any(
            key in lowered
            for key in (
                "discipline",
                "erratic",
                "lineup_scalar",
                "premium",
                "bottom",
                "slot",
                "away_offense",
                "home_offense",
            )
        ):
            buckets["hitter_discipline"].append(tag)
        elif any(
            key in lowered
            for key in (
                "babip_luck",
                "regression_penalty",
                "velo_dominance",
                "patient_lineup",
                "gb_pitcher",
                "velo_erratic",
                "velo_team",
            )
        ):
            buckets["pitcher_style"].append(tag)
        elif any(
            key in lowered
            for key in (
                "dead_arm",
                "lockdown",
                "lineup_scalar",
                "bullpen",
                "late_inning",
                "away_runs_late",
                "home_runs_late",
            )
        ):
            buckets["bullpen"].append(tag)
        else:
            buckets["other"].append(tag)
    return buckets


def _pitcher_tags(all_tags: list[str], pitcher_name: str) -> list[str]:
    prefix = pitcher_name.split()[0]  # last name anchor
    return [tag for tag in all_tags if pitcher_name in tag or tag.startswith(f"{prefix}:")]


def main() -> None:
    game_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    print(f"Focused pitcher diagnostic — {game_date.isoformat()}")
    print(f"Run time: {datetime.now().isoformat(timespec='seconds')}\n")

    _memoize_obp_lookups()
    history = baseball_data.games_for_prediction([game_date.year])
    models = load_frozen_models()
    games = baseball_data.fetch_games_for_date(game_date)
    by_id = {int(g["game_id"]): g for g in games}

    for game_id, meta in TARGETS.items():
        game = by_id.get(game_id)
        if game is None:
            print(f"=== {meta['label']} (game {game_id}) — NOT ON SLATE ===\n")
            continue

        pitcher_info = get_starting_pitcher_info(game_id)
        print("=" * 78)
        print(f"{meta['label']}  |  game_id={game_id}")
        print(
            f"Starters: {pitcher_info['away_pitcher_name']} (away) vs "
            f"{pitcher_info['home_pitcher_name']} (home)"
        )

        features = baseball_data.build_prediction_row(
            game["home_id"],
            game["away_id"],
            game_date,
            history,
            window=DEFAULT_ROLLING_WINDOW,
            venue_id=game.get("venue_id"),
        )
        base_home, base_away, _ = model.predict_matchup(models, features)
        matchup = apply_pitcher_matchup_adjustments(
            base_home,
            base_away,
            game_id=game_id,
            home_id=int(game["home_id"]),
            away_id=int(game["away_id"]),
            season=game_date.year,
            game_date=game_date,
        )

        evaluated = evaluate_game(game, game_date, history, models, window=DEFAULT_ROLLING_WINDOW)

        print("\nRun pipeline:")
        print(f"  RF baseline:      away {base_away:.2f} | home {base_home:.2f}")
        print(
            f"  After SP/matchup: away {matchup.away_runs:.2f} | home {matchup.home_runs:.2f}"
        )
        print(
            f"  Final (w/ BP/env): away {evaluated.away_runs:.2f} | home {evaluated.home_runs:.2f}"
        )
        print(
            f"  Win probability:  away {evaluated.away_prob:.1%} | home {evaluated.home_prob:.1%}"
        )

        fatigue = evaluated.fatigue
        print("\nBullpen fatigue scalars:")
        print(f"  Away status: {fatigue.away_status}")
        print(f"  Home status: {fatigue.home_status}")
        print(f"  Away opponent late scalar: {fatigue.away_opponent_late_scalar:.3f}")
        print(f"  Home opponent late scalar: {fatigue.home_opponent_late_scalar:.3f}")
        if evaluated.bullpen_tags:
            print("  bullpen_tags:")
            for tag in evaluated.bullpen_tags:
                print(f"    - {tag}")
        else:
            print("  bullpen_tags: (none)")

        print("\npitcher_matchup_tags (full):")
        if evaluated.pitcher_matchup_tags:
            for tag in evaluated.pitcher_matchup_tags:
                print(f"  - {tag}")
        else:
            print("  (none)")

        buckets = _bucket_tags(evaluated.pitcher_matchup_tags)
        print("\nModifier buckets:")
        for bucket, tags in buckets.items():
            if not tags:
                continue
            print(f"  [{bucket}]")
            for tag in tags:
                print(f"    - {tag}")

        print("\nPer-starter breakdown:")
        for pitcher_name in meta["pitchers"]:
            starter_tags = _pitcher_tags(evaluated.pitcher_matchup_tags, pitcher_name)
            print(f"  {pitcher_name}:")
            if starter_tags:
                for tag in starter_tags:
                    print(f"    - {tag}")
            else:
                print("    - (no direct tags — see offense/lineup buckets above)")

        favored = game["away_name"] if evaluated.away_prob > evaluated.home_prob else game["home_name"]
        print(f"\nModel favorite: {favored}")
        print()


if __name__ == "__main__":
    main()
