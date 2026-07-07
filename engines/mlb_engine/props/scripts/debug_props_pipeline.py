"""Fast props-pipeline diagnostic (no statcast/pybaseball)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseball_props.data.ingest import fetch_live_vegas_totals, get_rundown_api_key
from baseball_props.data.odds_props import (
    _map_to_odds_api_event_ids,
    fetch_all_player_props,
)


def main() -> None:
    rundown_key = get_rundown_api_key(required=False)
    print(f"[RUNDOWN_API_KEY] loaded={bool(rundown_key)} len={len(rundown_key or '')}")

    live_vegas = fetch_live_vegas_totals()
    print(f"Vegas rows: {len(live_vegas)}")

    event_ids = live_vegas["game_id"].dropna().astype(str).tolist()[:6]
    odds_map = _map_to_odds_api_event_ids(event_ids)
    print(f"Odds API ID map: {len(odds_map)}/{len(event_ids)}")
    for ext, odds in list(odds_map.items())[:2]:
        print(f"  {ext[:12]}... -> {odds[:12]}...")

    prop_lines = fetch_all_player_props(event_ids[:6] if event_ids else [])
    api_names = prop_lines["player_name"].dropna().unique().tolist()[:3] if not prop_lines.empty else []
    print(f"Prop line rows: {len(prop_lines)}")
    print(f"API players (sample): {api_names}")
    if not prop_lines.empty:
        print(f"Prop game_ids (unique): {prop_lines['game_id'].unique()[:3].tolist()}")


if __name__ == "__main__":
    main()
