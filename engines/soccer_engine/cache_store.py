"""
Local, file-based, same-day cache for expensive TheStatsAPI fetches.

Why this exists: daily_model.py's player-prop coverage requires one roster
call per team plus one season-stats call per rostered player -- a
full-slate day can mean several hundred HTTP calls, most of which return
the exact same answer as an hour ago (a player's season totals don't
change mid-day, and neither does a team's roster). Re-running the morning
pull, retrying after a partial failure, or grading multiple matches from
the same competition should not re-download all of that from scratch.

Design, deliberately simple:
  - One JSON file per (category, calendar day), e.g.
    cache/player_season_stats_2026_07_07.json. The date is baked into the
    filename, so "is this cache valid" is just "does today's file exist" --
    no separate expiry/TTL bookkeeping, and yesterday's file is
    automatically never consulted again.
  - Each file holds a flat `{cache_key: value}` JSON object, so many
    different entities (one per team, one per player, ...) fetched over
    the course of the same day accumulate into the same file instead of
    each needing its own.
  - `value` may be `None` -- that is a deliberately CACHED "the API had
    nothing for this key" answer (e.g. a player with no season stats),
    distinct from "we haven't asked yet". Re-checking `key in cache_dict`
    (not `cache_dict.get(key)`) is what tells the two apart; see
    daily_model.py's cached wrappers.
  - Never a source of truth: this is a quota-saving read-through cache in
    front of the API, not a replacement for it. A missing/corrupt file is
    treated as an empty cache, never an error.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def cache_path(category: str, for_date: date, *, cache_dir: Path = CACHE_DIR) -> Path:
    """e.g. cache_path("player_season_stats", date(2026, 7, 7)) -> cache/player_season_stats_2026_07_07.json"""

    return cache_dir / f"{category}_{for_date.strftime('%Y_%m_%d')}.json"


def read_cache(category: str, for_date: date, *, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """
    Load today's cache file for `category`. Returns {} (never raises) if the
    file doesn't exist yet or contains unreadable JSON -- a corrupt cache
    should degrade to "cache miss for everything", not crash the pipeline.
    """

    path = cache_path(category, for_date, cache_dir=cache_dir)
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_cache(category: str, for_date: date, data: dict[str, Any], *, cache_dir: Path = CACHE_DIR) -> None:
    """Persist `data` as today's cache file for `category`, creating cache_dir if needed."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(category, for_date, cache_dir=cache_dir)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
