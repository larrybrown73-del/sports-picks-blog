from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cache_store import cache_path, read_cache, write_cache


def test_cache_path_formats_category_and_date() -> None:
    path = cache_path("player_season_stats", date(2026, 7, 7), cache_dir=Path("cache"))
    assert path == Path("cache/player_season_stats_2026_07_07.json")


def test_read_cache_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    assert read_cache("anything", date(2026, 7, 7), cache_dir=tmp_path) == {}


def test_read_cache_returns_empty_dict_on_corrupt_json(tmp_path: Path) -> None:
    path = cache_path("broken", date(2026, 7, 7), cache_dir=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    assert read_cache("broken", date(2026, 7, 7), cache_dir=tmp_path) == {}


def test_read_cache_returns_empty_dict_when_json_root_is_not_an_object(tmp_path: Path) -> None:
    path = cache_path("list_root", date(2026, 7, 7), cache_dir=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    assert read_cache("list_root", date(2026, 7, 7), cache_dir=tmp_path) == {}


def test_write_cache_creates_missing_cache_dir(tmp_path: Path) -> None:
    nested_cache_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested_cache_dir.exists()

    write_cache("team_players", date(2026, 7, 7), {"tm_1": [{"id": "pl_1"}]}, cache_dir=nested_cache_dir)

    assert nested_cache_dir.exists()
    assert cache_path("team_players", date(2026, 7, 7), cache_dir=nested_cache_dir).exists()


def test_write_then_read_cache_round_trips_data_including_none_values(tmp_path: Path) -> None:
    data = {"pl_1|sn_1": {"goals": 5}, "pl_2|sn_1": None}
    write_cache("player_season_stats", date(2026, 7, 7), data, cache_dir=tmp_path)

    loaded = read_cache("player_season_stats", date(2026, 7, 7), cache_dir=tmp_path)
    assert loaded == data
    assert loaded["pl_2|sn_1"] is None


def test_cache_is_scoped_per_calendar_day(tmp_path: Path) -> None:
    write_cache("historical_matches", date(2026, 7, 6), {"comp_1": ["yesterday"]}, cache_dir=tmp_path)
    write_cache("historical_matches", date(2026, 7, 7), {"comp_1": ["today"]}, cache_dir=tmp_path)

    assert read_cache("historical_matches", date(2026, 7, 6), cache_dir=tmp_path) == {"comp_1": ["yesterday"]}
    assert read_cache("historical_matches", date(2026, 7, 7), cache_dir=tmp_path) == {"comp_1": ["today"]}
