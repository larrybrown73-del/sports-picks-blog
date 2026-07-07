#!/usr/bin/env python3
"""Run mock-moneyline slate evaluation with weather and bullpen columns."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from performance_check import _print_value_picks, get_upcoming_value_picks  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    game_date = date.today()
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    if args and args[0][:1].isdigit():
        game_date = date.fromisoformat(args[0])

    write_log = "--log" in sys.argv
    picks = get_upcoming_value_picks(game_date, write_log=write_log)
    _print_value_picks(picks, game_date)


if __name__ == "__main__":
    main()
