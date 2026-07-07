#!/usr/bin/env python3
"""Inject canvas JSON payloads into a .canvas.tsx file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GAME_MARKER_START = "// GAME_SPLITS_START"
GAME_MARKER_END = "// GAME_SPLITS_END"
INTEL_MARKER_START = "// BETTING_INTEL_START"
INTEL_MARKER_END = "// BETTING_INTEL_END"
SLATE_DATE_PATTERN = re.compile(r'const SLATE_DATE = "[^"]*";')

DEFAULT_CANVAS = (
    Path.home()
    / ".cursor"
    / "projects"
    / "d-Juniors-Files-baseball-props-model"
    / "canvases"
    / "mlb-props-july-3.canvas.tsx"
)


def _replace_block(text: str, start: str, end: str, block: str, *, anchor: str) -> str:
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _match: block, text, count=1)
    if anchor not in text:
        raise ValueError(f"Could not find insertion anchor '{anchor}'")
    return text.replace(anchor, f"{block}\n\n{anchor}", 1)


def _inject_slate_date(text: str, slate_date: str) -> str:
    replacement = f'const SLATE_DATE = "{slate_date}";'
    if SLATE_DATE_PATTERN.search(text):
        return SLATE_DATE_PATTERN.sub(replacement, text, count=1)
    anchor = "const SOURCE ="
    if anchor not in text:
        raise ValueError("Could not find SLATE_DATE insertion anchor")
    return text.replace(anchor, f"{replacement}\n{anchor}", 1)


def sync_canvas(canvas_path: Path, games_json: Path, intel_json: Path | None = None) -> None:
    games = json.loads(games_json.read_text(encoding="utf-8"))
    games_payload = json.dumps(games, indent=2)
    games_block = f"{GAME_MARKER_START}\nconst GAME_SPLITS: GameSplit[] = {games_payload};\n{GAME_MARKER_END}"

    text = canvas_path.read_text(encoding="utf-8")
    text = _replace_block(text, GAME_MARKER_START, GAME_MARKER_END, games_block, anchor="const SLATE_DATE")

    slate_date = None
    if intel_json is not None and intel_json.exists():
        intel = json.loads(intel_json.read_text(encoding="utf-8"))
        slate_date = intel.get("slate_date")
        intel_payload = json.dumps(intel, indent=2)
        intel_block = (
            f"{INTEL_MARKER_START}\n"
            f"const BETTING_INTEL: BettingIntel = {intel_payload};\n"
            f"{INTEL_MARKER_END}"
        )
        text = _replace_block(
            text,
            INTEL_MARKER_START,
            INTEL_MARKER_END,
            intel_block,
            anchor="const GAME_SPLITS",
        )

    if slate_date:
        text = _inject_slate_date(text, str(slate_date))

    canvas_path.write_text(text, encoding="utf-8")
    print(f"Synced {len(games)} games into {canvas_path}")
    if intel_json is not None and intel_json.exists():
        print(f"Synced betting intel from {intel_json}")
    if slate_date:
        print(f"Updated SLATE_DATE to {slate_date}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync canvas JSON into canvas TSX")
    parser.add_argument("--canvas", type=Path, default=None, help="Path to .canvas.tsx file")
    parser.add_argument("--json", type=Path, default=Path("exports/today/canvas_games.json"))
    parser.add_argument(
        "--intel-json",
        type=Path,
        default=None,
        help="Optional canvas_betting_intel.json path",
    )
    args = parser.parse_args()
    canvas = args.canvas or DEFAULT_CANVAS
    intel = args.intel_json
    if intel is None and args.json.parent.joinpath("canvas_betting_intel.json").exists():
        intel = args.json.parent / "canvas_betting_intel.json"
    sync_canvas(canvas, args.json, intel)


if __name__ == "__main__":
    main()
