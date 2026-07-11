#!/usr/bin/env python3
"""Export WNBA slate from the WNBA prediction engine to blog JSON."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "picks" / "wnba"
DEFAULT_WNBA_ROOT = Path(r"D:\Juniors Files\WNBA")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_wnba_root() -> Path:
    env = load_env_file(PROJECT_ROOT / ".env.local")
    raw = os.environ.get("WNBA_ENGINE_PATH") or env.get("WNBA_ENGINE_PATH") or str(DEFAULT_WNBA_ROOT)
    root = Path(raw)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def main() -> None:
    wnba_root = resolve_wnba_root()
    if not wnba_root.exists():
        raise SystemExit(f"WNBA engine not found at {wnba_root}")

    sys.path.insert(0, str(wnba_root))

    from data.env_loader import load_env

    load_env()

    from model.slate_export import build_slate_payload

    payload = build_slate_payload(seasons=[2025, 2026], top_per_team=12)
    slate_date = payload["games"][0]["date"] if payload.get("games") else date.today().isoformat()
    payload["date"] = slate_date
    payload["generatedAt"] = payload.get("generated_at", "")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = OUTPUT_DIR / f"{slate_date}.json"
    latest_path = OUTPUT_DIR / "latest.json"

    text = json.dumps(payload, indent=2)
    dated_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")

    print(f"Exported WNBA picks for {slate_date}")
    print(f"  Games: {len(payload.get('games', []))}")
    print(f"  Approved: {payload.get('approved_count', 0)}")
    print(f"  -> {dated_path}")
    print(f"  -> {latest_path}")


if __name__ == "__main__":
    main()
