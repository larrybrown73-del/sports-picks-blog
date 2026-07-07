#!/usr/bin/env python3
"""Wait until T-20 before first pitch, then hit the site ISR revalidation endpoint."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.first_pitch import get_first_pitch_times  # noqa: E402


def _revalidate(paths: list[str]) -> None:
    secret = os.environ.get("REVALIDATION_SECRET", "").strip()
    site_url = os.environ.get("SITE_URL", "").strip().rstrip("/")
    if not secret or not site_url:
        print("Skip revalidation: REVALIDATION_SECRET or SITE_URL not set.")
        return

    for page_path in paths:
        url = f"{site_url}/api/revalidate"
        response = requests.post(
            url,
            params={"secret": secret, "path": page_path},
            timeout=30,
        )
        print(f"Revalidate {page_path}: {response.status_code} {response.text[:200]}")


def main() -> int:
    first_pitch, trigger_time = get_first_pitch_times()
    if first_pitch is None or trigger_time is None:
        print("No first-pitch window today.")
        return 0

    now = datetime.now(timezone.utc)
    print(f"First pitch: {first_pitch.isoformat()}")
    print(f"Trigger (T-20): {trigger_time.isoformat()}")
    print(f"Now: {now.isoformat()}")

    if now >= first_pitch:
        print("First pitch already started; skipping ISR hook.")
        return 0

    if now < trigger_time:
        wait_seconds = int((trigger_time - now).total_seconds())
        max_sleep = int(os.environ.get("FIRST_PITCH_MAX_SLEEP_SECONDS", str(6 * 3600)))
        if wait_seconds > max_sleep:
            print(f"Wait {wait_seconds}s exceeds max {max_sleep}s; skipping sleep.")
            return 0
        print(f"Sleeping {wait_seconds}s until ISR trigger...")
        time.sleep(wait_seconds)

    _revalidate(["/", "/picks", "/performance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
