#!/usr/bin/env python3
"""Compute ISR trigger time: 20 minutes before today's first MLB first pitch."""

from __future__ import annotations

import sys
from datetime import timezone
from pathlib import Path

MLB_ENGINE = Path(__file__).resolve().parents[1] / "engines" / "mlb_engine"
sys.path.insert(0, str(MLB_ENGINE))

from utils.first_pitch import get_first_pitch_times  # noqa: E402


def get_first_pitch_trigger_time():
    first_pitch, trigger_time = get_first_pitch_times()
    if first_pitch is None or trigger_time is None:
        print("No games scheduled for today.")
        return None
    print(f"First Pitch Today: {first_pitch.astimezone(timezone.utc)}")
    print(f"Target ISR Trigger Time: {trigger_time.astimezone(timezone.utc)}")
    return trigger_time


if __name__ == "__main__":
    get_first_pitch_trigger_time()
