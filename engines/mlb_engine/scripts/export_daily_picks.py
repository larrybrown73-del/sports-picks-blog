#!/usr/bin/env python3
"""Monorepo entrypoint — delegates to project-root export_daily_picks.py."""

from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(str(PROJECT_ROOT / "scripts" / "export_daily_picks.py"), run_name="__main__")
