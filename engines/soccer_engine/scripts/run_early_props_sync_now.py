"""One-off manual invocation of the 9:30 AM early props sync (Job B)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "early_props_sync_manual.log", encoding="utf-8"), logging.StreamHandler()],
)

from scheduler import SoccerEvScheduler  # noqa: E402

sched = SoccerEvScheduler()
sched.run_early_props_sync()
print("DONE. Models available for competitions:", list(sched._models_by_competition.keys()))
