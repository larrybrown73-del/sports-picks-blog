"""One-off manual invocation used to catch up today's missed 8 AM morning pull."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from scheduler import SoccerEvScheduler  # noqa: E402

sched = SoccerEvScheduler()
sched.run_morning_pull()
print("DONE. Models built for competitions:", list(sched._models_by_competition.keys()))
