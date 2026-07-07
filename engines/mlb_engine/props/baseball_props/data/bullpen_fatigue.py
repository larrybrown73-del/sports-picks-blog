from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from baseball_props.config import TB_BULLPEN_FATIGUE_THRESHOLD
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_PITCHER_POSITION_CODE = "1"
_RELIEF_PITCH_CAP = 80
_CONSECUTIVE_DAYS_CAP = 3


@dataclass(frozen=True)
class BullpenFatigueResult:
    score: float
    status: str
    reliever_count: int


def _fatigue_status(score: float) -> str:
    if score >= TB_BULLPEN_FATIGUE_THRESHOLD:
        return "Fatigued"
    if score >= 0.40:
        return "Moderate"
    return "Fresh"


def _neutral_fatigue() -> BullpenFatigueResult:
    return BullpenFatigueResult(score=0.35, status="Moderate", reliever_count=0)


def _fetch_active_pitcher_ids(team_id: int) -> list[str]:
    from baseball_props.data.mlb_live import _get_json

    try:
        payload = _get_json(f"/teams/{team_id}/roster", {"rosterType": "active"})
    except Exception as exc:
        logger.debug("Bullpen roster fetch failed for team %s: %s", team_id, exc)
        return []

    pitcher_ids: list[str] = []
    for entry in payload.get("roster", []):
        position = entry.get("position", {}) or {}
        if str(position.get("code", "")) != _PITCHER_POSITION_CODE:
            continue
        person = entry.get("person", {}) or {}
        pid = person.get("id")
        if pid is not None:
            pitcher_ids.append(str(pid))
    return pitcher_ids


def _pitcher_workload_last_days(pitcher_id: str, lookback_days: int) -> tuple[int, int]:
    """Return (pitch_count, distinct_active_days) in the lookback window."""
    try:
        from pybaseball import statcast_pitcher

        end = date.today()
        start = end - timedelta(days=lookback_days)
        sc = statcast_pitcher(start.isoformat(), end.isoformat(), int(pitcher_id))
        if sc is None or sc.empty:
            return 0, 0
        pitches = len(sc)
        active_days = sc["game_date"].astype(str).nunique() if "game_date" in sc.columns else 0
        return pitches, int(active_days)
    except Exception as exc:
        logger.debug("Reliever workload fetch failed for %s: %s", pitcher_id, exc)
        return 0, 0


def _arm_fatigue_score(pitches: int, active_days: int) -> float:
    pitch_component = min(1.0, pitches / _RELIEF_PITCH_CAP)
    day_component = min(1.0, active_days / _CONSECUTIVE_DAYS_CAP)
    return pitch_component * 0.65 + day_component * 0.35


@lru_cache(maxsize=64)
def compute_bullpen_fatigue_score(
    team_id: str,
    *,
    lookback_days: int = 3,
) -> BullpenFatigueResult:
    """
    Estimate opponent bullpen fatigue from recent relief pitch volume.

    Score is 0.0–1.0; Fatigued when score >= TB_BULLPEN_FATIGUE_THRESHOLD.
    """
    tid_text = str(team_id).strip()
    if not tid_text.isdigit():
        return _neutral_fatigue()

    pitcher_ids = _fetch_active_pitcher_ids(int(tid_text))
    if not pitcher_ids:
        return _neutral_fatigue()

    max_score = 0.0
    evaluated = 0
    for pid in pitcher_ids[:14]:
        pitches, days = _pitcher_workload_last_days(pid, lookback_days)
        if pitches == 0 and days == 0:
            continue
        evaluated += 1
        max_score = max(max_score, _arm_fatigue_score(pitches, days))

    if evaluated == 0:
        return _neutral_fatigue()

    return BullpenFatigueResult(
        score=round(max_score, 3),
        status=_fatigue_status(max_score),
        reliever_count=evaluated,
    )


def build_team_bullpen_fatigue_table(team_ids: list[str]) -> pd.DataFrame:
    """Build per-team bullpen fatigue columns for slate context merge."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in team_ids:
        tid = str(raw).strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        result = compute_bullpen_fatigue_score(tid)
        rows.append(
            {
                "opp_team_id": tid,
                "opp_bullpen_fatigue_score": result.score,
                "opp_bullpen_fatigue_status": result.status,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "opp_team_id",
                "opp_bullpen_fatigue_score",
                "opp_bullpen_fatigue_status",
            ]
        )
    return pd.DataFrame(rows)
