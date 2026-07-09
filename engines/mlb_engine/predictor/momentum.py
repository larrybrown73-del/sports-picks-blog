"""Team momentum helpers (win streaks via MLB Stats API)."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

from config import TEAM_STREAK_MIN_WINS


@lru_cache(maxsize=64)
def team_win_streak(team_id: int, as_of_iso: str) -> int:
    """
    Count consecutive wins for *team_id* entering games on as_of date.

    Walks backward through completed regular-season games; streak breaks on
    loss or tie. Returns 0 when schedule data is unavailable.
    """
    import statsapi

    as_of = date.fromisoformat(as_of_iso)
    start = (as_of - timedelta(days=21)).strftime("%m/%d/%Y")
    end = (as_of - timedelta(days=1)).strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(start_date=start, end_date=end, team=team_id, sportId=1)
    except Exception:
        return 0

    results: list[str] = []
    for game in sorted(games, key=lambda row: row.get("game_date", "")):
        if game.get("game_type") != "R":
            continue
        if str(game.get("status", "")) != "Final":
            continue
        home_id = game.get("home_id")
        away_id = game.get("away_id")
        home_score = game.get("home_score")
        away_score = game.get("away_score")
        if home_score is None or away_score is None:
            continue
        if team_id == home_id:
            if home_score > away_score:
                results.append("W")
            elif home_score < away_score:
                results.append("L")
            else:
                results.append("T")
        elif team_id == away_id:
            if away_score > home_score:
                results.append("W")
            elif away_score < home_score:
                results.append("L")
            else:
                results.append("T")

    streak = 0
    for outcome in reversed(results):
        if outcome == "W":
            streak += 1
        else:
            break
    return streak


def apply_team_streak_bonus(prob: float, team_id: int, as_of: date) -> float:
    """Boost win probability when the team is on a qualifying hot streak."""
    from config import TEAM_STREAK_BONUS

    if prob <= 0:
        return prob
    streak = team_win_streak(int(team_id), as_of.isoformat())
    if streak >= TEAM_STREAK_MIN_WINS:
        return min(prob * TEAM_STREAK_BONUS, 0.99)
    return prob
