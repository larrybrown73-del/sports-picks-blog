from __future__ import annotations

from datetime import datetime, timedelta, timezone


def get_first_pitch_times() -> tuple[datetime | None, datetime | None]:
    """
    Return (first_pitch_utc, trigger_time_utc) where trigger is T-20 minutes.
    """
    import statsapi

    today = datetime.now().strftime("%m/%d/%Y")
    slate = statsapi.schedule(date=today, sportId=1)
    if not slate:
        return None, None

    game_times: list[datetime] = []
    for game in slate:
        if game.get("game_type") != "R":
            continue
        raw = game.get("game_datetime") or game.get("game_date")
        if not raw:
            continue
        gtime = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if gtime.tzinfo is None:
            gtime = gtime.replace(tzinfo=timezone.utc)
        game_times.append(gtime)

    if not game_times:
        return None, None

    first_pitch = min(game_times)
    trigger_time = first_pitch - timedelta(minutes=20)
    return first_pitch, trigger_time
