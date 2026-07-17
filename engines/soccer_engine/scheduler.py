"""
Master scheduler: orchestrates the three daily automation jobs on top of
the EV engine, using APScheduler's BackgroundScheduler (a daemon-thread
scheduler -- `SoccerEvScheduler.start()` returns immediately and does not
block the calling process/other work).

  Job A ("Morning Pull"), 8:00 AM America/New_York, every day:
    1. Fetch today's match schedule from TheStatsAPI.
    2. Fit one SoccerProjectionModel per competition represented in that
       schedule (see daily_model.py).
    3. Grade every match's TEAM-level markets only (moneyline / btts /
       total / spread) and push an "Early Value" Telegram message.
    4. For every match that day, schedule a one-shot Job C run for exactly
       60 minutes before that match's kickoff.

  Job B ("Early Props Sync"), 9:30 AM America/New_York, every day:
    Grades the FULL board -- team markets AND player props -- for every
    match fitted by Job A, and pushes an "Early Props Board" Telegram
    message. This exists specifically for early-kickoff slates (e.g. a
    World Cup window that starts at noon): waiting for Job C's
    confirmed-lineup pass (kickoff - 60 minutes) would otherwise be the
    FIRST time player props get graded at all, leaving as little as an
    hour of decision time before an early kickoff. Job B intentionally
    grades props off player_props_model.py's historical
    minutes-per-appearance stand-in (no confirmed teamsheet exists yet
    this early) -- the message says so -- and Job C still supersedes it
    with the confirmed-lineup-accurate "Locked-In Ticket" closer to
    kickoff. 9:30 AM is a fixed, deliberately-early time chosen to leave
    real runway before the earliest realistic kickoff, independent of any
    single match's own kickoff - 60 minutes math.

  IMPORTANT: both cron jobs above only fire while `SoccerEvScheduler` is
  actually a running process -- if nothing is running at the scheduled
  time (process not started yet, machine asleep/rebooting, a crash), that
  day's pull simply never happens, silently, with no alert to explain why
  nothing arrived. `start()` guards against that for both jobs: every time
  the process starts, it checks whether today's run has already completed
  (tracked via cache_store, one flag per job per UTC calendar day) and, if
  the scheduled time has already passed and it hasn't, runs it immediately
  as a catch-up instead of waiting until tomorrow. This makes a late start
  (a reboot, a crash-restart, a laptop that was asleep at the scheduled
  time) self-heal instead of silently losing that day's alerts.

  Job C ("Starting XI Pull"), one-shot, kickoff - 60 minutes, per match:
    1. Pull GET /football/matches/{match_id}/lineups (via lineups.py).
    2. If the lineup isn't confirmed yet (TheStatsAPI 404s until the
       official team sheet is announced, "approximately 1 hour before
       kickoff" per its own docs -- not a guarantee), reschedule itself a
       few minutes later instead of silently giving up.
    3. Once confirmed, register the lineup on that competition's model
       (`SoccerProjectionModel.add_lineup`) so player-prop legs price off
       confirmed starting_xi/substitutes instead of historical
       minutes-per-appearance (see player_props_model.py), grade the FULL
       board (team markets + player props), and push the locked-in ticket
       to Telegram.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from cache_store import CACHE_DIR, read_cache, write_cache
from daily_model import build_models_for_matches, fetch_daily_matches
from ev_engine_core import build_match_ev_board, get_thestatsapi_key, load_env_file, thestatsapi_get
from lineups import LineupNotAvailable, fetch_match_lineup
from projection_model import SoccerProjectionModel
from telegram_notifier import format_ev_board_message, send_telegram_message

logger = logging.getLogger("soccer_engine.scheduler")

SCHEDULER_TIMEZONE = "America/New_York"
MORNING_PULL_HOUR = 8
MORNING_PULL_MINUTE = 0
EARLY_PROPS_SYNC_HOUR = 9
EARLY_PROPS_SYNC_MINUTE = 30
LINEUP_LEAD_TIME_MINUTES = 60
LINEUP_RETRY_MINUTES = 5
LINEUP_MAX_RETRIES = 10  # ~50 minutes of retrying a late-announced lineup before giving up
MIN_CONFIDENCE_SCORE_FOR_ALERT = 70

# One flag per UTC calendar day (same "today" run_morning_pull/
# run_early_props_sync themselves use to fetch the schedule) recording
# whether that day's run already completed -- see start()'s catch-up logic
# above.
MORNING_PULL_STATE_CACHE_CATEGORY = "morning_pull_state"
EARLY_PROPS_SYNC_STATE_CACHE_CATEGORY = "early_props_sync_state"

# Comma-separated TheStatsAPI competition_ids (e.g. "comp_408698,comp_6107")
# to scope EVERY daily job to -- unset means "every match scheduled
# worldwide today", which is what triggered sustained rate limiting in
# production (see daily_model.fetch_daily_matches's docstring). Set this
# in .env.local for a period where only specific competitions matter (a
# single tournament window) without touching code.
COMPETITION_IDS_ENV_VAR = "SOCCER_ENGINE_COMPETITION_IDS"

# Master kill-switch for player-prop grading, set in .env.local. Exists
# because grading a player-prop leg needs sportsbook odds for it (GET
# /football/matches/{id}/odds/players), which TheStatsAPI 403s
# (ADDON_REQUIRED) on any account without the player_odds add-on -- a plan
# limitation, not a transient failure. With this off, every job skips the
# roster + season-stats fan-out entirely (the bulk of this pipeline's
# request volume) and grades team markets only, instead of burning API
# quota fetching player data that can never be used to grade anything.
INCLUDE_PLAYER_PROPS_ENV_VAR = "SOCCER_ENGINE_INCLUDE_PLAYER_PROPS"


def _competition_ids_from_env() -> frozenset[str] | None:
    raw = os.getenv(COMPETITION_IDS_ENV_VAR)
    if not raw:
        return None
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return frozenset(ids) or None


def _include_player_props_from_env() -> bool:
    raw = os.getenv(INCLUDE_PLAYER_PROPS_ENV_VAR)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class SoccerEvScheduler:
    """
    Thin, testable wrapper around a BackgroundScheduler instance.

    State that would otherwise be module-level globals (today's fitted
    models, per-match lineup-retry counters) lives on the instance instead,
    so this can be exercised in tests without a live scheduler thread or
    real network calls -- construct it, call the job methods directly, and
    inspect `self.scheduler.get_jobs()` / mocked collaborators.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: Path = CACHE_DIR,
        competition_ids: Iterable[str] | None = None,
    ) -> None:
        load_env_file()  # defaults to the repo-root .env.local regardless of process cwd
        resolved_key = api_key or get_thestatsapi_key()
        if not resolved_key:
            raise RuntimeError("THESTATSAPI_KEY is required to run the scheduler (set it in .env.local).")
        self.api_key = resolved_key
        self.scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
        self._models_by_competition: dict[str, SoccerProjectionModel] = {}
        self._lineup_retry_counts: dict[str, int] = {}
        # Overridable purely so tests can point this at a throwaway tmp_path
        # instead of writing/reading engines/soccer_engine/cache/ for real
        # (see _catch_up_missed_morning_pull / _morning_pull_already_completed_today).
        self._cache_dir = cache_dir
        # None (the default, also the default when unset in .env.local)
        # means "every match scheduled worldwide today" -- see
        # daily_model.fetch_daily_matches's docstring for why scoping this
        # matters. Explicit constructor arg wins over the env var.
        self._competition_ids = (
            frozenset(str(cid) for cid in competition_ids) if competition_ids is not None else _competition_ids_from_env()
        )
        if self._competition_ids:
            logger.info("Scoped to competition_ids=%s (via %s).", sorted(self._competition_ids), COMPETITION_IDS_ENV_VAR)
        self._include_player_props = _include_player_props_from_env()
        if not self._include_player_props:
            logger.warning(
                "Player props disabled (via %s) -- grading team markets only "
                "and skipping the roster/season-stats fetches entirely.",
                INCLUDE_PLAYER_PROPS_ENV_VAR,
            )

    def start(self) -> None:
        """Registers the two recurring cron jobs and starts the background thread. Non-blocking."""

        self.scheduler.add_job(
            self.run_morning_pull,
            trigger=CronTrigger(hour=MORNING_PULL_HOUR, minute=MORNING_PULL_MINUTE, timezone=SCHEDULER_TIMEZONE),
            id="morning_pull",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self.scheduler.add_job(
            self.run_early_props_sync,
            trigger=CronTrigger(
                hour=EARLY_PROPS_SYNC_HOUR, minute=EARLY_PROPS_SYNC_MINUTE, timezone=SCHEDULER_TIMEZONE
            ),
            id="early_props_sync",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self.scheduler.start()
        logger.info(
            "Scheduler started (%s); morning pull set for %02d:%02d, early props sync set for %02d:%02d.",
            SCHEDULER_TIMEZONE,
            MORNING_PULL_HOUR,
            MORNING_PULL_MINUTE,
            EARLY_PROPS_SYNC_HOUR,
            EARLY_PROPS_SYNC_MINUTE,
        )
        # Order matters: the early-props catch-up doesn't strictly require
        # the morning pull to have completed (run_early_props_sync builds
        # its own models on demand via _fetch_and_ensure_models if needed),
        # but running the morning-pull catch-up first means the simpler
        # "Early Value" message goes out before the more detailed "Early
        # Props" one in the overwhelmingly common case where both are due.
        self._catch_up_missed_morning_pull()
        self._catch_up_missed_early_props_sync()

    def _catch_up_missed_morning_pull(self) -> None:
        """
        Self-healing for the "process wasn't running at 8 AM" failure mode
        (not started yet, machine was asleep, a crash) -- see the
        module-level docstring. A freshly-added CronTrigger only fires at
        the NEXT occurrence of 8:00 AM, so starting late otherwise means
        silently waiting until tomorrow with no alert explaining why.
        """

        self._catch_up_missed_job(
            already_completed=self._morning_pull_already_completed_today,
            scheduled_hour=MORNING_PULL_HOUR,
            scheduled_minute=MORNING_PULL_MINUTE,
            job_name="Morning pull",
            run=self.run_morning_pull,
        )

    def _catch_up_missed_early_props_sync(self) -> None:
        """Same self-healing as _catch_up_missed_morning_pull, for the 9:30 AM early props sync."""

        self._catch_up_missed_job(
            already_completed=self._early_props_sync_already_completed_today,
            scheduled_hour=EARLY_PROPS_SYNC_HOUR,
            scheduled_minute=EARLY_PROPS_SYNC_MINUTE,
            job_name="Early props sync",
            run=self.run_early_props_sync,
        )

    def _catch_up_missed_job(
        self,
        *,
        already_completed: Any,
        scheduled_hour: int,
        scheduled_minute: int,
        job_name: str,
        run: Any,
    ) -> None:
        if already_completed():
            return

        now_local = datetime.now(ZoneInfo(SCHEDULER_TIMEZONE))
        todays_run_time = now_local.replace(hour=scheduled_hour, minute=scheduled_minute, second=0, microsecond=0)
        if now_local < todays_run_time:
            return  # scheduled time hasn't happened yet today -- the cron job will fire normally

        logger.warning(
            "%s for today hasn't run yet and %02d:%02d %s has already passed "
            "(process likely wasn't running at the scheduled time) -- catching up now.",
            job_name,
            scheduled_hour,
            scheduled_minute,
            SCHEDULER_TIMEZONE,
        )
        run()

    def _morning_pull_already_completed_today(self) -> bool:
        today = datetime.now(timezone.utc).date()
        return bool(read_cache(MORNING_PULL_STATE_CACHE_CATEGORY, today, cache_dir=self._cache_dir).get("completed"))

    def _mark_morning_pull_completed_today(self) -> None:
        today = datetime.now(timezone.utc).date()
        write_cache(
            MORNING_PULL_STATE_CACHE_CATEGORY,
            today,
            {"completed": True, "completed_at": datetime.now(timezone.utc).isoformat()},
            cache_dir=self._cache_dir,
        )

    def _early_props_sync_already_completed_today(self) -> bool:
        today = datetime.now(timezone.utc).date()
        return bool(
            read_cache(EARLY_PROPS_SYNC_STATE_CACHE_CATEGORY, today, cache_dir=self._cache_dir).get("completed")
        )

    def _mark_early_props_sync_completed_today(self) -> None:
        today = datetime.now(timezone.utc).date()
        write_cache(
            EARLY_PROPS_SYNC_STATE_CACHE_CATEGORY,
            today,
            {"completed": True, "completed_at": datetime.now(timezone.utc).isoformat()},
            cache_dir=self._cache_dir,
        )

    def shutdown(self, *, wait: bool = True) -> None:
        self.scheduler.shutdown(wait=wait)

    def _fetch_and_ensure_models(self, today: Any) -> list[dict[str, Any]]:
        """
        Fetches today's match schedule and, if `self._models_by_competition`
        isn't already populated for this process, builds it.

        Shared by both run_morning_pull and run_early_props_sync so the
        9:30 AM job doesn't need the 8:00 AM job to have run in THIS same
        process lifetime to have models available -- e.g. a process
        restart between the two that leaves the in-memory
        `_models_by_competition` empty even though the morning pull's
        on-disk "completed" flag says today's fetch already happened.
        Cheap to call more than once: model building is skipped if
        `_models_by_competition` is already populated, and the underlying
        per-competition/team/player fetches are themselves cache-backed
        (cache_store.py), so even a from-scratch rebuild only re-does calls
        that weren't already cached earlier today.

        Raises on a failed schedule fetch -- callers decide how to handle
        that (specifically: never mark their own job "completed" on a
        failure, so a later restart retries instead of assuming success).
        """

        matches = fetch_daily_matches(self.api_key, today, competition_ids=self._competition_ids)
        if not self._models_by_competition and matches:
            self._models_by_competition = build_models_for_matches(
                self.api_key,
                matches,
                cache_dir=self._cache_dir,
                build_player_profiles=self._include_player_props,
            )
        return matches

    # ------------------------------------------------------------------
    # Job A: the morning pull
    # ------------------------------------------------------------------
    def run_morning_pull(self) -> None:
        today = datetime.now(timezone.utc).date()
        logger.info("Morning pull starting for %s", today.isoformat())

        try:
            matches = self._fetch_and_ensure_models(today)
        except Exception:
            logger.exception("Morning pull failed to fetch today's schedule")
            return  # do NOT mark complete -- a later restart should retry this catch-up

        # From here on the day's schedule is known (even if empty) -- mark
        # complete now so a later restart/catch-up this same day doesn't
        # re-fetch and re-send a duplicate "Early Value" message.
        self._mark_morning_pull_completed_today()

        if not matches:
            logger.info("No matches scheduled for %s; skipping.", today.isoformat())
            return

        early_results = []
        for match in matches:
            model = self._models_by_competition.get(match.get("competition_id"))
            if model is None:
                logger.info(
                    "No fitted model for competition %s; skipping early board for match %s.",
                    match.get("competition_id"),
                    match.get("id"),
                )
            else:
                early_results.extend(self._grade_board(match, model, include_player_props=False))

            self._schedule_lineup_check(match)

        message = format_ev_board_message(
            early_results, title="\U0001F4CA Early Value Board", min_confidence_score=MIN_CONFIDENCE_SCORE_FOR_ALERT
        )
        self._send_telegram_safely(message)

    # ------------------------------------------------------------------
    # Job B: the early props sync
    # ------------------------------------------------------------------
    def run_early_props_sync(self) -> None:
        """
        Grades the FULL board (team markets + player props) for every match
        fitted so far today, off whatever data is currently available --
        historical minutes-per-appearance if no lineup has been confirmed
        yet, which is the normal case this early. See the module docstring
        for why this exists as its own fixed-time job instead of just
        waiting for Job C's confirmed-lineup pass.
        """

        today = datetime.now(timezone.utc).date()
        logger.info("Early props sync starting for %s", today.isoformat())

        try:
            matches = self._fetch_and_ensure_models(today)
        except Exception:
            logger.exception("Early props sync failed to fetch today's schedule")
            return  # do NOT mark complete -- a later restart should retry this catch-up

        self._mark_early_props_sync_completed_today()

        if not matches:
            logger.info("No matches scheduled for %s; skipping early props sync.", today.isoformat())
            return

        full_results = []
        for match in matches:
            model = self._models_by_competition.get(match.get("competition_id"))
            if model is None:
                logger.info(
                    "No fitted model for competition %s; skipping early props for match %s.",
                    match.get("competition_id"),
                    match.get("id"),
                )
                continue
            full_results.extend(self._grade_board(match, model, include_player_props=self._include_player_props))

        message = format_ev_board_message(
            full_results,
            title="\U0001F31F Early Props Board (provisional)",
            min_confidence_score=MIN_CONFIDENCE_SCORE_FOR_ALERT,
        )
        self._send_telegram_safely(message)

    def _grade_board(self, match: dict[str, Any], model: SoccerProjectionModel, *, include_player_props: bool) -> list[Any]:
        match_id = match.get("id")
        try:
            return build_match_ev_board(self.api_key, match_id, model, include_player_props=include_player_props)
        except Exception:
            logger.exception("Failed to grade board for match %s (include_player_props=%s)", match_id, include_player_props)
            return []

    def _schedule_lineup_check(self, match: dict[str, Any]) -> None:
        match_id = match.get("id")
        kickoff = _parse_utc_datetime(match.get("utc_date"))
        if not match_id or kickoff is None:
            logger.warning("Skipping lineup-check scheduling for match with missing id/kickoff: %r", match)
            return

        run_at = kickoff - timedelta(minutes=LINEUP_LEAD_TIME_MINUTES)
        now = datetime.now(timezone.utc)
        if run_at <= now:
            # Kickoff is already under an hour away (or in the past) by the
            # time the morning pull ran for this match -- run the lineup
            # check almost immediately instead of scheduling it for a
            # moment that's already gone.
            run_at = now + timedelta(seconds=5)

        self.scheduler.add_job(
            self.run_starting_xi_check,
            trigger=DateTrigger(run_date=run_at),
            args=[match_id],
            id=f"lineup_check_{match_id}",
            replace_existing=True,
            misfire_grace_time=1800,
        )
        logger.info(
            "Scheduled lineup check for match %s at %s (kickoff %s).",
            match_id,
            run_at.isoformat(),
            kickoff.isoformat(),
        )

    # ------------------------------------------------------------------
    # Job B: the starting XI pull
    # ------------------------------------------------------------------
    def run_starting_xi_check(self, match_id: str) -> None:
        try:
            lineup = fetch_match_lineup(self.api_key, match_id)
        except LineupNotAvailable:
            self._retry_or_give_up(match_id, reason="lineup not announced yet (404)")
            return
        except Exception:
            logger.exception("Failed to fetch lineups for match %s", match_id)
            return

        if not lineup.confirmed:
            self._retry_or_give_up(match_id, reason="lineup present but not yet confirmed")
            return

        model = self._model_for_match_id(match_id)
        if model is None:
            logger.warning("No fitted model available for match %s; skipping final board.", match_id)
            return

        # From this point on, any player-prop leg for this match will be
        # priced off the confirmed starting_xi/substitutes rather than
        # historical minutes-per-appearance -- see
        # player_props_model.expected_minutes_factor.
        model.add_lineup(lineup)

        try:
            results = build_match_ev_board(
                self.api_key, match_id, model, include_player_props=self._include_player_props
            )
        except Exception:
            logger.exception("Failed to grade final board for match %s", match_id)
            return

        message = format_ev_board_message(
            results, title="\U0001F512 Locked-In Ticket", min_confidence_score=MIN_CONFIDENCE_SCORE_FOR_ALERT
        )
        self._send_telegram_safely(message)
        self._lineup_retry_counts.pop(match_id, None)

    def _retry_or_give_up(self, match_id: str, *, reason: str) -> None:
        attempts = self._lineup_retry_counts.get(match_id, 0)
        if attempts >= LINEUP_MAX_RETRIES:
            logger.warning(
                "Giving up on lineup check for match %s after %d attempts (%s).", match_id, attempts, reason
            )
            return

        self._lineup_retry_counts[match_id] = attempts + 1
        retry_at = datetime.now(timezone.utc) + timedelta(minutes=LINEUP_RETRY_MINUTES)
        self.scheduler.add_job(
            self.run_starting_xi_check,
            trigger=DateTrigger(run_date=retry_at),
            args=[match_id],
            id=f"lineup_check_{match_id}",
            replace_existing=True,
            misfire_grace_time=600,
        )
        logger.info("Retrying lineup check for match %s at %s (%s).", match_id, retry_at.isoformat(), reason)

    def _model_for_match_id(self, match_id: str) -> SoccerProjectionModel | None:
        # build_match_ev_board fetches its own match detail internally, so
        # the competition_id used to fit this morning's models isn't
        # attached to a bare match_id here -- resolve it with one cheap
        # lookup rather than threading extra state through every job.
        try:
            match_payload = thestatsapi_get(f"/football/matches/{match_id}", self.api_key)
        except Exception:
            logger.exception("Failed to resolve competition for match %s", match_id)
            return None

        competition_id = match_payload.get("data", match_payload).get("competition_id")
        return self._models_by_competition.get(competition_id)

    def _send_telegram_safely(self, message: str) -> None:
        try:
            send_telegram_message(message)
        except Exception:
            logger.exception("Failed to send Telegram message")


def _parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def start_scheduler(api_key: str | None = None) -> SoccerEvScheduler:
    """Convenience entrypoint: build + start a scheduler, returning it so the caller can shut it down later."""

    instance = SoccerEvScheduler(api_key=api_key)
    instance.start()
    return instance


if __name__ == "__main__":
    import sys
    import time

    # File logging (in addition to stderr, when one exists) matters here
    # specifically because this is meant to run headless/unattended (e.g.
    # launched silently via pythonw.exe at login, or a Task Scheduler entry
    # with no visible console) -- without it, a silent failure would be
    # just as invisible as "the process never started" was.
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_handlers: list[logging.Handler] = [logging.FileHandler(LOG_DIR / "scheduler.log", encoding="utf-8")]
    # pythonw.exe (used for a silent, windowless launch) sets sys.stderr to
    # None -- a plain StreamHandler would crash the first time it logs.
    if sys.stderr is not None:
        log_handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=log_handlers,
    )
    running_scheduler = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        running_scheduler.shutdown()
