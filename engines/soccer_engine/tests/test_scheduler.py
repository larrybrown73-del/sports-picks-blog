from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.date import DateTrigger

import scheduler as scheduler_module
from cache_store import read_cache
from lineups import LineupNotAvailable, MatchLineup
from scheduler import (
    LINEUP_LEAD_TIME_MINUTES,
    LINEUP_MAX_RETRIES,
    MORNING_PULL_STATE_CACHE_CATEGORY,
    SCHEDULER_TIMEZONE,
    SoccerEvScheduler,
)


class _FixedDateTime(datetime):
    """Subclass swapped in for scheduler_module.datetime so `datetime.now(tz)` returns a controlled instant."""

    _fixed: datetime

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._fixed
        return cls._fixed.astimezone(tz)


def _freeze_time(monkeypatch: pytest.MonkeyPatch, ny_time: datetime) -> None:
    frozen = _FixedDateTime
    frozen._fixed = ny_time.astimezone(timezone.utc)
    monkeypatch.setattr(scheduler_module, "datetime", frozen)


class _FakeModel:
    """Stand-in for SoccerProjectionModel that just records add_lineup calls."""

    def __init__(self) -> None:
        self.added_lineups: list[MatchLineup] = []

    def add_lineup(self, lineup: MatchLineup) -> None:
        self.added_lineups.append(lineup)


class _CapturingScheduler:
    """Stand-in for BackgroundScheduler that just records add_job calls."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.started = False
        self.shut_down = False

    def add_job(self, func, trigger=None, args=None, id=None, replace_existing=False, misfire_grace_time=None):
        self.jobs[id] = {"func": func, "trigger": trigger, "args": args or []}

    def start(self) -> None:
        self.started = True

    def shutdown(self, wait: bool = True) -> None:
        self.shut_down = True


@pytest.fixture(autouse=True)
def _isolate_from_real_env_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    SoccerEvScheduler.__init__ calls load_env_file(), which reads the REAL
    repo-root .env.local and only fills in a var if it ISN'T already in
    os.environ -- so these tests must not be at the mercy of whatever
    operators have set there (e.g. SOCCER_ENGINE_COMPETITION_IDS scoped to
    a live tournament) for their expected defaults to hold. Pre-seeding an
    empty value (rather than delenv, which would leave the var missing and
    let load_env_file fill it back in from the file) blocks that fallback
    while still parsing as "unset" to _competition_ids_from_env /
    _include_player_props_from_env.
    """

    monkeypatch.setenv("SOCCER_ENGINE_COMPETITION_IDS", "")
    monkeypatch.setenv("SOCCER_ENGINE_INCLUDE_PLAYER_PROPS", "")


@pytest.fixture()
def sched(tmp_path: Path) -> SoccerEvScheduler:
    instance = SoccerEvScheduler(api_key="test_key", cache_dir=tmp_path)
    instance.scheduler = _CapturingScheduler()
    # Isolates existing tests from the new (real-clock-dependent) catch-up
    # behavior added to start() -- see the dedicated
    # test_catch_up_missed_*_* tests below for that logic.
    instance._catch_up_missed_morning_pull = lambda: None
    instance._catch_up_missed_early_props_sync = lambda: None
    return instance


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def test_scheduler_include_player_props_defaults_to_true(sched: SoccerEvScheduler) -> None:
    assert sched._include_player_props is True


def test_scheduler_include_player_props_disabled_via_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCCER_ENGINE_INCLUDE_PLAYER_PROPS", "false")
    instance = SoccerEvScheduler(api_key="test_key", cache_dir=tmp_path)
    assert instance._include_player_props is False


def test_run_early_props_sync_respects_include_player_props_false(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    sched._include_player_props = False
    fake_model = object()
    monkeypatch.setattr(
        scheduler_module,
        "fetch_daily_matches",
        lambda api_key, today, **kw: [{"id": "mt_1", "competition_id": "comp_1"}],
    )
    monkeypatch.setattr(
        scheduler_module, "build_models_for_matches", lambda api_key, matches, **kw: {"comp_1": fake_model}
    )

    graded_calls = []
    monkeypatch.setattr(
        scheduler_module,
        "build_match_ev_board",
        lambda api_key, match_id, model, *, include_player_props: graded_calls.append(include_player_props)
        or ["fake_result"],
    )
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: None)
    monkeypatch.setattr(
        scheduler_module, "format_ev_board_message", lambda results, *, title, min_confidence_score: title
    )

    sched.run_early_props_sync()

    assert graded_calls == [False]


def test_fetch_and_ensure_models_passes_build_player_profiles_flag(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    sched._include_player_props = False
    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: [{"id": "mt_1"}])

    captured = {}

    def fake_build(api_key, matches, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(scheduler_module, "build_models_for_matches", fake_build)

    sched._fetch_and_ensure_models(datetime.now(timezone.utc).date())

    assert captured.get("build_player_profiles") is False


def test_scheduler_competition_ids_defaults_to_none(sched: SoccerEvScheduler) -> None:
    assert sched._competition_ids is None  # unscoped -- grades every match scheduled worldwide today


def test_scheduler_competition_ids_from_constructor(tmp_path: Path) -> None:
    instance = SoccerEvScheduler(api_key="test_key", cache_dir=tmp_path, competition_ids=["comp_1", "comp_2"])
    assert instance._competition_ids == frozenset({"comp_1", "comp_2"})


def test_scheduler_competition_ids_from_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCCER_ENGINE_COMPETITION_IDS", "comp_world_cup, comp_other ,")
    instance = SoccerEvScheduler(api_key="test_key", cache_dir=tmp_path)
    assert instance._competition_ids == frozenset({"comp_world_cup", "comp_other"})


def test_scheduler_constructor_arg_overrides_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOCCER_ENGINE_COMPETITION_IDS", "comp_from_env")
    instance = SoccerEvScheduler(api_key="test_key", cache_dir=tmp_path, competition_ids=["comp_from_arg"])
    assert instance._competition_ids == frozenset({"comp_from_arg"})


def test_fetch_and_ensure_models_passes_competition_ids_through(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    sched._competition_ids = frozenset({"comp_1"})
    captured = {}

    def fake_fetch(api_key, today, **kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", fake_fetch)

    sched._fetch_and_ensure_models(datetime.now(timezone.utc).date())

    assert captured.get("competition_ids") == frozenset({"comp_1"})


def test_start_registers_morning_cron_job_and_starts_scheduler(sched: SoccerEvScheduler) -> None:
    sched.start()
    assert "morning_pull" in sched.scheduler.jobs
    assert "early_props_sync" in sched.scheduler.jobs
    assert sched.scheduler.started is True


def test_schedule_lineup_check_computes_kickoff_minus_lead_time(sched: SoccerEvScheduler) -> None:
    kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
    match = {"id": "mt_1", "utc_date": _iso(kickoff)}

    sched._schedule_lineup_check(match)

    job = sched.scheduler.jobs["lineup_check_mt_1"]
    assert job["args"] == ["mt_1"]
    assert isinstance(job["trigger"], DateTrigger)
    expected_run_date = kickoff - timedelta(minutes=LINEUP_LEAD_TIME_MINUTES)
    assert abs((job["trigger"].run_date - expected_run_date).total_seconds()) < 1


def test_schedule_lineup_check_runs_almost_immediately_if_kickoff_already_near(sched: SoccerEvScheduler) -> None:
    kickoff = datetime.now(timezone.utc) + timedelta(minutes=10)  # under the 60-minute lead time
    match = {"id": "mt_2", "utc_date": _iso(kickoff)}

    sched._schedule_lineup_check(match)

    run_date = sched.scheduler.jobs["lineup_check_mt_2"]["trigger"].run_date
    now = datetime.now(timezone.utc)
    assert now < run_date < now + timedelta(minutes=1)


def test_schedule_lineup_check_skips_matches_missing_id_or_kickoff(sched: SoccerEvScheduler) -> None:
    sched._schedule_lineup_check({"id": None, "utc_date": "2026-01-01T00:00:00.000Z"})
    sched._schedule_lineup_check({"id": "mt_3", "utc_date": None})
    assert sched.scheduler.jobs == {}


def test_run_morning_pull_grades_matches_and_schedules_lineup_checks(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
    fake_matches = [{"id": "mt_1", "competition_id": "comp_1", "utc_date": _iso(kickoff)}]
    fake_model = object()

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: fake_matches)
    monkeypatch.setattr(
        scheduler_module,
        "build_models_for_matches",
        lambda api_key, matches, **kw: {"comp_1": fake_model},
    )

    graded_calls = []

    def fake_build_board(api_key, match_id, model, *, include_player_props):
        graded_calls.append((match_id, model, include_player_props))
        return ["fake_result"]

    monkeypatch.setattr(scheduler_module, "build_match_ev_board", fake_build_board)

    sent_messages = []
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: sent_messages.append(text))
    monkeypatch.setattr(
        scheduler_module,
        "format_ev_board_message",
        lambda results, *, title, min_confidence_score: f"{title}:{len(results)}",
    )

    sched.run_morning_pull()

    assert graded_calls == [("mt_1", fake_model, False)]
    assert "lineup_check_mt_1" in sched.scheduler.jobs
    assert sent_messages == ["\U0001F4CA Early Value Board:1"]


def test_run_morning_pull_skips_matches_with_no_fitted_model(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    kickoff = datetime.now(timezone.utc) + timedelta(hours=5)
    fake_matches = [{"id": "mt_1", "competition_id": "comp_unmodeled", "utc_date": _iso(kickoff)}]

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: fake_matches)
    monkeypatch.setattr(
        scheduler_module, "build_models_for_matches", lambda api_key, matches, **kw: {}
    )

    called = []
    monkeypatch.setattr(scheduler_module, "build_match_ev_board", lambda *a, **k: called.append(1))
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: None)
    monkeypatch.setattr(
        scheduler_module, "format_ev_board_message", lambda results, *, title, min_confidence_score: title
    )

    sched.run_morning_pull()

    assert called == []  # never graded -- no model for that competition
    assert "lineup_check_mt_1" in sched.scheduler.jobs  # still scheduled for a lineup check later


def test_run_morning_pull_handles_empty_schedule_gracefully(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: [])
    sent = []
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: sent.append(text))

    sched.run_morning_pull()

    assert sent == []  # no matches today -- no message, no crash
    assert sched.scheduler.jobs == {}


def test_run_early_props_sync_grades_full_board_for_every_match(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_matches = [{"id": "mt_1", "competition_id": "comp_1"}]
    fake_model = object()

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: fake_matches)
    monkeypatch.setattr(
        scheduler_module,
        "build_models_for_matches",
        lambda api_key, matches, **kw: {"comp_1": fake_model},
    )

    graded_calls = []

    def fake_build_board(api_key, match_id, model, *, include_player_props):
        graded_calls.append((match_id, model, include_player_props))
        return ["fake_result"]

    monkeypatch.setattr(scheduler_module, "build_match_ev_board", fake_build_board)

    sent_messages = []
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: sent_messages.append(text))
    monkeypatch.setattr(
        scheduler_module,
        "format_ev_board_message",
        lambda results, *, title, min_confidence_score: f"{title}:{len(results)}",
    )

    sched.run_early_props_sync()

    assert graded_calls == [("mt_1", fake_model, True)]  # full board, unlike the morning pull's team-only grading
    assert len(sent_messages) == 1
    assert "Early Props Board" in sent_messages[0]


def test_run_early_props_sync_reuses_models_already_built_this_process(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    If the morning pull already ran in this same process (the normal case
    -- Job A always fires before Job B), Job B must not rebuild models
    from scratch.
    """

    fake_model = object()
    sched._models_by_competition["comp_1"] = fake_model

    monkeypatch.setattr(
        scheduler_module,
        "fetch_daily_matches",
        lambda api_key, today, **kw: [{"id": "mt_1", "competition_id": "comp_1"}],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_models_for_matches should not be called -- models already built this process")

    monkeypatch.setattr(scheduler_module, "build_models_for_matches", fail_if_called)
    monkeypatch.setattr(scheduler_module, "build_match_ev_board", lambda *a, **k: ["fake_result"])
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: None)
    monkeypatch.setattr(
        scheduler_module, "format_ev_board_message", lambda results, *, title, min_confidence_score: title
    )

    sched.run_early_props_sync()  # must not raise


def test_run_early_props_sync_skips_matches_with_no_fitted_model(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "fetch_daily_matches",
        lambda api_key, today, **kw: [{"id": "mt_1", "competition_id": "comp_unmodeled"}],
    )
    monkeypatch.setattr(scheduler_module, "build_models_for_matches", lambda api_key, matches, **kw: {})

    called = []
    monkeypatch.setattr(scheduler_module, "build_match_ev_board", lambda *a, **k: called.append(1))
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: None)
    monkeypatch.setattr(
        scheduler_module, "format_ev_board_message", lambda results, *, title, min_confidence_score: title
    )

    sched.run_early_props_sync()

    assert called == []  # never graded -- no model for that competition


def test_run_early_props_sync_handles_empty_schedule_gracefully(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: [])
    sent = []
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: sent.append(text))

    sched.run_early_props_sync()

    assert sent == []  # no matches today -- no message, no crash
    assert sched._early_props_sync_already_completed_today() is True  # still marked -- the day's schedule is known


def test_run_early_props_sync_marks_state_completed_on_success(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: [])

    assert sched._early_props_sync_already_completed_today() is False
    sched.run_early_props_sync()
    assert sched._early_props_sync_already_completed_today() is True


def test_run_early_props_sync_does_not_mark_completed_on_fetch_failure(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_error(api_key, today, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", raise_error)

    sched.run_early_props_sync()

    assert sched._early_props_sync_already_completed_today() is False


def test_run_starting_xi_check_retries_on_404(sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_found(api_key: str, match_id: str):
        raise LineupNotAvailable("not announced yet")

    monkeypatch.setattr(scheduler_module, "fetch_match_lineup", raise_not_found)

    sched.run_starting_xi_check("mt_1")

    assert sched._lineup_retry_counts["mt_1"] == 1
    assert "lineup_check_mt_1" in sched.scheduler.jobs


def test_run_starting_xi_check_gives_up_after_max_retries(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_not_found(api_key: str, match_id: str):
        raise LineupNotAvailable("not announced yet")

    monkeypatch.setattr(scheduler_module, "fetch_match_lineup", raise_not_found)
    sched._lineup_retry_counts["mt_1"] = LINEUP_MAX_RETRIES

    sched.run_starting_xi_check("mt_1")

    assert sched._lineup_retry_counts["mt_1"] == LINEUP_MAX_RETRIES  # unchanged
    assert "lineup_check_mt_1" not in sched.scheduler.jobs  # gave up, nothing rescheduled


def test_run_starting_xi_check_retries_when_lineup_not_yet_confirmed(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    unconfirmed = MatchLineup(
        match_id="mt_2",
        confirmed=False,
        starting_player_ids=frozenset(),
        substitute_player_ids=frozenset(),
        home_team_id="t_home",
        away_team_id="t_away",
    )
    monkeypatch.setattr(scheduler_module, "fetch_match_lineup", lambda api_key, match_id: unconfirmed)

    sched.run_starting_xi_check("mt_2")

    assert sched._lineup_retry_counts["mt_2"] == 1
    assert "lineup_check_mt_2" in sched.scheduler.jobs


def test_run_starting_xi_check_grades_full_board_and_sends_when_confirmed(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_model = _FakeModel()
    sched._models_by_competition["comp_1"] = fake_model
    sched._lineup_retry_counts["mt_1"] = 2

    confirmed_lineup = MatchLineup(
        match_id="mt_1",
        confirmed=True,
        starting_player_ids=frozenset({"p_1"}),
        substitute_player_ids=frozenset({"p_2"}),
        home_team_id="t_home",
        away_team_id="t_away",
    )
    monkeypatch.setattr(scheduler_module, "fetch_match_lineup", lambda api_key, match_id: confirmed_lineup)
    monkeypatch.setattr(
        scheduler_module, "thestatsapi_get", lambda path, api_key: {"data": {"competition_id": "comp_1"}}
    )

    graded_calls = []

    def fake_build_board(api_key, match_id, model, *, include_player_props):
        graded_calls.append((match_id, model, include_player_props))
        return ["result"]

    monkeypatch.setattr(scheduler_module, "build_match_ev_board", fake_build_board)

    sent = []
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: sent.append(text))
    monkeypatch.setattr(
        scheduler_module, "format_ev_board_message", lambda results, *, title, min_confidence_score: title
    )

    sched.run_starting_xi_check("mt_1")

    assert fake_model.added_lineups == [confirmed_lineup]  # lineup wired into the model before grading
    assert graded_calls == [("mt_1", fake_model, True)]
    assert sent == ["\U0001F512 Locked-In Ticket"]
    assert "mt_1" not in sched._lineup_retry_counts


def test_run_morning_pull_marks_state_completed_on_success(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", lambda api_key, today, **kw: [])
    monkeypatch.setattr(scheduler_module, "send_telegram_message", lambda text: None)

    assert sched._morning_pull_already_completed_today() is False
    sched.run_morning_pull()
    assert sched._morning_pull_already_completed_today() is True


def test_run_morning_pull_does_not_mark_completed_on_fetch_failure(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_error(api_key, today, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(scheduler_module, "fetch_daily_matches", raise_error)

    sched.run_morning_pull()

    # A failed fetch must leave the day un-marked so a later restart retries
    # the catch-up instead of assuming the day was already handled.
    assert sched._morning_pull_already_completed_today() is False


def _use_real_catch_up_logic(sched: SoccerEvScheduler, method_name: str) -> None:
    """Undo the `sched` fixture's default no-op stub -- these tests exist specifically to exercise the real logic."""

    del sched.__dict__[method_name]


def test_catch_up_missed_morning_pull_skips_if_already_completed_today(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_morning_pull")
    # Freeze time BEFORE marking complete: _mark_morning_pull_completed_today
    # writes its cache entry keyed off datetime.now() at call time, so
    # marking "today" complete using the real wall clock and then checking
    # against a frozen, different "today" would never match -- the mark and
    # the later self-heal check must agree on what day it is.
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 9, 0, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))
    sched._mark_morning_pull_completed_today()

    called = []
    monkeypatch.setattr(sched, "run_morning_pull", lambda: called.append(1))

    sched._catch_up_missed_morning_pull()

    assert called == []


def test_catch_up_missed_morning_pull_skips_before_scheduled_time(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_morning_pull")
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 7, 0, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))

    called = []
    monkeypatch.setattr(sched, "run_morning_pull", lambda: called.append(1))

    sched._catch_up_missed_morning_pull()

    assert called == []  # 8 AM hasn't happened yet today -- let the cron job handle it normally


def test_catch_up_missed_morning_pull_runs_immediately_if_window_already_passed(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_morning_pull")
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 9, 30, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))

    called = []
    monkeypatch.setattr(sched, "run_morning_pull", lambda: called.append(1))

    sched._catch_up_missed_morning_pull()

    assert called == [1]  # process started late (e.g. after a reboot) -- self-heal instead of waiting until tomorrow


def test_catch_up_missed_early_props_sync_skips_if_already_completed_today(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_early_props_sync")
    # Freeze time BEFORE marking complete -- see the analogous comment in
    # test_catch_up_missed_morning_pull_skips_if_already_completed_today.
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 10, 0, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))
    sched._mark_early_props_sync_completed_today()

    called = []
    monkeypatch.setattr(sched, "run_early_props_sync", lambda: called.append(1))

    sched._catch_up_missed_early_props_sync()

    assert called == []


def test_catch_up_missed_early_props_sync_skips_before_scheduled_time(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_early_props_sync")
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 9, 0, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))

    called = []
    monkeypatch.setattr(sched, "run_early_props_sync", lambda: called.append(1))

    sched._catch_up_missed_early_props_sync()

    assert called == []  # 9:30 AM hasn't happened yet today -- let the cron job handle it normally


def test_catch_up_missed_early_props_sync_runs_immediately_if_window_already_passed(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_real_catch_up_logic(sched, "_catch_up_missed_early_props_sync")
    _freeze_time(monkeypatch, datetime(2026, 7, 7, 10, 15, tzinfo=ZoneInfo(SCHEDULER_TIMEZONE)))

    called = []
    monkeypatch.setattr(sched, "run_early_props_sync", lambda: called.append(1))

    sched._catch_up_missed_early_props_sync()

    assert called == [1]  # process started late -- self-heal instead of waiting until tomorrow


def test_start_triggers_catch_up_after_registering_cron_job(
    sched: SoccerEvScheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Undo the fixture's default no-op stubs for this one test -- this is
    # exactly what it exists to verify.
    morning_pull_calls = []
    early_props_calls = []
    sched._catch_up_missed_morning_pull = lambda: morning_pull_calls.append(1)
    sched._catch_up_missed_early_props_sync = lambda: early_props_calls.append(1)

    sched.start()

    assert "morning_pull" in sched.scheduler.jobs
    assert "early_props_sync" in sched.scheduler.jobs
    assert morning_pull_calls == [1]
    assert early_props_calls == [1]
