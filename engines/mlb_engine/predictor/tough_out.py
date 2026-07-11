"""Tough Out lineups, innings-eater luck, and pre-All-Star motivation factors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import statsapi

from config import (
    ALL_STAR_BREAK_START,
    BREAK_PUSH_BONUS,
    BREAK_PUSH_WIN_PCT_MIN,
    CONTACT_STARTER_K_BB_PCT,
    FIP_CONSTANT,
    GRITTY_OFFENSE_SCALAR,
    INNINGS_EATER_ERA_MAX,
    INNINGS_EATER_ERA_MIN,
    INNINGS_EATER_MIN_IP,
    LOOK_AHEAD_BOTTOM_OPP_N,
    LOOK_AHEAD_NEXT_TOP_N,
    LOOK_AHEAD_SCHEDULE_HORIZON_DAYS,
    LOOK_AHEAD_TOP_WIN_PCT_N,
    LOOK_AHEAD_TRAP_PENALTY,
    LUCK_ERA_FIP_GAP,
    LUCK_REGRESSION_PENALTY,
    PRE_ALL_STAR_WINDOW_DAYS,
    TOUGH_OUT_CONTACT_TOP_N,
    TOUGH_OUT_MIN_PA,
    TOUGH_OUT_SLG_EXCLUDE_BOTTOM_N,
    TOUGH_OUT_SLG_EXCLUDE_TOP_N,
    TOUGH_OUT_WHIFF_TOP_N,
    VACATION_MODE_PENALTY,
    VACATION_MODE_WIN_PCT_MAX,
)
from data_health import safe_feature_fetch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamContactProfile:
    team_id: int
    contact_pct: float | None
    whiff_pct: float | None
    slugging: float | None
    plate_appearances: float = 0.0


@dataclass(frozen=True)
class InningsEaterProfile:
    pitcher_id: int
    k_bb_pct: float | None
    era: float | None
    fip: float | None
    innings_pitched: float | None
    is_innings_eater: bool = False


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_innings_pitched(ip_value: object) -> float | None:
    if ip_value is None:
        return None
    text = str(ip_value).strip()
    if not text:
        return None
    if "." not in text:
        return float(text)
    whole, frac = text.split(".", 1)
    return float(whole) + float(frac[:1] or 0) / 3.0


def compute_fip(
    *,
    home_runs: float,
    walks: float,
    hit_by_pitch: float,
    strikeouts: float,
    innings_pitched: float,
    constant: float = FIP_CONSTANT,
) -> float | None:
    """Standard FIP: ((13*HR + 3*(BB+HBP) - 2*K) / IP) + constant."""
    if innings_pitched <= 0:
        return None
    return (
        (13.0 * home_runs + 3.0 * (walks + hit_by_pitch) - 2.0 * strikeouts)
        / innings_pitched
    ) + constant


def all_star_break_start(season: int) -> date | None:
    raw = ALL_STAR_BREAK_START.get(season)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def is_pre_all_star_window(game_date: date, *, season: int | None = None) -> bool:
    """True when game_date falls within PRE_ALL_STAR_WINDOW_DAYS before the break."""
    season = season or game_date.year
    break_start = all_star_break_start(season)
    if break_start is None:
        return False
    window_open = break_start - timedelta(days=PRE_ALL_STAR_WINDOW_DAYS)
    return window_open <= game_date < break_start


def _contact_metrics_from_stat(stat: dict[str, Any]) -> tuple[float | None, float | None, float | None, float]:
    at_bats = _safe_float(stat.get("atBats")) or 0.0
    strikeouts = _safe_float(stat.get("strikeOuts")) or 0.0
    plate_appearances = _safe_float(stat.get("plateAppearances")) or 0.0
    slugging = _safe_float(stat.get("sluggingPercentage"))
    if slugging is None:
        slugging = _safe_float(stat.get("slg"))

    contact_pct = None
    if at_bats > 0:
        contact_pct = 100.0 * (at_bats - strikeouts) / at_bats

    # Whiff% proxy from K% (swings-and-misses track tightly with team K rate).
    whiff_pct = None
    if plate_appearances > 0:
        whiff_pct = 100.0 * strikeouts / plate_appearances

    return contact_pct, whiff_pct, slugging, plate_appearances


@lru_cache(maxsize=8)
def _league_team_contact_profiles(season: int) -> tuple[TeamContactProfile, ...]:
    def _fetch() -> tuple[TeamContactProfile, ...]:
        payload = statsapi.get(
            "teams_stats",
            {
                "season": season,
                "group": "hitting",
                "stats": "season",
                "sportIds": 1,
            },
        )
        profiles: list[TeamContactProfile] = []
        for block in payload.get("stats", []):
            for split in block.get("splits") or []:
                team = split.get("team") or {}
                team_id = team.get("id")
                if team_id is None:
                    continue
                contact_pct, whiff_pct, slugging, pa = _contact_metrics_from_stat(
                    split.get("stat") or {}
                )
                profiles.append(
                    TeamContactProfile(
                        team_id=int(team_id),
                        contact_pct=contact_pct,
                        whiff_pct=whiff_pct,
                        slugging=slugging,
                        plate_appearances=pa,
                    )
                )
        return tuple(profiles)

    return safe_feature_fetch(
        f"league_team_contact_{season}",
        _fetch,
        fallback=tuple(),
    )


@lru_cache(maxsize=8)
def tough_out_team_ids(season: int) -> frozenset[int]:
    """
    Elite contact + low whiff, mid-pack power/slugging.

    Contact% top N, Whiff%/K% bottom N (lowest), SLG neither elite nor punchless.
    """
    profiles = [
        row
        for row in _league_team_contact_profiles(season)
        if row.plate_appearances >= TOUGH_OUT_MIN_PA
        and row.contact_pct is not None
        and row.whiff_pct is not None
        and row.slugging is not None
    ]
    if len(profiles) < max(TOUGH_OUT_CONTACT_TOP_N, TOUGH_OUT_WHIFF_TOP_N) + 2:
        return frozenset()

    by_contact = sorted(profiles, key=lambda row: float(row.contact_pct), reverse=True)
    by_whiff = sorted(profiles, key=lambda row: float(row.whiff_pct))
    by_slg = sorted(profiles, key=lambda row: float(row.slugging), reverse=True)

    elite_contact = {row.team_id for row in by_contact[:TOUGH_OUT_CONTACT_TOP_N]}
    low_whiff = {row.team_id for row in by_whiff[:TOUGH_OUT_WHIFF_TOP_N]}
    elite_power = {row.team_id for row in by_slg[:TOUGH_OUT_SLG_EXCLUDE_TOP_N]}
    punchless = {
        row.team_id for row in by_slg[-TOUGH_OUT_SLG_EXCLUDE_BOTTOM_N:]
    }
    mid_power = {
        row.team_id
        for row in profiles
        if row.team_id not in elite_power and row.team_id not in punchless
    }
    return frozenset(elite_contact & low_whiff & mid_power)


def is_tough_out(team_id: int, *, season: int) -> bool:
    return int(team_id) in tough_out_team_ids(season)


def fetch_innings_eater_profile(
    pitcher_id: int,
    *,
    season: int,
    era: float | None = None,
) -> InningsEaterProfile:
    """Classify innings eaters and compute season FIP for luck checks."""

    def _fetch() -> InningsEaterProfile:
        try:
            payload = statsapi.get(
                "people",
                {
                    "personIds": pitcher_id,
                    "hydrate": f"stats(group=[pitching],type=[season],season={season})",
                },
            )
        except Exception as exc:
            logger.debug("Innings-eater profile fetch failed for %s: %s", pitcher_id, exc)
            return InningsEaterProfile(
                pitcher_id=pitcher_id,
                k_bb_pct=None,
                era=era,
                fip=None,
                innings_pitched=None,
            )

        people = payload.get("people") or []
        if not people:
            return InningsEaterProfile(
                pitcher_id=pitcher_id,
                k_bb_pct=None,
                era=era,
                fip=None,
                innings_pitched=None,
            )

        stats = people[0].get("stats") or []
        splits = (stats[0].get("splits") or []) if stats else []
        stat = splits[0].get("stat") or {} if splits else {}

        batters_faced = _safe_float(stat.get("battersFaced")) or 0.0
        strikeouts = _safe_float(stat.get("strikeOuts")) or 0.0
        walks = _safe_float(stat.get("baseOnBalls")) or 0.0
        hit_by_pitch = _safe_float(stat.get("hitByPitch")) or 0.0
        home_runs = _safe_float(stat.get("homeRuns")) or 0.0
        innings = _parse_innings_pitched(stat.get("inningsPitched"))
        season_era = _safe_float(stat.get("era"))
        if season_era is None:
            season_era = era

        k_bb_pct = None
        if batters_faced > 0:
            k_bb_pct = 100.0 * (strikeouts - walks) / batters_faced

        fip = None
        if innings is not None:
            fip = compute_fip(
                home_runs=home_runs,
                walks=walks,
                hit_by_pitch=hit_by_pitch,
                strikeouts=strikeouts,
                innings_pitched=innings,
            )

        is_eater = (
            k_bb_pct is not None
            and k_bb_pct < CONTACT_STARTER_K_BB_PCT
            and season_era is not None
            and INNINGS_EATER_ERA_MIN <= season_era <= INNINGS_EATER_ERA_MAX
            and innings is not None
            and innings >= INNINGS_EATER_MIN_IP
        )
        return InningsEaterProfile(
            pitcher_id=pitcher_id,
            k_bb_pct=k_bb_pct,
            era=season_era,
            fip=fip,
            innings_pitched=innings,
            is_innings_eater=is_eater,
        )

    return safe_feature_fetch(
        f"innings_eater_profile_{pitcher_id}_{season}",
        _fetch,
        fallback=InningsEaterProfile(
            pitcher_id=pitcher_id,
            k_bb_pct=None,
            era=era,
            fip=None,
            innings_pitched=None,
        ),
    )


def is_innings_eater_arm(profile: InningsEaterProfile) -> bool:
    """K-BB% < 14% gate used by the Tough Out run boost (ERA band optional)."""
    if profile.is_innings_eater:
        return True
    return (
        profile.k_bb_pct is not None
        and profile.k_bb_pct < CONTACT_STARTER_K_BB_PCT
    )


def luck_regression_scalar(profile: InningsEaterProfile) -> tuple[float, str | None]:
    """
    When an innings eater's ERA is well below FIP, boost opponent run projection.
    """
    if not profile.is_innings_eater:
        return 1.0, None
    if profile.era is None or profile.fip is None:
        return 1.0, None
    if profile.era < profile.fip - LUCK_ERA_FIP_GAP:
        return LUCK_REGRESSION_PENALTY, "luck_regression"
    return 1.0, None


@lru_cache(maxsize=8)
def _league_win_pct_table(season: int) -> tuple[tuple[int, float], ...]:
    """Return (team_id, win_pct) rows sorted best-to-worst."""

    def _fetch() -> tuple[tuple[int, float], ...]:
        try:
            payload = statsapi.standings_data(
                leagueId="103,104",
                season=str(season),
                standingsTypes="regularSeason",
            )
        except Exception as exc:
            logger.debug("Standings table fetch failed: %s", exc)
            return tuple()

        rows: list[tuple[int, float]] = []
        if not isinstance(payload, dict):
            return tuple()
        for division in payload.values():
            if not isinstance(division, dict):
                continue
            for row in division.get("teams") or []:
                team_id = int(row.get("team_id") or 0)
                wins = _safe_float(row.get("w"))
                losses = _safe_float(row.get("l"))
                if team_id <= 0 or wins is None or losses is None or (wins + losses) <= 0:
                    continue
                rows.append((team_id, wins / (wins + losses)))
        rows.sort(key=lambda item: item[1], reverse=True)
        return tuple(rows)

    return safe_feature_fetch(f"league_win_pct_table_{season}", _fetch, fallback=tuple())


def win_pct_rank_sets(
    season: int,
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """Top-N, bottom-N, and next-series Top-N win% cohorts."""
    table = _league_win_pct_table(season)
    if len(table) < LOOK_AHEAD_BOTTOM_OPP_N:
        return frozenset(), frozenset(), frozenset()
    top = frozenset(team_id for team_id, _ in table[:LOOK_AHEAD_TOP_WIN_PCT_N])
    bottom = frozenset(
        team_id for team_id, _ in table[-LOOK_AHEAD_BOTTOM_OPP_N:]
    )
    next_top = frozenset(team_id for team_id, _ in table[:LOOK_AHEAD_NEXT_TOP_N])
    return top, bottom, next_top


@lru_cache(maxsize=8)
def _team_division_map(season: int) -> dict[int, int]:
    def _fetch() -> dict[int, int]:
        try:
            payload = statsapi.get("teams", {"sportId": 1, "season": season})
        except Exception as exc:
            logger.debug("Team division map fetch failed: %s", exc)
            return {}
        mapping: dict[int, int] = {}
        for team in payload.get("teams") or []:
            team_id = team.get("id")
            division = team.get("division") or {}
            division_id = division.get("id")
            if team_id is None or division_id is None:
                continue
            mapping[int(team_id)] = int(division_id)
        return mapping

    return safe_feature_fetch(f"team_division_map_{season}", _fetch, fallback={})


def are_division_rivals(team_a: int, team_b: int, *, season: int) -> bool:
    divisions = _team_division_map(season)
    div_a = divisions.get(int(team_a))
    div_b = divisions.get(int(team_b))
    return div_a is not None and div_a == div_b


def next_series_opponent(
    team_id: int,
    *,
    current_opponent_id: int,
    game_date: date,
    season: int,
) -> int | None:
    """
    First future opponent after the current series ends.

    Walks the schedule after game_date, skipping remaining games vs current_opponent.
    """

    def _fetch() -> int | None:
        start = (game_date + timedelta(days=1)).strftime("%m/%d/%Y")
        end = (
            game_date + timedelta(days=LOOK_AHEAD_SCHEDULE_HORIZON_DAYS)
        ).strftime("%m/%d/%Y")
        try:
            games = statsapi.schedule(
                start_date=start,
                end_date=end,
                team=team_id,
                sportId=1,
            )
        except Exception as exc:
            logger.debug("Next-series schedule fetch failed for %s: %s", team_id, exc)
            return None

        for game in sorted(games, key=lambda row: str(row.get("game_date", ""))):
            if game.get("game_type") not in (None, "R"):
                continue
            home_id = game.get("home_id")
            away_id = game.get("away_id")
            if home_id is None or away_id is None:
                continue
            opponent = int(away_id) if int(home_id) == int(team_id) else int(home_id)
            if opponent == int(current_opponent_id):
                continue
            return opponent
        return None

    return safe_feature_fetch(
        f"next_series_opp_{team_id}_{current_opponent_id}_{game_date.isoformat()}_{season}",
        _fetch,
        fallback=None,
    )


def look_ahead_trap_scalar(
    *,
    team_id: int,
    opponent_id: int,
    game_date: date,
    season: int,
) -> tuple[float, str | None]:
    """
    Haircut a Top-7 club's runs when facing a Bottom-10 feeder before a tough series.

    Next series is "tough" if it is vs a Top-10 win% club or a division rival.
    """
    top, bottom, next_top = win_pct_rank_sets(season)
    if int(team_id) not in top or int(opponent_id) not in bottom:
        return 1.0, None

    nxt = next_series_opponent(
        team_id,
        current_opponent_id=opponent_id,
        game_date=game_date,
        season=season,
    )
    if nxt is None:
        return 1.0, None

    tough_next = int(nxt) in next_top or are_division_rivals(
        team_id, nxt, season=season
    )
    if not tough_next:
        return 1.0, None
    return LOOK_AHEAD_TRAP_PENALTY, "look_ahead_trap"


@lru_cache(maxsize=64)
def team_win_pct(team_id: int, season: int) -> float | None:
    table = _league_win_pct_table(season)
    for tid, pct in table:
        if tid == int(team_id):
            return pct
    return None


def vacation_mode_scalar(
    *,
    team_id: int,
    is_home: bool,
    game_date: date,
    season: int,
) -> tuple[float, str | None]:
    """Road clubs under .420 get a run-projection haircut near the break."""
    if is_home or not is_pre_all_star_window(game_date, season=season):
        return 1.0, None
    win_pct = team_win_pct(team_id, season)
    if win_pct is None or win_pct >= VACATION_MODE_WIN_PCT_MAX:
        return 1.0, None
    return VACATION_MODE_PENALTY, "vacation_mode"


def break_push_bonus(
    prob: float,
    *,
    team_id: int,
    game_date: date,
    season: int,
) -> float:
    """Winning Tough Out clubs get a win-prob bump in the getaway window."""
    if prob <= 0 or not is_pre_all_star_window(game_date, season=season):
        return prob
    if not is_tough_out(team_id, season=season):
        return prob
    win_pct = team_win_pct(team_id, season)
    if win_pct is None or win_pct <= BREAK_PUSH_WIN_PCT_MIN:
        return prob
    return min(prob * BREAK_PUSH_BONUS, 0.99)


def apply_tough_out_run_scalars(
    offense_runs: float,
    *,
    offense_team_id: int,
    pitcher_id: int | None,
    pitcher_era: float | None,
    is_home_offense: bool,
    game_date: date,
    season: int,
    label: str,
    opponent_team_id: int | None = None,
) -> tuple[float, list[str]]:
    """Stack Tough Out, luck-regression, vacation-mode, and look-ahead run multipliers."""
    tags: list[str] = []
    runs = offense_runs

    eater: InningsEaterProfile | None = None
    if pitcher_id is not None:
        eater = fetch_innings_eater_profile(
            pitcher_id,
            season=season,
            era=pitcher_era,
        )

    if eater is not None and is_innings_eater_arm(eater) and is_tough_out(
        offense_team_id, season=season
    ):
        runs *= GRITTY_OFFENSE_SCALAR
        tags.append(f"{label}:gritty_offense:{GRITTY_OFFENSE_SCALAR:.2f}")

    if eater is not None:
        luck_scalar, luck_tag = luck_regression_scalar(eater)
        if luck_tag:
            runs *= luck_scalar
            tags.append(f"{label}:{luck_tag}:{luck_scalar:.2f}")

    vac_scalar, vac_tag = vacation_mode_scalar(
        team_id=offense_team_id,
        is_home=is_home_offense,
        game_date=game_date,
        season=season,
    )
    if vac_tag:
        runs *= vac_scalar
        tags.append(f"{label}:{vac_tag}:{vac_scalar:.2f}")

    if opponent_team_id is not None:
        trap_scalar, trap_tag = look_ahead_trap_scalar(
            team_id=offense_team_id,
            opponent_id=opponent_team_id,
            game_date=game_date,
            season=season,
        )
        if trap_tag:
            runs *= trap_scalar
            tags.append(f"{label}:{trap_tag}:{trap_scalar:.2f}")

    return runs, tags
