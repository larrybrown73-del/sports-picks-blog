"""Full-season pitcher-vs-hitter matchup adjustments for slate run projections."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import Any

import requests
import statsapi

from baseball_data import get_starting_pitcher_info
from config import (
    BABIP_LUCK_BONUS,
    BABIP_LUCK_ERA_CEILING,
    BABIP_LUCK_THRESHOLD,
    GROUND_BALL_PITCHER_GB_PCT,
    PATIENT_LINEUP_ADVANTAGE,
    PATIENT_LINEUP_BB_PCT,
    PATIENT_LINEUP_PITCHES_PER_PA,
    POWER_PITCHER_VELO_FLOOR,
    REGRESSION_PENALTY,
    SMOKE_MIRRORS_BABIP_CEILING,
    SMOKE_MIRRORS_ERA_CEILING,
    SMOKE_MIRRORS_WHIP_FLOOR,
    VELO_DOMINANCE_SCALAR,
    VELO_MATCHUP_FASTBALL_MPH,
    VELO_STRUGGLE_BOTTOM_N,
)
from data_health import safe_feature_fetch
from hitter_discipline import apply_lineup_discipline_to_runs, fetch_game_lineup
from starter_rest_hierarchy import apply_starter_context_to_runs
from tough_out import apply_tough_out_run_scalars

logger = logging.getLogger(__name__)

_FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "FC", "FS"}


@dataclass(frozen=True)
class PitcherSeasonProfile:
    pitcher_id: int
    pitcher_name: str
    season_era: float | None
    season_whip: float | None
    season_babip: float | None
    avg_fastball_velo: float | None
    ground_ball_pct: float | None
    games_started: float = 0.0


@dataclass(frozen=True)
class TeamOffenseProfile:
    team_id: int
    walk_pct: float | None
    pitches_per_pa: float | None


@dataclass
class PitcherMatchupResult:
    home_runs: float
    away_runs: float
    tags: list[str] = field(default_factory=list)


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


def _compute_babip(stat: dict[str, Any]) -> float | None:
    hits = _safe_float(stat.get("hits"))
    home_runs = _safe_float(stat.get("homeRuns")) or 0.0
    at_bats = _safe_float(stat.get("atBats"))
    strike_outs = _safe_float(stat.get("strikeOuts")) or 0.0
    sac_flies = _safe_float(stat.get("sacFlies")) or 0.0
    if hits is None or at_bats is None:
        direct = _safe_float(stat.get("babip"))
        return direct
    denominator = at_bats - strike_outs - home_runs + sac_flies
    if denominator <= 0:
        return None
    return (hits - home_runs) / denominator


def _compute_ground_ball_pct(stat: dict[str, Any]) -> float | None:
    direct = _safe_float(stat.get("groundBallPercentage"))
    if direct is not None:
        return direct
    ground_outs = _safe_float(stat.get("groundOuts"))
    air_outs = _safe_float(stat.get("airOuts"))
    if ground_outs is None or air_outs is None:
        return None
    total = ground_outs + air_outs
    if total <= 0:
        return None
    return 100.0 * ground_outs / total


def _extract_pitching_season_stat(payload: dict[str, Any]) -> dict[str, Any] | None:
    for block in payload.get("stats", []):
        group = block.get("group")
        group_name = ""
        if isinstance(group, dict):
            group_name = str(group.get("displayName", "")).lower()
        else:
            group_name = str(group or "").lower()
        if group_name != "pitching":
            continue
        type_info = block.get("type") or {}
        type_name = str(
            type_info.get("displayName") if isinstance(type_info, dict) else type_info
        ).lower()
        if type_name and "statcast" in type_name:
            continue
        splits = block.get("splits") or []
        if splits:
            return splits[0].get("stat") or {}
    return None


def _extract_statcast_pitching_splits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in payload.get("stats", []):
        type_name = (block.get("type") or {}).get("displayName", "")
        if "statcast" not in str(type_name).lower():
            continue
        rows.extend(block.get("splits") or [])
    return rows


def _fetch_avg_fastball_velo(pitcher_id: int, season: int) -> float | None:
    """Average fastball velocity from Statcast pitch logs."""
    try:
        from datetime import date as dt_date

        from pybaseball import statcast_pitcher

        start = dt_date(season, 3, 1).isoformat()
        end = min(dt_date.today(), dt_date(season, 11, 15)).isoformat()
        frame = statcast_pitcher(start, end, pitcher_id)
        if frame is None or frame.empty or "release_speed" not in frame.columns:
            return None
        pitch_col = "pitch_type" if "pitch_type" in frame.columns else None
        if pitch_col:
            fastballs = frame[
                frame[pitch_col].astype(str).str.upper().isin(_FASTBALL_TYPES)
            ]
        else:
            fastballs = frame.iloc[0:0]
        speeds = fastballs["release_speed"].dropna().astype(float)
        if speeds.empty:
            speeds = frame["release_speed"].dropna().astype(float)
        if speeds.empty:
            return None
        return float(speeds.mean())
    except Exception as exc:
        logger.debug("Fastball velo fetch failed for %s: %s", pitcher_id, exc)
        return None


def fetch_pitcher_season_profile(
    pitcher_id: int,
    *,
    pitcher_name: str = "",
    season: int | None = None,
) -> PitcherSeasonProfile | None:
    """Pull full-season ERA/WHIP/BABIP plus velocity and GB% for a starter."""
    season = season or date.today().year

    def _fetch() -> PitcherSeasonProfile | None:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats",
            params={
                "stats": "season",
                "group": "pitching",
                "season": season,
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        stat = _extract_pitching_season_stat({"stats": payload.get("stats", [])}) or {}

        person_name = pitcher_name
        if not person_name:
            try:
                person_payload = statsapi.get("people", {"personIds": pitcher_id})
                people = person_payload.get("people", [])
                if people:
                    person_name = str(people[0].get("fullName") or "")
            except Exception:
                person_name = ""

        era = _safe_float(stat.get("era"))
        whip = _safe_float(stat.get("whip"))
        if whip is None:
            walks = _safe_float(stat.get("baseOnBalls")) or 0.0
            hits = _safe_float(stat.get("hits")) or 0.0
            innings = _parse_innings_pitched(stat.get("inningsPitched"))
            if innings and innings > 0:
                whip = (walks + hits) / innings

        babip = _compute_babip(stat)
        gb_pct = _compute_ground_ball_pct(stat)
        games_started = _safe_float(stat.get("gamesStarted")) or 0.0

        avg_velo = _fetch_avg_fastball_velo(pitcher_id, season)

        if era is None and babip is None and avg_velo is None and gb_pct is None:
            return None

        resolved_name = person_name or f"SP {pitcher_id}"
        return PitcherSeasonProfile(
            pitcher_id=pitcher_id,
            pitcher_name=resolved_name,
            season_era=era,
            season_whip=whip,
            season_babip=babip,
            avg_fastball_velo=avg_velo,
            ground_ball_pct=gb_pct,
            games_started=games_started,
        )

    return safe_feature_fetch(
        f"pitcher_season_profile_{pitcher_id}",
        _fetch,
        fallback=None,
    )


def fetch_team_offense_profile(team_id: int, season: int | None = None) -> TeamOffenseProfile:
    """Season team walk rate and pitches per plate appearance."""
    season = season or date.today().year

    def _fetch() -> TeamOffenseProfile:
        stats = statsapi.get(
            "team_stats",
            {
                "teamId": team_id,
                "season": season,
                "group": "hitting",
                "stats": "season",
                "sportIds": 1,
            },
        )
        splits = (stats.get("stats") or [{}])[0].get("splits") or []
        stat = splits[0].get("stat", {}) if splits else {}
        plate_appearances = _safe_float(stat.get("plateAppearances"))
        walks = _safe_float(stat.get("baseOnBalls"))
        pitches = _safe_float(stat.get("numberOfPitches"))
        walk_pct = None
        if plate_appearances and plate_appearances > 0 and walks is not None:
            walk_pct = 100.0 * walks / plate_appearances
        pitches_per_pa = None
        if plate_appearances and plate_appearances > 0 and pitches is not None:
            pitches_per_pa = pitches / plate_appearances
        return TeamOffenseProfile(
            team_id=team_id,
            walk_pct=walk_pct,
            pitches_per_pa=pitches_per_pa,
        )

    return safe_feature_fetch(
        f"team_offense_profile_{team_id}",
        _fetch,
        fallback=TeamOffenseProfile(team_id=team_id, walk_pct=None, pitches_per_pa=None),
    )


@lru_cache(maxsize=4)
def _team_velo_struggle_ranks(season: int) -> dict[int, int]:
    """
    Rank MLB teams by wOBA vs 95+ MPH fastballs (1 = worst).

    Uses pybaseball Statcast when available; returns empty dict on failure.
    """
    try:
        from datetime import date as dt_date

        import pandas as pd
        from pybaseball import statcast

        start = dt_date(season, 3, 1).isoformat()
        end = min(dt_date.today(), dt_date(season, 11, 15)).isoformat()
        frame = statcast(start, end)
        if frame is None or frame.empty:
            return {}

        required = {"release_speed", "pitch_type", "events", "stand", "player_name"}
        if not required.issubset(frame.columns):
            return {}

        fastballs = frame[
            frame["release_speed"].astype(float) >= VELO_MATCHUP_FASTBALL_MPH
        ].copy()
        fastballs = fastballs[
            fastballs["pitch_type"].astype(str).str.upper().isin(_FASTBALL_TYPES)
        ]
        if fastballs.empty or "home_team" not in fastballs.columns:
            return {}

        def _batting_team(row: pd.Series) -> str | None:
            topbot = str(row.get("inning_topbot", "")).strip().lower()
            if topbot == "top":
                return str(row.get("away_team") or "")
            if topbot == "bot":
                return str(row.get("home_team") or "")
            return None

        fastballs["batting_team"] = fastballs.apply(_batting_team, axis=1)
        fastballs = fastballs[fastballs["batting_team"].notna()]
        if fastballs.empty:
            return {}

        if "estimated_woba_using_speedangle" in fastballs.columns:
            fastballs["quality"] = pd.to_numeric(
                fastballs["estimated_woba_using_speedangle"], errors="coerce"
            )
        else:
            fastballs["quality"] = fastballs["events"].map(
                {
                    "single": 0.88,
                    "double": 1.25,
                    "triple": 1.58,
                    "home_run": 2.00,
                    "walk": 0.69,
                    "hit_by_pitch": 0.72,
                }
            )

        team_rows: list[dict[str, float | str]] = []
        for team_name, group in fastballs.groupby("batting_team"):
            values = group["quality"].dropna()
            if len(values) < 40:
                continue
            team_rows.append(
                {"team_name": str(team_name), "score": float(values.mean())}
            )

        if len(team_rows) < VELO_STRUGGLE_BOTTOM_N:
            return {}

        teams = statsapi.get("teams", {"sportId": 1, "season": season}).get("teams", [])
        name_to_id = {team["name"]: int(team["id"]) for team in teams}
        ranked = sorted(team_rows, key=lambda row: float(row["score"]))
        ranks: dict[int, int] = {}
        for index, row in enumerate(ranked, start=1):
            team_id = name_to_id.get(str(row["team_name"]))
            if team_id is not None:
                ranks[team_id] = index
        return ranks
    except Exception as exc:
        logger.warning("Velo-struggle ranking unavailable: %s", exc)
        return {}


def pitcher_runs_allowed_scalar(profile: PitcherSeasonProfile) -> tuple[float, str | None]:
    """Return multiplier applied to runs allowed by this starter."""
    era = profile.season_era
    whip = profile.season_whip
    babip = profile.season_babip

    if (
        babip is not None
        and era is not None
        and babip > BABIP_LUCK_THRESHOLD
        and era < BABIP_LUCK_ERA_CEILING
    ):
        return BABIP_LUCK_BONUS, "babip_luck_bonus"

    if (
        era is not None
        and whip is not None
        and babip is not None
        and era < SMOKE_MIRRORS_ERA_CEILING
        and whip > SMOKE_MIRRORS_WHIP_FLOOR
        and babip < SMOKE_MIRRORS_BABIP_CEILING
    ):
        return REGRESSION_PENALTY, "regression_penalty"

    return 1.0, None


def is_power_pitcher(profile: PitcherSeasonProfile) -> bool:
    return (
        profile.avg_fastball_velo is not None
        and profile.avg_fastball_velo > POWER_PITCHER_VELO_FLOOR
    )


def is_ground_ball_pitcher(profile: PitcherSeasonProfile) -> bool:
    return (
        profile.ground_ball_pct is not None
        and profile.ground_ball_pct > GROUND_BALL_PITCHER_GB_PCT
    )


def is_patient_lineup(profile: TeamOffenseProfile) -> bool:
    walk_ok = profile.walk_pct is not None and profile.walk_pct > PATIENT_LINEUP_BB_PCT
    pitch_ok = (
        profile.pitches_per_pa is not None
        and profile.pitches_per_pa > PATIENT_LINEUP_PITCHES_PER_PA
    )
    return walk_ok or pitch_ok


def is_velo_struggler(team_id: int, season: int) -> bool:
    ranks = _team_velo_struggle_ranks(season)
    rank = ranks.get(team_id)
    return rank is not None and rank <= VELO_STRUGGLE_BOTTOM_N


def _apply_offense_adjustments(
    offense_runs: float,
    *,
    pitcher: PitcherSeasonProfile | None,
    offense: TeamOffenseProfile,
    season: int,
    label: str,
    defending_team_id: int | None = None,
    game_date: date | None = None,
    is_home_offense: bool = False,
    lineup: list | None = None,
) -> tuple[float, list[str]]:
    tags: list[str] = []
    runs = offense_runs

    if pitcher is None:
        # Vacation / look-ahead / missing-star can still apply without a known SP.
        if game_date is not None:
            runs, tough_tags = apply_tough_out_run_scalars(
                runs,
                offense_team_id=offense.team_id,
                pitcher_id=None,
                pitcher_era=None,
                is_home_offense=is_home_offense,
                game_date=game_date,
                season=season,
                label=label,
                opponent_team_id=defending_team_id,
                lineup=lineup,
            )
            tags.extend(tough_tags)
        return runs, tags

    if defending_team_id is not None and game_date is not None:
        runs, starter_tags = apply_starter_context_to_runs(
            runs,
            pitcher_id=pitcher.pitcher_id,
            defending_team_id=defending_team_id,
            game_date=game_date,
            season=season,
            label=label,
        )
        tags.extend(starter_tags)

    stability_scalar, stability_tag = pitcher_runs_allowed_scalar(pitcher)
    if stability_tag:
        runs *= stability_scalar
        tags.append(f"{label}:{stability_tag}:{stability_scalar:.2f}")

    if is_power_pitcher(pitcher) and is_velo_struggler(offense.team_id, season):
        runs *= VELO_DOMINANCE_SCALAR
        tags.append(f"{label}:velo_dominance:{VELO_DOMINANCE_SCALAR:.2f}")

    if is_ground_ball_pitcher(pitcher) and is_patient_lineup(offense):
        runs *= PATIENT_LINEUP_ADVANTAGE
        tags.append(f"{label}:patient_lineup:{PATIENT_LINEUP_ADVANTAGE:.2f}")

    if game_date is not None:
        runs, tough_tags = apply_tough_out_run_scalars(
            runs,
            offense_team_id=offense.team_id,
            pitcher_id=pitcher.pitcher_id,
            pitcher_era=pitcher.season_era,
            is_home_offense=is_home_offense,
            game_date=game_date,
            season=season,
            label=label,
            opponent_team_id=defending_team_id,
            lineup=lineup,
        )
        tags.extend(tough_tags)

    return runs, tags


def apply_pitcher_matchup_adjustments(
    home_runs: float,
    away_runs: float,
    *,
    game_id: int,
    home_id: int,
    away_id: int,
    season: int | None = None,
    game_date: date | None = None,
) -> PitcherMatchupResult:
    """
    Apply full-season pitcher guardrails and style matchups to projected runs.

    home_runs = runs scored by the home offense (vs away starter)
    away_runs = runs scored by the away offense (vs home starter)
    """
    season = season or date.today().year
    game_date = game_date or date.today()
    tags: list[str] = []

    pitcher_info = get_starting_pitcher_info(game_id)
    home_pitcher_id = pitcher_info.get("home_pitcher_id")
    away_pitcher_id = pitcher_info.get("away_pitcher_id")

    home_pitcher = (
        fetch_pitcher_season_profile(
            int(home_pitcher_id),
            pitcher_name=str(pitcher_info.get("home_pitcher_name") or ""),
            season=season,
        )
        if home_pitcher_id is not None
        else None
    )
    away_pitcher = (
        fetch_pitcher_season_profile(
            int(away_pitcher_id),
            pitcher_name=str(pitcher_info.get("away_pitcher_name") or ""),
            season=season,
        )
        if away_pitcher_id is not None
        else None
    )

    away_offense = fetch_team_offense_profile(away_id, season=season)
    home_offense = fetch_team_offense_profile(home_id, season=season)
    away_lineup, home_lineup = fetch_game_lineup(game_id)

    away_runs, away_tags = _apply_offense_adjustments(
        away_runs,
        pitcher=home_pitcher,
        offense=away_offense,
        season=season,
        label=home_pitcher.pitcher_name if home_pitcher else "home_sp",
        defending_team_id=home_id,
        game_date=game_date,
        is_home_offense=False,
        lineup=away_lineup,
    )
    home_runs, home_tags = _apply_offense_adjustments(
        home_runs,
        pitcher=away_pitcher,
        offense=home_offense,
        season=season,
        label=away_pitcher.pitcher_name if away_pitcher else "away_sp",
        defending_team_id=away_id,
        game_date=game_date,
        is_home_offense=True,
        lineup=home_lineup,
    )
    tags.extend(away_tags)
    tags.extend(home_tags)

    away_runs, away_hitter_tags = apply_lineup_discipline_to_runs(
        away_runs,
        away_lineup,
        season=season,
        label="away_offense",
    )
    home_runs, home_hitter_tags = apply_lineup_discipline_to_runs(
        home_runs,
        home_lineup,
        season=season,
        label="home_offense",
    )
    tags.extend(away_hitter_tags)
    tags.extend(home_hitter_tags)

    return PitcherMatchupResult(home_runs=home_runs, away_runs=away_runs, tags=tags)
