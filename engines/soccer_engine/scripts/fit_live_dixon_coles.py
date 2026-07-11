#!/usr/bin/env python3
"""
Live smoke test for the Dixon-Coles fit: hits TheStatsAPI for real, fits
`fit_dixon_coles(fetch_historical_matches(THESTATSAPI_KEY, competition_id))`
for one competition, and logs the resulting team ratings.

This is a verification/debugging tool, not part of the production pipeline
(daily_model.py already does this fit as part of the scheduler's morning
pull) -- it exists to let a human confirm the live data fetch + fit actually
works end-to-end against the real API, with real numbers to eyeball.

Usage (from anywhere; requires network access and a valid THESTATSAPI_KEY in
sports-picks-blog/.env.local or the environment):

    python fit_live_dixon_coles.py                      # defaults to Premier League
    python fit_live_dixon_coles.py --competition "La Liga"
    python fit_live_dixon_coles.py --competition-id comp_3039 --seasons-back 2
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
REPO_ROOT = ENGINE_ROOT.parent.parent

from ev_engine_core import get_thestatsapi_key, thestatsapi_get  # noqa: E402
from historical_data import MatchResult, fetch_historical_matches  # noqa: E402
from team_strength import SoccerModelDataError, TeamRatings, fit_dixon_coles  # noqa: E402

DEFAULT_COMPETITION_NAME = "Premier League"
DEFAULT_SEASONS_BACK_DAYS = 365 * 2  # ~2 seasons of history for a stable fit


def find_competition_id(api_key: str, name: str) -> tuple[str, str]:
    """
    GET /football/competitions?search=<name>, returning (id, canonical_name)
    for the best match. Never hardcodes a competition id -- TheStatsAPI is
    the only source of truth for what a competition's real id is.
    """

    payload = thestatsapi_get("/football/competitions", api_key, params={"search": name, "per_page": 50})
    rows = payload.get("data", []) or []
    if not rows:
        raise SoccerModelDataError(f"No competitions found on TheStatsAPI matching search={name!r}")

    exact = [row for row in rows if str(row.get("name", "")).strip().lower() == name.strip().lower()]
    match = exact[0] if exact else rows[0]
    competition_id = str(match.get("id") or "")
    canonical_name = str(match.get("name") or name)
    if not competition_id:
        raise SoccerModelDataError(f"Competition search result for {name!r} is missing an id: {match!r}")
    return competition_id, canonical_name


def team_name_map(matches: list[MatchResult]) -> dict[str, str]:
    """Build team_id -> team_name straight from the fetched match rows -- no extra API calls needed."""

    names: dict[str, str] = {}
    for match in matches:
        names.setdefault(match.home_team_id, match.home_team_name)
        names.setdefault(match.away_team_id, match.away_team_name)
    return names


def log_team_ratings(ratings: TeamRatings, names: dict[str, str], *, top_n: int = 10) -> None:
    print(f"\nFitted at (UTC): {ratings.fitted_at}")
    print(f"Home advantage (log scale): {ratings.home_advantage:+.4f}")
    print(f"Rho (low-score correlation): {ratings.rho:+.4f}")
    print(f"League avg home goals: {ratings.league_avg_home_goals:.3f}")
    print(f"League avg away goals: {ratings.league_avg_away_goals:.3f}")
    print(f"Teams fitted: {len(ratings.attack)}")

    ranked = sorted(ratings.attack.items(), key=lambda item: item[1], reverse=True)
    print(f"\nTop {top_n} teams by fitted attack strength (log scale):")
    header = f"  {'Team':<28}{'Attack':>10}{'Defense':>10}{'Matches':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for team_id, attack in ranked[:top_n]:
        defense = ratings.defense.get(team_id, 0.0)
        played = ratings.matches_played.get(team_id, 0)
        team_name = names.get(team_id, team_id)
        print(f"  {team_name:<28}{attack:>10.4f}{defense:>10.4f}{played:>10d}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default=DEFAULT_COMPETITION_NAME, help="Competition name to search for.")
    parser.add_argument("--competition-id", default=None, help="Skip the name search and use this id directly.")
    parser.add_argument(
        "--days-back", type=int, default=DEFAULT_SEASONS_BACK_DAYS, help="How many days of history to fetch."
    )
    args = parser.parse_args()

    api_key = get_thestatsapi_key(env_file=str(REPO_ROOT / ".env.local"))
    if not api_key:
        print("THESTATSAPI_KEY is not set (checked environment and .env.local). Aborting.", file=sys.stderr)
        return 1

    if args.competition_id:
        competition_id, competition_name = args.competition_id, args.competition_id
    else:
        print(f"Searching TheStatsAPI for competition matching {args.competition!r}...")
        competition_id, competition_name = find_competition_id(api_key, args.competition)
    print(f"Using competition: {competition_name!r} ({competition_id})")

    date_to = date.today()
    date_from = date_to - timedelta(days=args.days_back)
    print(f"Fetching finished matches from {date_from.isoformat()} to {date_to.isoformat()}...")

    matches = fetch_historical_matches(
        api_key,
        competition_id,
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
    )
    print(f"Fetched {len(matches)} finished matches with usable scorelines.")
    if not matches:
        print("No usable historical matches returned -- cannot fit. Aborting.", file=sys.stderr)
        return 1

    ratings = fit_dixon_coles(matches)
    names = team_name_map(matches)
    log_team_ratings(ratings, names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
