"""
Micro-Market EV Arbitrage Engine (soccer).

ARCHITECTURE CONTRACT (do not violate when extending this file):
  1. "True Probability" (the projection model's output) and "Sportsbook Odds"
     (what a book is offering) are represented by two separate dataclasses --
     MarketLeg (odds side) and ProbabilityEstimate (model side) -- and are
     only ever combined inside `evaluate_leg` / `evaluate_leg`'s callers.
     Nothing upstream of that boundary should compute anything that mixes
     the two, so a bug in the odds ingestion can never silently corrupt the
     model's probability output (and vice versa).
  2. No probability, edge, or EV value is rounded anywhere in this module.
     Floats are carried at full precision from the moment they're computed
     until a caller explicitly formats them for display (see
     `EVResult.to_record`, which is the one sanctioned place to round, and
     only because it is producing UI/report strings). `confidence_score`
     is the one intentional exception: it is a 1-100 *score*, not a
     probability, and rounding it to an int is part of its definition.
  3. Betting lines (point spreads, totals, player-prop lines like
     "2.5 shots") are always floats, never ints. A line of 2.5 is a
     half-integer specifically so it can never push, and truncating it to
     `int(2.5) == 2` would silently change what the bet even means.
  4. Missing odds/stats are dropped, never guessed. If a sportsbook payload
     has no price for an outcome, that outcome is skipped outright (see the
     `if price is None: continue` / `if odd is None: continue` guards in the
     `flatten_thestatsapi_*` functions below) -- we never substitute a
     league-average price, a prior snapshot, or an interpolated value.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

logger = logging.getLogger("soccer_engine.ev_engine_core")

OddsFormat = Literal["american", "decimal"]

DEFAULT_STAKE = 1.0
MIN_POSITIVE_EV = 0.0
DEFAULT_MIN_EDGE = 0.0

THESTATSAPI_BASE_URL = "https://api.thestatsapi.com/api"
THESTATSAPI_ENV_VAR = "THESTATSAPI_KEY"

# Repo root (sports-picks-blog/), computed from this file's own location
# rather than assumed from the process's current working directory.
# `.env.local` lives here. Anything that resolves cwd-relative (a bare
# ".env.local" string) works fine for interactive use from the repo root,
# but silently fails the moment this code runs from anywhere else -- e.g. a
# Windows Task Scheduler entry, a cron job, or scheduler.py launched from
# engines/soccer_engine/ instead of the repo root. Defaulting to this
# absolute path removes that whole failure class.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"

MARKET_VOLATILITY: dict[str, float] = {
    "moneyline": 0.18,
    "spread": 0.25,
    "total": 0.28,
    "btts": 0.26,
    "team_total": 0.32,
    "goalscorer": 0.45,
    "assists": 0.38,
    "passes": 0.30,
    "shots": 0.42,
    "tackles": 0.46,
    "cards": 0.52,
    "corners": 0.40,
}


@dataclass(frozen=True)
class MarketLeg:
    """A normalized sportsbook bet leg from a team or player micro-market."""

    game_id: str
    market_type: str
    selection: str
    odds: float
    sportsbook: str
    odds_format: OddsFormat = "american"
    # Always a float, never an int: prop/total/spread lines are frequently
    # half-integers (2.5 shots, 0.5 SOT, 3.5 corners) specifically so the
    # bet can never push. int(2.5) == 2 would silently change the wager.
    line: float | None = None
    side: str | None = None
    entity_name: str | None = None
    entity_id: str | None = None
    team: str | None = None
    event_id: str | None = None
    market_key: str | None = None
    outcome_name: str | None = None
    last_update: str | None = None
    sample_size: int | None = None
    volatility: float | None = None
    correlation_group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def participant_key(self) -> str:
        participant = self.entity_id or self.entity_name or self.team or self.selection
        return _slug(participant)

    @property
    def market_family(self) -> str:
        return market_family(self.market_type)


@dataclass(frozen=True)
class ProbabilityEstimate:
    """True-probability output from the soccer projection model."""

    true_probability: float
    # None (the common case) means "this is a strict win/lose market", so
    # `expected_value` derives loss probability as `1 - true_probability`.
    # Set this explicitly for markets where a third, non-win/non-loss
    # outcome is possible -- e.g. an Asian Handicap on a whole-number line
    # can "push" (stake refunded), so true_probability + loss_probability
    # can legitimately sum to less than 1, with the remainder being push
    # probability that correctly contributes nothing to EV.
    loss_probability: float | None = None
    sample_size: int | None = None
    volatility: float | None = None
    model_version: str | None = None
    warnings: list[str] = field(default_factory=list)


class TrueProbabilityProvider(Protocol):
    """Adapter protocol for plugging in an existing soccer projection model."""

    def __call__(self, leg: MarketLeg) -> ProbabilityEstimate | float:
        ...


@dataclass(frozen=True)
class EVResult:
    """Full EV grading record for one normalized sportsbook leg."""

    leg: MarketLeg
    implied_probability: float
    true_probability: float
    edge: float
    edge_pct: float
    decimal_odds: float
    potential_profit: float
    ev: float
    ev_per_unit: float
    positive_ev: bool
    confidence_score: int
    loss_probability: float = 0.0
    sample_size: int | None = None
    volatility: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        return {
            "game_id": self.leg.game_id,
            "sportsbook": self.leg.sportsbook,
            "market_type": self.leg.market_type,
            "market_family": self.leg.market_family,
            "selection": self.leg.selection,
            "entity_name": self.leg.entity_name,
            "team": self.leg.team,
            "line": self.leg.line,
            "side": self.leg.side,
            "odds": self.leg.odds,
            "odds_format": self.leg.odds_format,
            "decimal_odds": round(self.decimal_odds, 4),
            "implied_probability": round(self.implied_probability, 4),
            "true_probability": round(self.true_probability, 4),
            "loss_probability": round(self.loss_probability, 4),
            "edge_pct": round(self.edge_pct, 2),
            "ev_per_unit": round(self.ev_per_unit, 4),
            "positive_ev": self.positive_ev,
            "confidence_score": self.confidence_score,
            "sample_size": self.sample_size,
            "volatility": self.volatility,
            "warnings": "; ".join(self.warnings),
        }


def load_env_file(env_file: str | os.PathLike[str] = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=value pairs without requiring python-dotenv."""

    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_thestatsapi_key(
    env_var: str = THESTATSAPI_ENV_VAR,
    *,
    env_file: str | os.PathLike[str] | None = DEFAULT_ENV_FILE,
) -> str | None:
    """
    Read the TheStatsAPI bearer token from the environment.

    The key must stay out of source control. Set THESTATSAPI_KEY in a local,
    gitignored `.env.local` (or the real environment) rather than hardcoding it.
    """

    if env_file is not None and not os.getenv(env_var):
        load_env_file(env_file)
    return os.getenv(env_var)


# ---------------------------------------------------------------------------
# Odds math (format-agnostic; works for both American-quoted and decimal books)
#
# IMPORTANT ON VIG/JUICE: every implied-probability function below converts a
# SINGLE outcome's price. That number always has the sportsbook's margin
# baked into it -- e.g. a "true" 50/50 coin-flip market priced at -110/-110
# implies 52.38% + 52.38% = 104.76%, not 100%. That extra ~4.76% is the vig.
# `implied_probability` is intentionally vig-inclusive because it answers
# "what does this specific price imply", which is exactly what the EV
# formula needs (you pay the vig when you place the bet, so the EV
# calculation must be measured against the real, vig-included price). If you
# instead need a *fair*, no-vig probability estimate (e.g. to sanity-check
# the model against consensus market pricing), use `devig_probabilities` /
# `market_overround` further down, which operate on a full market's outcomes
# at once, not a single price.
# ---------------------------------------------------------------------------


def american_to_decimal(american_odds: float | int) -> float:
    """
    Convert American odds to decimal odds.

    Rounded to the nearest whole number first because American odds are only
    ever quoted in whole numbers (e.g. -110, +150) -- a fractional American
    price would be a transcription bug upstream, not a real quote, so we
    normalize defensively rather than silently propagate a malformed value.
    """

    odds = int(round(float(american_odds)))
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        # A +150 favorite payout means "bet 100 to win 150", i.e. 2.5x return.
        return 1.0 + odds / 100.0
    # A -110 underdog price means "bet 110 to win 100": the magnitude of the
    # negative number is the stake required for a flat 100-unit profit.
    return 1.0 + 100.0 / abs(odds)


def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds (the inverse of
    `american_to_decimal`). Always returns a whole number -- like American
    odds themselves, there's no such thing as a fractional "+150.5".

    Decimal 2.0 (even money) is the boundary: >= 2.0 is priced as a "bet 100
    to win N" favorite-or-plus-money price (positive), below 2.0 is priced
    as a "bet N to win 100" odds-on price (negative). This is the standard
    sportsbook convention, not an arbitrary choice.
    """

    decimal = float(decimal_odds)
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def decimal_to_implied(decimal_odds: float) -> float:
    """
    Implied probability = 1 / decimal odds. This is the market's breakeven
    win rate for the bet as priced -- it is NOT a fair/no-vig probability
    (see the module-level note above `american_to_decimal`).
    """

    decimal = float(decimal_odds)
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal


def american_to_implied(american_odds: float | int) -> float:
    return decimal_to_implied(american_to_decimal(american_odds))


def odds_to_decimal(odds: float | int, odds_format: OddsFormat = "american") -> float:
    if odds_format == "decimal":
        decimal = float(odds)
        if decimal <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.0")
        return decimal
    return american_to_decimal(odds)


def implied_probability(odds: float | int, odds_format: OddsFormat = "american") -> float:
    """Single-outcome, vig-inclusive implied probability. See module note above."""

    return decimal_to_implied(odds_to_decimal(odds, odds_format))


def market_overround(implied_probabilities: Iterable[float]) -> float:
    """
    Sum of implied probabilities across every outcome in one market (e.g. all
    of home/draw/away, or both sides of a total-goals line, from the SAME
    sportsbook at the SAME snapshot in time).

    A perfectly fair, zero-vig market sums to exactly 1.0. Real sportsbook
    markets sum to something greater than 1.0 -- that excess IS the vig/juice,
    expressed as a probability. This function only measures the vig; it does
    not remove it (see `devig_probabilities` for that).
    """

    return sum(implied_probabilities)


def devig_probabilities(implied_probabilities: Sequence[float]) -> list[float]:
    """
    Remove the vig from a full market's implied probabilities using the
    standard proportional ("multiplicative") method: scale every outcome's
    implied probability down by the market's total overround so they sum to
    exactly 1.0.

    Why this exists: `implied_probability()` on a single price cannot, by
    itself, "account for the vig" -- the vig only becomes visible once you
    look at every outcome in the market together. This function is that
    market-level step. It answers "if this book had zero margin, what would
    it actually think each outcome's probability is", which is useful for
    calibrating/sanity-checking a model's True Probability output against
    market consensus.

    This is explicitly NOT used inside `expected_value()` / `evaluate_leg()`.
    EV must be computed against the real, vig-included price you are actually
    betting into -- de-vigged probabilities are for analysis and model
    calibration only, never for grading a specific bet's profitability.
    """

    total = market_overround(implied_probabilities)
    if total <= 0:
        raise ValueError("Cannot de-vig a market with non-positive total implied probability")
    return [p / total for p in implied_probabilities]


def expected_value(
    true_probability: float,
    decimal_odds: float,
    *,
    stake: float = DEFAULT_STAKE,
    loss_probability: float | None = None,
) -> float:
    """
    EV = (True_Win_Probability * Potential_Profit)
       - (True_Loss_Probability * Wager_Amount)

    `true_probability` must be the MODEL's estimate, never the market's
    implied probability -- if you pass implied probability in here you get a
    tautological EV of ~0 (minus the vig), which defeats the entire point of
    having an independent projection model. `clamp_probability` guards only
    against a malformed/out-of-range model output (e.g. a bug producing
    1.4); it is not a rounding step and does not otherwise alter the float.

    `loss_probability` defaults to `1 - true_probability`, which is correct
    for any strict win/lose market (moneyline, BTTS, totals, most props).
    Pass it explicitly for markets with a possible push (e.g. a
    whole-number Asian Handicap line) so the omitted push probability
    contributes exactly zero to EV instead of being forced into the loss
    term.
    """

    prob = clamp_probability(true_probability)
    if stake <= 0:
        raise ValueError("Stake must be positive")
    potential_profit = stake * (decimal_odds - 1.0)
    true_loss_probability = clamp_probability(loss_probability) if loss_probability is not None else (1.0 - prob)
    return prob * potential_profit - true_loss_probability * stake


# ---------------------------------------------------------------------------
# TheStatsAPI HTTP client
#
# Base URL: https://api.thestatsapi.com/api
# Auth: `Authorization: Bearer <THESTATSAPI_KEY>` on every request.
# `requests` is imported lazily so pure math / ingestion-from-dict usage
# (e.g. unit tests) never requires the dependency to be installed.
# ---------------------------------------------------------------------------


def _require_requests() -> Any:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without requests installed
        raise ImportError(
            "The 'requests' package is required for live TheStatsAPI calls. "
            "Install it with: pip install requests"
        ) from exc
    return requests


def thestatsapi_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


# ---------------------------------------------------------------------------
# Rate limiting / retry-with-backoff for real (non-test) TheStatsAPI calls.
#
# A cold cache day (the daily_model.py pipeline building models for a full,
# multi-competition slate from scratch) can mean hundreds of sequential
# roster/season-stats/historical-results calls, which is enough to trip
# TheStatsAPI's own rate limiting (429) well before the pipeline is done --
# observed in production. Both mitigations below live at this single choke
# point (every fetch_* function in this codebase ultimately calls
# `_thestatsapi_get`) so every caller benefits automatically:
#   1. A process-wide minimum interval between requests, paced up front
#      rather than reactively -- avoids tripping the limit in the first
#      place instead of just recovering from it.
#   2. Exponential backoff (honoring a `Retry-After` header when the server
#      sends one) with a bounded number of retries for the 429s that do
#      still happen, so a transient rate-limit window is ridden out
#      instead of immediately failing the entire fetch (which the
#      per-player/team/competition callers in daily_model.py would
#      otherwise have to just skip, per their own error-isolation).
#
# Both are skipped entirely when a caller passes an explicit `session`
# (this codebase's one and only DI/test seam -- never used by production
# code, only by tests substituting a fake HTTP client) so the test suite
# stays fast and deterministic; this is exercised directly against the
# `session is None` (real `requests`) branch instead, via monkeypatching
# `_require_requests`.
# ---------------------------------------------------------------------------

DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.6
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
DEFAULT_BACKOFF_MAX_SECONDS = 30.0
RATE_LIMIT_STATUS_CODE = 429


class _RateLimiter:
    """Thread-safe minimum-interval throttle (APScheduler's BackgroundScheduler runs jobs from a thread pool)."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._lock = threading.Lock()
        self._last_request_monotonic: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request_monotonic is not None:
                remaining = self.min_interval_seconds - (now - self._last_request_monotonic)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_monotonic = time.monotonic()


_rate_limiter = _RateLimiter(DEFAULT_MIN_REQUEST_INTERVAL_SECONDS)


def _retry_delay_seconds(response: Any, attempt: int) -> float:
    """
    Prefer the server's own Retry-After header (seconds form) when present;
    otherwise exponential backoff with jitter, so a burst of simultaneously
    rate-limited requests doesn't immediately re-collide on retry.
    """

    retry_after = getattr(response, "headers", None)
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after.get("Retry-After")))
        except (TypeError, ValueError):
            pass
    backoff = min(DEFAULT_BACKOFF_MAX_SECONDS, DEFAULT_BACKOFF_BASE_SECONDS * (2**attempt))
    return backoff + random.uniform(0, backoff * 0.25)


def _thestatsapi_get(
    path: str,
    api_key: str,
    *,
    params: dict[str, Any] | None = None,
    session: Any | None = None,
    timeout: float = 20.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    url = f"{THESTATSAPI_BASE_URL}{path}"
    headers = thestatsapi_headers(api_key)

    if session is not None:
        # Test/DI seam -- deliberately never throttled or retried, see the
        # module note above.
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    client = _require_requests()
    attempt = 0
    while True:
        _rate_limiter.wait()
        response = client.get(url, headers=headers, params=params, timeout=timeout)
        if getattr(response, "status_code", None) == RATE_LIMIT_STATUS_CODE and attempt < max_retries:
            delay = _retry_delay_seconds(response, attempt)
            logger.warning(
                "TheStatsAPI rate-limited %s (attempt %d/%d) -- backing off %.1fs.",
                path,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            attempt += 1
            continue
        response.raise_for_status()
        return response.json()


def thestatsapi_get(
    path: str,
    api_key: str,
    *,
    params: dict[str, Any] | None = None,
    session: Any | None = None,
    timeout: float = 20.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """
    Public, generic GET against TheStatsAPI for endpoints not already
    wrapped by a dedicated `fetch_thestatsapi_*` function below (e.g. the
    historical-results/player-season-stats pagination used by
    historical_data.py to build the projection model). Kept as a shared
    building block so every caller gets the same auth header, base URL,
    timeout, throttling, and 429-retry-with-backoff handling.
    """

    return _thestatsapi_get(path, api_key, params=params, session=session, timeout=timeout, max_retries=max_retries)


def is_thestatsapi_not_found(exc: Exception) -> bool:
    """
    True if `exc` is the HTTPError `thestatsapi_get`/`_thestatsapi_get` raise
    for a 404 response. Shared by every caller that needs to tell "this
    specific resource doesn't exist (yet)" apart from a real failure --
    e.g. lineups.py (a team sheet not announced yet) and
    historical_data.py (a player with no stats page for a given
    season/competition, such as a new signing) -- so a single missing
    resource is handled as an expected, named condition instead of an
    unhandled exception crashing the whole pipeline.
    """

    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def is_thestatsapi_addon_required(exc: Exception) -> bool:
    """
    True if `exc` is the HTTPError `thestatsapi_get`/`_thestatsapi_get` raise
    for a 403 caused by a missing subscription add-on (observed in
    production on GET /football/matches/{id}/odds/players: TheStatsAPI
    returns `{"error": {"code": "ADDON_REQUIRED", ...}}` when the account's
    plan doesn't include player-level odds). This is NOT a transient
    condition -- no amount of retrying or backing off fixes a plan
    limitation, unlike a 429 -- so callers should treat it as "this data
    source is unavailable for now" and stop trying for the rest of the run,
    rather than retry it or (worse) keep paying for the roster/season-stats
    fetches that only exist to support grading it.
    """

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 403:
        return False
    try:
        body = response.json()
    except Exception:
        return False
    return isinstance(body, dict) and body.get("error", {}).get("code") == "ADDON_REQUIRED"


def fetch_thestatsapi_matches(
    api_key: str,
    *,
    competition_id: str | None = None,
    season_id: str | None = None,
    team_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = None,
    page: int = 1,
    per_page: int = 20,
    session: Any | None = None,
) -> dict[str, Any]:
    """GET /football/matches - list upcoming or historical fixtures."""

    params = {
        "competition_id": competition_id,
        "season_id": season_id,
        "team_id": team_id,
        "date_from": date_from,
        "date_to": date_to,
        "status": status,
        "page": page,
        "per_page": per_page,
    }
    params = {key: value for key, value in params.items() if value is not None}
    return _thestatsapi_get("/football/matches", api_key, params=params, session=session)


def fetch_thestatsapi_match(api_key: str, match_id: str, *, session: Any | None = None) -> dict[str, Any]:
    """GET /football/matches/{match_id}."""

    return _thestatsapi_get(f"/football/matches/{match_id}", api_key, session=session)


def fetch_thestatsapi_match_odds(
    api_key: str,
    match_id: str,
    *,
    live: bool = False,
    session: Any | None = None,
) -> dict[str, Any]:
    """GET /football/matches/{match_id}/odds (or /odds/live for in-play prices)."""

    suffix = "/odds/live" if live else "/odds"
    return _thestatsapi_get(f"/football/matches/{match_id}{suffix}", api_key, session=session)


def fetch_thestatsapi_player_odds(
    api_key: str,
    match_id: str,
    *,
    markets: str | None = None,
    bookmaker: str = "bet365",
    session: Any | None = None,
) -> dict[str, Any]:
    """
    GET /football/matches/{match_id}/odds/players.

    `markets` is a comma-separated subset of: anytime_goalscorer, first_goalscorer,
    player_shots, player_shots_on_target, player_assists. Omit for all markets.
    """

    params: dict[str, Any] = {"bookmaker": bookmaker}
    if markets:
        params["markets"] = markets
    return _thestatsapi_get(f"/football/matches/{match_id}/odds/players", api_key, params=params, session=session)


# ---------------------------------------------------------------------------
# Ingestion & implied probability (TheStatsAPI response -> MarketLeg)
# ---------------------------------------------------------------------------


def flatten_thestatsapi_match_odds(
    match_id: str,
    payload: dict[str, Any],
    *,
    home_team: str | None = None,
    away_team: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    live: bool = False,
) -> list[MarketLeg]:
    """
    Normalize a `/football/matches/{match_id}/odds` (or `/odds/live`) response
    into MarketLeg objects covering Match Result, BTTS, Total Goals, Corners,
    and Asian Handicap. Odds are quoted as decimals by TheStatsAPI.

    Team names/IDs are optional context (pull them from
    `fetch_thestatsapi_match`); without them, legs fall back to generic
    "home"/"away" labels and omit id metadata. `home_team_id`/`away_team_id`
    are stashed in `leg.metadata` specifically so a projection model (e.g.
    a Dixon-Coles team-strength model keyed by team_id) can join back to its
    own ratings without re-deriving an identity from a display name.
    """

    data = payload.get("data", payload)
    resolved_match_id = str(data.get("match_id") or match_id)
    side_labels = {"home": home_team or "home", "away": away_team or "away", "draw": "Draw"}
    team_id_metadata = {"home_team_id": home_team_id, "away_team_id": away_team_id}
    legs: list[MarketLeg] = []

    for bookmaker in data.get("bookmakers", []) or []:
        sportsbook = str(bookmaker.get("bookmaker") or "")
        markets = bookmaker.get("markets", {}) or {}

        for side, prices in (markets.get("match_odds") or {}).items():
            price = _thestatsapi_price(prices, live=live)
            # Ruthless missing-data policy: if this bookmaker has no
            # opening/last_seen/live price for this side, that outcome is
            # dropped outright. We never fall back to another bookmaker's
            # price, a prior snapshot, or an interpolated value here --
            # a silently-substituted price would corrupt the EV calculation
            # against a number nobody can actually bet at.
            if price is None:
                continue
            legs.append(
                MarketLeg(
                    game_id=resolved_match_id,
                    event_id=resolved_match_id,
                    sportsbook=sportsbook,
                    market_type="match_odds",
                    selection=side_labels.get(side, side),
                    odds=price,
                    odds_format="decimal",
                    side=side,
                    team=side_labels.get(side) if side in {"home", "away"} else None,
                    metadata={"home_team": home_team, "away_team": away_team, "live": live, **team_id_metadata},
                )
            )

        for side, prices in (markets.get("btts") or {}).items():
            price = _thestatsapi_price(prices, live=live)
            if price is None:
                continue
            legs.append(
                MarketLeg(
                    game_id=resolved_match_id,
                    event_id=resolved_match_id,
                    sportsbook=sportsbook,
                    market_type="btts",
                    selection=f"BTTS {side.title()}",
                    odds=price,
                    odds_format="decimal",
                    side=side,
                    metadata={"home_team": home_team, "away_team": away_team, "live": live, **team_id_metadata},
                )
            )

        for market_key, label in (("total_goals", "Total goals"), ("match_corners", "Match corners")):
            for line_key, sides in (markets.get(market_key) or {}).items():
                line = _parse_line_key(line_key)
                for side, prices in (sides or {}).items():
                    price = _thestatsapi_price(prices, live=live)
                    if price is None:
                        continue
                    legs.append(
                        MarketLeg(
                            game_id=resolved_match_id,
                            event_id=resolved_match_id,
                            sportsbook=sportsbook,
                            market_type=market_key,
                            selection=f"{label} {side.title()} {line_key}".strip(),
                            odds=price,
                            odds_format="decimal",
                            line=line,
                            side=side,
                            metadata={"home_team": home_team, "away_team": away_team, "live": live, **team_id_metadata},
                        )
                    )

        for side, lines in (markets.get("asian_handicap") or {}).items():
            team_label = side_labels.get(side, side)
            for line_key, prices in (lines or {}).items():
                price = _thestatsapi_price(prices, live=live)
                if price is None:
                    continue
                legs.append(
                    MarketLeg(
                        game_id=resolved_match_id,
                        event_id=resolved_match_id,
                        sportsbook=sportsbook,
                        market_type="asian_handicap",
                        selection=f"{team_label} {line_key} AH".strip(),
                        odds=price,
                        odds_format="decimal",
                        line=_parse_line_key(line_key),
                        side=side,
                        team=team_label if side in {"home", "away"} else None,
                        metadata={"home_team": home_team, "away_team": away_team, "live": live, **team_id_metadata},
                    )
                )

    return legs


def flatten_thestatsapi_player_odds(
    match_id: str,
    payload: dict[str, Any],
    *,
    home_team: str | None = None,
    away_team: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
) -> list[MarketLeg]:
    """
    Normalize a `/football/matches/{match_id}/odds/players` response into
    MarketLeg objects. Covers anytime_goalscorer, first_goalscorer,
    player_shots, player_shots_on_target, and player_assists.
    """

    data = payload.get("data", payload)
    resolved_match_id = str(data.get("match_id") or match_id)
    sportsbook = str(data.get("bookmaker") or "")
    resolved_home = home_team or (data.get("home_team") or {}).get("name")
    resolved_away = away_team or (data.get("away_team") or {}).get("name")
    legs: list[MarketLeg] = []

    for market in data.get("markets", []) or []:
        market_name = str(market.get("name") or "")
        for player in market.get("players", []) or []:
            odd = player.get("odd")
            # Same ruthless policy as the team-odds flattener above: a
            # player-prop entry with a null price is dropped, not imputed
            # from a season-average price or a different bookmaker's number.
            if odd is None:
                continue

            player_name = str(player.get("name") or "")
            market_type = player.get("market_type")
            line = _maybe_float(player.get("line"))

            selection_parts = [player_name]
            if market_type:
                selection_parts.append(str(market_type))
            if line is not None:
                selection_parts.append(str(line))

            legs.append(
                MarketLeg(
                    game_id=resolved_match_id,
                    event_id=resolved_match_id,
                    sportsbook=sportsbook,
                    market_type=market_name,
                    selection=" ".join(selection_parts),
                    odds=float(odd),
                    odds_format="decimal",
                    line=line,
                    side=str(market_type).lower() if market_type else None,
                    entity_name=player_name or None,
                    entity_id=str(player["id"]) if player.get("id") is not None else None,
                    metadata={
                        "home_team": resolved_home,
                        "away_team": resolved_away,
                        "home_team_id": home_team_id,
                        "away_team_id": away_team_id,
                    },
                )
            )

    return legs


def normalize_market_data(
    raw_markets: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    match_id: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    live: bool = False,
) -> list[MarketLeg]:
    """
    Normalize sportsbook market data into MarketLeg objects.

    Accepts either:
    - A TheStatsAPI match-odds payload (`/football/matches/{id}/odds[/live]`),
      detected by a `data.bookmakers` list.
    - A TheStatsAPI player-prop odds payload (`/football/matches/{id}/odds/players`),
      detected by `data.bookmaker` + `data.markets`.
    - Generic rows with keys like game_id, market_type, selection, odds, sportsbook,
      for manually curated bets or other data sources.
    """

    if isinstance(raw_markets, dict):
        data = raw_markets.get("data", raw_markets)
        if isinstance(data, dict) and "bookmakers" in data:
            return flatten_thestatsapi_match_odds(
                match_id or str(data.get("match_id") or ""),
                raw_markets,
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                live=live,
            )
        if isinstance(data, dict) and "bookmaker" in data and "markets" in data:
            return flatten_thestatsapi_player_odds(
                match_id or str(data.get("match_id") or ""),
                raw_markets,
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
        return [_leg_from_generic_row(raw_markets)]

    return [_leg_from_generic_row(row) for row in raw_markets]


# ---------------------------------------------------------------------------
# EV calculation engine
# ---------------------------------------------------------------------------


def evaluate_leg(
    leg: MarketLeg,
    probability_provider: TrueProbabilityProvider | Callable[[MarketLeg], ProbabilityEstimate | float],
    *,
    stake: float = DEFAULT_STAKE,
    min_ev: float = MIN_POSITIVE_EV,
    min_edge: float = DEFAULT_MIN_EDGE,
) -> EVResult:
    """
    Grade exactly one (odds leg, model probability) pair. This is the single
    point in the whole engine where the odds side (`leg`) and the
    probability side (`probability_provider` output) are allowed to meet --
    see the module-level architecture contract at the top of this file.
    """

    # `probability_provider` is the hook into your existing projection model.
    # It is called with ONLY the leg (selection/market metadata), never with
    # the leg's own odds, so the model cannot accidentally anchor its
    # probability estimate on the market's price.
    estimate = _coerce_probability_estimate(probability_provider(leg))
    true_prob = clamp_probability(estimate.true_probability)
    resolved_loss_prob = (
        clamp_probability(estimate.loss_probability) if estimate.loss_probability is not None else (1.0 - true_prob)
    )
    decimal = odds_to_decimal(leg.odds, leg.odds_format)
    implied = decimal_to_implied(decimal)
    ev = expected_value(true_prob, decimal, stake=stake, loss_probability=resolved_loss_prob)
    edge = true_prob - implied
    sample_size = estimate.sample_size if estimate.sample_size is not None else leg.sample_size
    volatility = estimate.volatility if estimate.volatility is not None else leg.volatility
    if volatility is None:
        # Not a fabricated stat about THIS leg -- a documented, per-market-
        # family baseline volatility used only to size the confidence-score
        # penalty when neither the model nor the ingestion pipeline supplied
        # a measured volatility. This never touches odds, probability, or EV.
        volatility = MARKET_VOLATILITY.get(leg.market_family, 0.35)
    warnings = [*estimate.warnings]

    # Sample-size and volatility warnings feed the confidence score, not the
    # EV math itself -- a thin-sample or volatile market can still carry a
    # mathematically positive EV; it's just less trustworthy, which is a
    # confidence concern, not a probability/EV correctness concern.
    if sample_size is not None and sample_size < 10:
        warnings.append("low sample size")
    if volatility >= 0.45:
        warnings.append("high volatility market")

    positive = ev > min_ev and edge >= min_edge
    score = confidence_score(
        edge=edge,
        ev_per_unit=ev / stake,
        true_probability=true_prob,
        sample_size=sample_size,
        volatility=volatility,
        warnings=warnings,
    )

    return EVResult(
        leg=leg,
        implied_probability=implied,
        true_probability=true_prob,
        edge=edge,
        edge_pct=edge * 100.0,
        decimal_odds=decimal,
        potential_profit=stake * (decimal - 1.0),
        ev=ev,
        ev_per_unit=ev / stake,
        positive_ev=positive,
        confidence_score=score,
        loss_probability=resolved_loss_prob,
        sample_size=sample_size,
        volatility=volatility,
        warnings=warnings,
    )


def rank_ev_board(
    raw_markets: Iterable[dict[str, Any]] | dict[str, Any] | Iterable[MarketLeg],
    probability_provider: TrueProbabilityProvider | Callable[[MarketLeg], ProbabilityEstimate | float],
    *,
    stake: float = DEFAULT_STAKE,
    min_ev: float = MIN_POSITIVE_EV,
    min_edge: float = DEFAULT_MIN_EDGE,
    positive_only: bool = True,
    apply_correlation_filter: bool = True,
    max_per_game: int | None = None,
    match_id: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
    home_team_id: str | None = None,
    away_team_id: str | None = None,
    live: bool = False,
) -> list[EVResult]:
    raw_list = list(raw_markets) if not isinstance(raw_markets, dict) else raw_markets
    if isinstance(raw_list, list) and raw_list and all(isinstance(row, MarketLeg) for row in raw_list):
        legs = raw_list
    else:
        legs = normalize_market_data(
            raw_list,  # type: ignore[arg-type]
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            live=live,
        )

    results = []
    for leg in legs:
        try:
            results.append(
                evaluate_leg(
                    leg,
                    probability_provider,
                    stake=stake,
                    min_ev=min_ev,
                    min_edge=min_edge,
                )
            )
        except Exception:
            # One malformed leg (e.g. a sportsbook feed glitch producing
            # decimal odds <= 1.0) must not cost every OTHER leg on this
            # same match/board its grade -- a match's board going from "9
            # legs graded, 1 skipped" to "0 legs graded, no alert at all"
            # over a single bad data point is exactly the kind of ruthless
            # missing-data handling this engine is supposed to have.
            logger.exception("Failed to evaluate leg %r; skipping.", leg)
    if positive_only:
        results = [result for result in results if result.positive_ev]

    results.sort(key=lambda result: (result.confidence_score, result.ev_per_unit, result.edge), reverse=True)

    if apply_correlation_filter:
        results = filter_correlated_bets(results)

    if max_per_game is not None:
        results = _limit_per_game(results, max_per_game)

    return results


def build_match_ev_board(
    api_key: str,
    match_id: str,
    probability_provider: TrueProbabilityProvider | Callable[[MarketLeg], ProbabilityEstimate | float],
    *,
    include_player_props: bool = True,
    live: bool = False,
    player_markets: str | None = None,
    session: Any | None = None,
    **rank_kwargs: Any,
) -> list[EVResult]:
    """
    End-to-end bridge: fetch one match's TheStatsAPI odds board (team + player
    markets), normalize every leg, and rank it against your projection model's
    true probabilities. This is the main entry point tying the model to live
    sportsbook data.
    """

    match_payload = fetch_thestatsapi_match(api_key, match_id, session=session)
    match_data = match_payload.get("data", match_payload)
    home_team_obj = match_data.get("home_team") or {}
    away_team_obj = match_data.get("away_team") or {}
    home_team = home_team_obj.get("name")
    away_team = away_team_obj.get("name")
    home_team_id = str(home_team_obj["id"]) if home_team_obj.get("id") is not None else None
    away_team_id = str(away_team_obj["id"]) if away_team_obj.get("id") is not None else None

    odds_payload = fetch_thestatsapi_match_odds(api_key, match_id, live=live, session=session)
    legs = flatten_thestatsapi_match_odds(
        match_id,
        odds_payload,
        home_team=home_team,
        away_team=away_team,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        live=live,
    )

    if include_player_props:
        player_payload = fetch_thestatsapi_player_odds(api_key, match_id, markets=player_markets, session=session)
        legs.extend(
            flatten_thestatsapi_player_odds(
                match_id,
                player_payload,
                home_team=home_team,
                away_team=away_team,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
        )

    return rank_ev_board(legs, probability_provider, **rank_kwargs)


# ---------------------------------------------------------------------------
# AI confidence score (1-100)
# ---------------------------------------------------------------------------


def confidence_score(
    *,
    edge: float,
    ev_per_unit: float,
    true_probability: float,
    sample_size: int | None,
    volatility: float,
    warnings: list[str] | None = None,
) -> int:
    """
    Score 1-100, emphasizing model-vs-market edge while penalizing unstable inputs.
    """

    edge_points = max(0.0, edge) * 500.0
    ev_points = max(0.0, ev_per_unit) * 120.0
    probability_points = max(0.0, true_probability - 0.50) * 25.0
    base = 35.0 + edge_points + ev_points + probability_points

    if sample_size is None:
        sample_penalty = 8.0
    elif sample_size < 5:
        sample_penalty = 22.0
    elif sample_size < 10:
        sample_penalty = 14.0
    elif sample_size < 25:
        sample_penalty = 6.0
    else:
        sample_penalty = 0.0

    volatility_penalty = max(0.0, volatility - 0.20) * 45.0
    warning_penalty = min(12.0, 4.0 * len(warnings or []))
    score = round(base - sample_penalty - volatility_penalty - warning_penalty)
    return min(100, max(1, score))


# ---------------------------------------------------------------------------
# Correlation filter (de-duplicator)
# ---------------------------------------------------------------------------


def filter_correlated_bets(results: Iterable[EVResult]) -> list[EVResult]:
    """
    Keep the highest-scoring result per correlated game/player/team market cluster.

    This removes redundant legs like "Lamine Yamal 3+ Shots" and
    "Lamine Yamal 1+ SOT" from the same top-ranked board.
    """

    ordered = sorted(results, key=lambda result: (result.confidence_score, result.ev_per_unit), reverse=True)
    kept: list[EVResult] = []
    seen: set[tuple[str, str, str]] = set()

    for result in ordered:
        key = correlation_key(result.leg)
        if key in seen:
            continue
        kept.append(result)
        seen.add(key)

    return kept


def correlation_key(leg: MarketLeg) -> tuple[str, str, str]:
    if leg.correlation_group:
        return (leg.game_id, leg.participant_key, _slug(leg.correlation_group))
    return (leg.game_id, leg.participant_key, leg.market_family)


def market_family(market_type: str) -> str:
    """
    Collapse a raw market_type string into a coarse family used for
    correlation grouping (see `correlation_key`). This is deliberately
    coarse: "player_shots" and "player_shots_on_target" both map to
    "shots" on purpose, because a player's shots-on-target count and total
    shots count are driven by the same underlying event stream for that
    player in that game -- betting both is functionally the same wager
    twice, which is exactly the redundancy the correlation filter exists to
    remove.
    """

    market = _slug(market_type)
    if market in {"h2h", "moneyline", "winner", "match_winner", "draw_no_bet", "match_odds"}:
        return "moneyline"
    if "btts" in market or "both_teams" in market:
        return "btts"
    if "goalscorer" in market:
        return "goalscorer"
    if "shot_on_target" in market or market.endswith("sot") or "sot" in market:
        return "shots"
    if "shot" in market:
        return "shots"
    if "assist" in market:
        return "assists"
    if "pass" in market:
        return "passes"
    if "tackle" in market:
        return "tackles"
    if "spread" in market or "handicap" in market:
        return "spread"
    if "total" in market or "goals" in market:
        return "total"
    if "corner" in market:
        return "corners"
    if "card" in market or "booking" in market:
        return "cards"
    return market or "unknown"


def clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _coerce_probability_estimate(value: ProbabilityEstimate | float) -> ProbabilityEstimate:
    if isinstance(value, ProbabilityEstimate):
        return value
    return ProbabilityEstimate(true_probability=float(value))


def _leg_from_generic_row(row: dict[str, Any]) -> MarketLeg:
    odds = row.get("odds")
    if odds is None:
        raise ValueError(f"Missing odds for market row: {row}")

    return MarketLeg(
        game_id=str(row.get("game_id") or row.get("event_id") or ""),
        event_id=str(row.get("event_id")) if row.get("event_id") is not None else None,
        sportsbook=str(row.get("sportsbook") or row.get("bookmaker") or ""),
        market_type=str(row.get("market_type") or row.get("market") or row.get("market_key") or ""),
        market_key=str(row.get("market_key")) if row.get("market_key") is not None else None,
        selection=str(row.get("selection") or row.get("pick") or row.get("outcome_name") or ""),
        outcome_name=str(row.get("outcome_name")) if row.get("outcome_name") is not None else None,
        odds=float(odds),
        odds_format=str(row.get("odds_format") or "american").lower(),  # type: ignore[arg-type]
        line=_maybe_float(row.get("line") if row.get("line") is not None else row.get("point")),
        side=str(row.get("side")) if row.get("side") is not None else None,
        entity_name=str(row.get("entity_name") or row.get("player_name") or "") or None,
        entity_id=str(row.get("entity_id") or row.get("player_id") or "") or None,
        team=str(row.get("team") or "") or None,
        last_update=str(row.get("last_update")) if row.get("last_update") is not None else None,
        sample_size=int(row["sample_size"]) if row.get("sample_size") is not None else None,
        volatility=_maybe_float(row.get("volatility")),
        correlation_group=str(row.get("correlation_group"))
        if row.get("correlation_group") is not None
        else None,
        metadata=dict(row.get("metadata") or {}),
    )


def _limit_per_game(results: Iterable[EVResult], max_per_game: int) -> list[EVResult]:
    counts: dict[str, int] = {}
    limited: list[EVResult] = []
    for result in results:
        current = counts.get(result.leg.game_id, 0)
        if current >= max_per_game:
            continue
        limited.append(result)
        counts[result.leg.game_id] = current + 1
    return limited


def _thestatsapi_price(prices: Any, *, live: bool) -> float | None:
    if not isinstance(prices, dict):
        return None
    if live:
        value = prices.get("live", prices.get("last_seen", prices.get("opening")))
    else:
        value = prices.get("last_seen", prices.get("opening", prices.get("live")))
    return _maybe_float(value)


def _parse_line_key(value: Any) -> float | None:
    try:
        return float(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _slug(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
