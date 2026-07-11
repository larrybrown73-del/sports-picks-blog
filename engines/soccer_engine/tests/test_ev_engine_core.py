from __future__ import annotations

import math

import ev_engine_core as ev_engine_core_module
from ev_engine_core import (
    MarketLeg,
    ProbabilityEstimate,
    american_to_decimal,
    american_to_implied,
    devig_probabilities,
    expected_value,
    filter_correlated_bets,
    flatten_thestatsapi_match_odds,
    flatten_thestatsapi_player_odds,
    get_thestatsapi_key,
    implied_probability,
    is_thestatsapi_not_found,
    market_overround,
    normalize_market_data,
    odds_to_decimal,
    rank_ev_board,
    thestatsapi_get,
)


class _FakeRateLimitedResponse:
    def __init__(self, status_code: int, *, json_data: dict | None = None, retry_after: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after is not None else {}
        self._json_data = json_data or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


class _FakeRealClient:
    """Stands in for the real `requests` module (what `_require_requests()` normally returns)."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, *, headers=None, params=None, timeout=None):
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response


def test_thestatsapi_get_retries_429_then_succeeds(monkeypatch) -> None:
    """
    Exercises the `session is None` (real-network) branch of
    `_thestatsapi_get`, which is where throttling/retry-with-backoff live
    -- monkeypatches `_require_requests` (rather than passing a `session`)
    so that branch actually runs, and `time.sleep` so the test doesn't
    really wait through the backoff.
    """

    client = _FakeRealClient(
        [
            _FakeRateLimitedResponse(429, retry_after="0"),
            _FakeRateLimitedResponse(429, retry_after="0"),
            _FakeRateLimitedResponse(200, json_data={"data": "ok"}),
        ]
    )
    monkeypatch.setattr(ev_engine_core_module, "_require_requests", lambda: client)
    monkeypatch.setattr(ev_engine_core_module._rate_limiter, "min_interval_seconds", 0.0)  # isolate backoff sleeps only
    sleep_calls = []
    monkeypatch.setattr(ev_engine_core_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = thestatsapi_get("/football/matches", "test_key")

    assert result == {"data": "ok"}
    assert client.calls == 3
    assert len(sleep_calls) == 2  # one backoff sleep per 429 before the eventual success


def test_thestatsapi_get_gives_up_after_max_retries(monkeypatch) -> None:
    always_429 = _FakeRealClient([_FakeRateLimitedResponse(429, retry_after="0")])
    monkeypatch.setattr(ev_engine_core_module, "_require_requests", lambda: always_429)
    monkeypatch.setattr(ev_engine_core_module._rate_limiter, "min_interval_seconds", 0.0)
    monkeypatch.setattr(ev_engine_core_module.time, "sleep", lambda seconds: None)

    try:
        thestatsapi_get("/football/matches", "test_key", max_retries=2)
        raise AssertionError("expected the persistent 429 to eventually raise")
    except RuntimeError as exc:
        assert "429" in str(exc)
    assert always_429.calls == 3  # initial attempt + 2 retries, then give up


def test_thestatsapi_get_does_not_retry_non_429_errors(monkeypatch) -> None:
    client = _FakeRealClient([_FakeRateLimitedResponse(500)])
    monkeypatch.setattr(ev_engine_core_module, "_require_requests", lambda: client)
    monkeypatch.setattr(ev_engine_core_module._rate_limiter, "min_interval_seconds", 0.0)
    sleep_calls = []
    monkeypatch.setattr(ev_engine_core_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    try:
        thestatsapi_get("/football/matches", "test_key")
        raise AssertionError("expected the 500 to propagate immediately")
    except RuntimeError:
        pass
    assert client.calls == 1  # no retry loop for a non-429 error
    assert sleep_calls == []  # only the (skipped, since calls==1) rate-limiter pacing sleep would fire here


def test_thestatsapi_get_with_explicit_session_skips_throttle_and_retry(monkeypatch) -> None:
    """A caller-supplied session (the test/DI seam) must never be throttled or retried -- see the module note."""

    class _ImmediateFailSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, *, headers=None, params=None, timeout=None):
            self.calls += 1
            return _FakeRateLimitedResponse(429)

    session = _ImmediateFailSession()
    sleep_calls = []
    monkeypatch.setattr(ev_engine_core_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    try:
        thestatsapi_get("/football/matches", "test_key", session=session)
        raise AssertionError("expected the 429 to propagate immediately for an explicit session")
    except RuntimeError:
        pass
    assert session.calls == 1  # no retry loop when a session is explicitly provided
    assert sleep_calls == []


def test_odds_conversion_and_expected_value() -> None:
    assert math.isclose(american_to_decimal(-110), 1.909090909, rel_tol=1e-6)
    assert math.isclose(american_to_implied(150), 0.4, rel_tol=1e-6)
    assert math.isclose(odds_to_decimal(2.5, "decimal"), 2.5, rel_tol=1e-6)

    ev = expected_value(0.55, american_to_decimal(-110), stake=1.0)
    assert ev > 0
    assert math.isclose(ev, 0.05, rel_tol=1e-2)


def test_is_thestatsapi_not_found() -> None:
    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    error_404 = Exception("not found")
    error_404.response = _Response(404)
    assert is_thestatsapi_not_found(error_404) is True

    error_500 = Exception("server error")
    error_500.response = _Response(500)
    assert is_thestatsapi_not_found(error_500) is False

    assert is_thestatsapi_not_found(Exception("no response attribute")) is False


def test_is_thestatsapi_addon_required() -> None:
    class _Response:
        def __init__(self, status_code: int, body: dict) -> None:
            self.status_code = status_code
            self._body = body

        def json(self) -> dict:
            return self._body

    addon_error = Exception("forbidden")
    addon_error.response = _Response(
        403, {"error": {"code": "ADDON_REQUIRED", "message": "player_odds add-on required"}}
    )
    assert ev_engine_core_module.is_thestatsapi_addon_required(addon_error) is True

    # A 403 for some other reason (bad auth, etc.) must not be mistaken for this.
    other_403 = Exception("forbidden")
    other_403.response = _Response(403, {"error": {"code": "INVALID_TOKEN"}})
    assert ev_engine_core_module.is_thestatsapi_addon_required(other_403) is False

    not_403 = Exception("not found")
    not_403.response = _Response(404, {})
    assert ev_engine_core_module.is_thestatsapi_addon_required(not_403) is False

    assert ev_engine_core_module.is_thestatsapi_addon_required(Exception("no response attribute")) is False


def test_flatten_thestatsapi_match_odds() -> None:
    payload = {
        "data": {
            "match_id": "mt_1",
            "bookmakers": [
                {
                    "bookmaker": "Pinnacle",
                    "markets": {
                        "match_odds": {
                            "home": {"opening": "1.750", "last_seen": "1.810"},
                            "draw": {"opening": "3.500", "last_seen": "3.400"},
                            "away": {"opening": "4.500", "last_seen": "4.200"},
                        },
                        "btts": {
                            "yes": {"opening": "1.700", "last_seen": "1.720"},
                            "no": {"opening": "2.100", "last_seen": "2.050"},
                        },
                        "total_goals": {
                            "2.5": {
                                "over": {"opening": "1.900", "last_seen": "1.950"},
                                "under": {"opening": "1.950", "last_seen": "1.900"},
                            }
                        },
                        "asian_handicap": {
                            "home": {"-0.5": {"opening": "1.900", "last_seen": "1.850"}},
                            "away": {"+0.5": {"opening": "1.950", "last_seen": "2.000"}},
                        },
                    },
                }
            ],
        }
    }

    legs = flatten_thestatsapi_match_odds("mt_1", payload, home_team="Spain", away_team="Portugal")

    # 3 match_odds + 2 btts + 2 total_goals (over/under) + 2 asian_handicap = 9
    assert len(legs) == 9

    home_leg = next(leg for leg in legs if leg.market_type == "match_odds" and leg.side == "home")
    assert home_leg.selection == "Spain"
    assert home_leg.odds == 1.81
    assert home_leg.odds_format == "decimal"

    over_leg = next(leg for leg in legs if leg.market_type == "total_goals" and leg.side == "over")
    assert over_leg.line == 2.5
    assert math.isclose(over_leg.odds, 1.95)

    ah_leg = next(leg for leg in legs if leg.market_type == "asian_handicap" and leg.side == "home")
    assert ah_leg.line == -0.5
    assert ah_leg.team == "Spain"


def test_normalize_market_data_detects_thestatsapi_match_odds() -> None:
    payload = {
        "data": {
            "match_id": "mt_2",
            "bookmakers": [
                {
                    "bookmaker": "Bet365",
                    "markets": {
                        "match_odds": {
                            "home": {"live": "1.900"},
                            "draw": {"live": "3.500"},
                            "away": {"live": "4.000"},
                        }
                    },
                }
            ],
        }
    }

    legs = normalize_market_data(payload, live=True)
    assert len(legs) == 3
    assert all(leg.odds_format == "decimal" for leg in legs)
    assert all(leg.game_id == "mt_2" for leg in legs)


def test_flatten_thestatsapi_player_odds() -> None:
    payload = {
        "data": {
            "match_id": "mt_3",
            "bookmaker": "bet365",
            "markets": [
                {
                    "name": "anytime_goalscorer",
                    "players": [
                        {"id": "pl_1", "name": "Lamine Yamal", "line": 0.5, "market_type": None, "odd": 1.85}
                    ],
                },
                {
                    "name": "player_shots",
                    "players": [
                        {"id": "pl_1", "name": "Lamine Yamal", "line": 2.5, "market_type": "Over", "odd": 2.4}
                    ],
                },
                {
                    "name": "player_shots_on_target",
                    "players": [
                        {"id": "pl_1", "name": "Lamine Yamal", "line": 0.5, "market_type": "Over", "odd": 1.5}
                    ],
                },
            ],
        }
    }

    legs = flatten_thestatsapi_player_odds("mt_3", payload)

    assert len(legs) == 3
    shots_leg = next(leg for leg in legs if leg.market_type == "player_shots")
    assert shots_leg.entity_name == "Lamine Yamal"
    assert shots_leg.line == 2.5
    assert shots_leg.odds_format == "decimal"


def test_rank_ev_board_returns_positive_ev_sorted() -> None:
    legs = [
        MarketLeg(
            game_id="mt_4", market_type="match_odds", selection="Spain", odds=1.90,
            odds_format="decimal", sportsbook="Bet365", team="Spain", side="home", sample_size=40,
        ),
        MarketLeg(
            game_id="mt_4", market_type="match_odds", selection="Portugal", odds=4.00,
            odds_format="decimal", sportsbook="Bet365", team="Portugal", side="away", sample_size=40,
        ),
    ]

    def provider(leg: MarketLeg) -> ProbabilityEstimate:
        if leg.selection == "Spain":
            return ProbabilityEstimate(true_probability=0.60, sample_size=40, volatility=0.20)
        return ProbabilityEstimate(true_probability=0.15, sample_size=40, volatility=0.20)

    results = rank_ev_board(legs, provider)

    assert len(results) == 1
    assert results[0].leg.selection == "Spain"
    assert results[0].positive_ev is True


def test_rank_ev_board_skips_one_malformed_leg_without_losing_the_rest() -> None:
    """
    A single leg with unusable odds (e.g. a sportsbook feed glitch sending
    decimal odds <= 1.0) must not crash grading for every other leg on the
    same board -- observed in production: one bad leg on a match otherwise
    took its whole board down to zero results.
    """

    legs = [
        MarketLeg(
            game_id="mt_5", market_type="match_odds", selection="Spain", odds=1.90,
            odds_format="decimal", sportsbook="Bet365", team="Spain", side="home", sample_size=40,
        ),
        MarketLeg(
            game_id="mt_5", market_type="match_odds", selection="Draw", odds=0.5,  # invalid: must be > 1.0
            odds_format="decimal", sportsbook="Bet365", side="draw", sample_size=40,
        ),
    ]

    def provider(leg: MarketLeg) -> ProbabilityEstimate:
        if leg.selection == "Spain":
            return ProbabilityEstimate(true_probability=0.60, sample_size=40, volatility=0.20)
        return ProbabilityEstimate(true_probability=0.20, sample_size=40, volatility=0.20)

    results = rank_ev_board(legs, provider)

    assert len(results) == 1  # the malformed "Draw" leg was skipped, not fatal
    assert results[0].leg.selection == "Spain"
    assert results[0].edge_pct > 0
    assert results[0].confidence_score > 50


def test_confidence_penalizes_low_sample_high_volatility() -> None:
    strong_leg = MarketLeg(
        game_id="mt_5", market_type="player_assists", selection="Lamine Yamal Over 0.5 assists",
        odds=1.95, odds_format="decimal", sportsbook="bet365", entity_name="Lamine Yamal",
        line=0.5, sample_size=50, volatility=0.25,
    )
    volatile_leg = MarketLeg(
        game_id="mt_5", market_type="player_shots_on_target", selection="Lamine Yamal 1+ SOT",
        odds=1.95, odds_format="decimal", sportsbook="bet365", entity_name="Lamine Yamal",
        line=0.5, sample_size=4, volatility=0.55,
    )

    def provider(_: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.60)

    results = rank_ev_board([strong_leg, volatile_leg], provider, apply_correlation_filter=False)
    by_market = {result.leg.market_type: result for result in results}

    assert len(results) == 2
    assert by_market["player_assists"].confidence_score > by_market["player_shots_on_target"].confidence_score
    assert "low sample size" in by_market["player_shots_on_target"].warnings
    assert "high volatility market" in by_market["player_shots_on_target"].warnings


def test_correlation_filter_keeps_best_player_shots_leg() -> None:
    legs = [
        MarketLeg(
            game_id="mt_6", market_type="player_shots", selection="Lamine Yamal 3+ Shots",
            odds=2.40, odds_format="decimal", sportsbook="bet365", entity_name="Lamine Yamal",
            line=2.5, sample_size=35, volatility=0.35,
        ),
        MarketLeg(
            game_id="mt_6", market_type="player_shots_on_target", selection="Lamine Yamal 1+ SOT",
            odds=1.91, odds_format="decimal", sportsbook="bet365", entity_name="Lamine Yamal",
            line=0.5, sample_size=35, volatility=0.45,
        ),
    ]

    def provider(leg: MarketLeg) -> ProbabilityEstimate:
        if leg.market_type == "player_shots":
            return ProbabilityEstimate(true_probability=0.56, sample_size=35, volatility=0.35)
        return ProbabilityEstimate(true_probability=0.58, sample_size=35, volatility=0.45)

    unfiltered = rank_ev_board(legs, provider, apply_correlation_filter=False)
    filtered = filter_correlated_bets(unfiltered)

    assert len(unfiltered) == 2
    assert len(filtered) == 1
    assert filtered[0].leg.selection == "Lamine Yamal 3+ Shots"


def test_market_overround_and_devig_sum_to_one() -> None:
    # A three-way 1X2 market priced with a realistic ~6% overround.
    home = implied_probability(1.90, "decimal")
    draw = implied_probability(3.60, "decimal")
    away = implied_probability(4.20, "decimal")

    overround = market_overround([home, draw, away])
    assert overround > 1.0  # every real sportsbook market carries some vig

    fair = devig_probabilities([home, draw, away])
    assert math.isclose(sum(fair), 1.0, rel_tol=1e-9)
    # Proportional de-vig preserves relative ordering of favorites/underdogs.
    assert fair[0] > fair[1] > fair[2]


def test_devig_probabilities_rejects_zero_total() -> None:
    try:
        devig_probabilities([0.0, 0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive total implied probability")


def test_expected_value_supports_explicit_push_probability() -> None:
    # A whole-number Asian Handicap: 55% win, 35% loss, 10% push. Omitting
    # the push from the formula (by NOT using 1 - true_probability as the
    # loss term) must change the EV relative to treating it as a strict
    # win/lose market.
    decimal_odds = 1.95
    strict_ev = expected_value(0.55, decimal_odds, stake=1.0)  # implicitly 45% loss
    push_aware_ev = expected_value(0.55, decimal_odds, stake=1.0, loss_probability=0.35)

    assert not math.isclose(strict_ev, push_aware_ev)
    # Less loss probability counted (0.35 vs 0.45) must mean less EV drag.
    assert push_aware_ev > strict_ev


def test_evaluate_leg_uses_probability_estimate_loss_probability() -> None:
    leg = MarketLeg(
        game_id="mt_20", market_type="asian_handicap", selection="Home -1.0 AH",
        odds=1.95, odds_format="decimal", sportsbook="Bet365", team="Home", side="home", line=-1.0,
        sample_size=30,
    )

    def provider(_: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.55, loss_probability=0.35, sample_size=30, volatility=0.20)

    result = rank_ev_board([leg], provider, positive_only=False, apply_correlation_filter=False)[0]
    assert math.isclose(result.loss_probability, 0.35)
    expected_ev = expected_value(0.55, 1.95, stake=1.0, loss_probability=0.35)
    assert math.isclose(result.ev, expected_ev, rel_tol=1e-12)


def test_thestatsapi_key_loader_reads_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("THESTATSAPI_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("THESTATSAPI_KEY=fapi_placeholder\n", encoding="utf-8")

    assert get_thestatsapi_key(env_file=env_file) == "fapi_placeholder"
