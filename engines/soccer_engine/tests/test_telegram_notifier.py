from __future__ import annotations

import math

import pytest

from ev_engine_core import MarketLeg, rank_ev_board, ProbabilityEstimate
from telegram_notifier import (
    TelegramConfigError,
    format_ev_board_message,
    get_telegram_credentials,
    send_telegram_message,
)


class _FakeResponse:
    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json_data


class _FakeSession:
    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json_data = json_data
        self._status_code = status_code
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict, timeout: float):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(self._json_data, self._status_code)


def test_get_telegram_credentials_reads_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("TELEGRAM_BOT_TOKEN=123:ABC\nTELEGRAM_CHAT_ID=456\n", encoding="utf-8")

    bot_token, chat_id = get_telegram_credentials(env_file=env_file)
    assert bot_token == "123:ABC"
    assert chat_id == "456"


def test_get_telegram_credentials_raises_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    missing_env_file = tmp_path / "does_not_exist.env"

    with pytest.raises(TelegramConfigError):
        get_telegram_credentials(env_file=missing_env_file)


def test_send_telegram_message_posts_expected_payload() -> None:
    session = _FakeSession({"ok": True, "result": {"message_id": 1}})
    result = send_telegram_message(
        "hello world", bot_token="123:ABC", chat_id="456", session=session, parse_mode="HTML"
    )

    assert result["ok"] is True
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://api.telegram.org/bot123:ABC/sendMessage"
    assert call["json"]["chat_id"] == "456"
    assert call["json"]["text"] == "hello world"
    assert call["json"]["parse_mode"] == "HTML"


def test_send_telegram_message_raises_on_application_level_failure() -> None:
    session = _FakeSession({"ok": False, "description": "chat not found"})
    with pytest.raises(RuntimeError):
        send_telegram_message("hello", bot_token="123:ABC", chat_id="456", session=session)


def test_send_telegram_message_chunks_long_text() -> None:
    session = _FakeSession({"ok": True})
    long_text = "\n".join(f"line {i}" for i in range(2000))
    send_telegram_message(long_text, bot_token="123:ABC", chat_id="456", session=session)

    assert len(session.calls) > 1
    for call in session.calls:
        assert len(call["json"]["text"]) <= 3500
    # Reassembling every chunk's text (minus the newlines we split on)
    # should reproduce every original line, in order, with none dropped.
    reassembled_lines = "\n".join(call["json"]["text"] for call in session.calls).split("\n")
    assert reassembled_lines == long_text.split("\n")


def _leg(**overrides) -> MarketLeg:
    defaults = dict(
        game_id="mt_1",
        market_type="match_odds",
        selection="Home Team",
        odds=2.2,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        team="Home Team",
    )
    defaults.update(overrides)
    return MarketLeg(**defaults)


def test_format_ev_board_message_filters_by_confidence_and_sorts_descending() -> None:
    high_confidence_leg = _leg(game_id="mt_1", selection="Home Team")
    low_confidence_leg = _leg(game_id="mt_2", selection="Away Team", side="away", team="Away Team")

    def high_provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.75, sample_size=200, volatility=0.1)

    def low_provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.48, sample_size=3, volatility=0.9)

    high_result = rank_ev_board([high_confidence_leg], high_provider, positive_only=False)[0]
    low_result = rank_ev_board([low_confidence_leg], low_provider, positive_only=False)[0]

    # Sanity-check the fixture actually produces a high/low split before
    # relying on it below (avoids a silently-vacuous assertion if the
    # underlying confidence formula ever changes).
    # Inclusive threshold: a leg that scores exactly the cutoff must still
    # appear. Previously this was a strict `>` which silently dropped
    # anything that landed on the boundary.
    assert high_result.confidence_score >= 50
    assert low_result.confidence_score < 50

    message = format_ev_board_message([high_result, low_result], title="Test Board", min_confidence_score=50)

    assert "Test Board" in message
    assert "Home Team" in message
    assert "Away Team" not in message


def test_format_ev_board_message_empty_results_says_no_bets() -> None:
    message = format_ev_board_message([], title="Empty Board", min_confidence_score=70)
    assert "Empty Board" in message
    assert "No bets scored at or above 70" in message


def test_format_ev_board_message_escapes_html_special_characters() -> None:
    # team feeds BOTH the match-context line (no metadata home/away here, so
    # it falls back to leg.team) and the bet-selection line ("... to Win")
    # -- selection= alone wouldn't exercise escaping any more, since
    # match_odds legs no longer render leg.selection directly (see
    # _bet_selection_line).
    leg = _leg(team="Team A & <B>")

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.75, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)
    assert "<B>" not in message
    assert "&amp;" in message
    assert math.isclose(result.true_probability, 0.75)


def test_format_ev_board_message_uses_html_parse_mode_friendly_markup() -> None:
    """Sanity-check the message actually contains the bold/italic tags send_telegram_message's default parse_mode=HTML expects."""

    leg = _leg(metadata={"home_team": "Home Team", "away_team": "Away Team"})

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.75, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)
    assert "<b>Home Team vs Away Team</b>" in message
    assert "<i>Match Result</i>" in message


def test_format_ev_board_message_does_not_repeat_team_name_for_match_odds() -> None:
    """
    Regression test for the original complaint: a match_odds leg's
    selection is literally just the team name, so the old formatter printed
    "**Home Team** \u2014 Home Team". The new format must show the team name
    exactly once per line, in two DIFFERENT lines conveying different info
    (which match, vs. which side is picked to win).
    """

    leg = _leg(metadata={"home_team": "Home Team", "away_team": "Away Team"})

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.75, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)

    assert "Home Team vs Away Team" in message
    assert "Home Team to Win" in message


def test_format_ev_board_message_strips_asian_handicap_suffix_and_shows_signed_line() -> None:
    leg = _leg(
        market_type="asian_handicap",
        selection="Home Team -1.5 AH",
        line=-1.5,
        metadata={"home_team": "Home Team", "away_team": "Away Team"},
    )

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.6, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)

    assert "Home Team -1.5" in message
    assert "AH" not in message  # the raw "AH" suffix is dropped; "Asian Handicap" appears only as the market label
    assert "<i>Asian Handicap</i>" in message


def test_format_ev_board_message_builds_player_prop_selection_from_structured_fields() -> None:
    leg = _leg(
        market_type="Player Shots",
        selection="Kylian Mbappe player_shots 1.5",
        entity_name="Kylian Mbappe",
        entity_id="pl_1",
        side="over",
        line=1.5,
        team=None,
        metadata={"home_team": "France", "away_team": "Argentina"},
    )

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.6, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)

    assert "France vs Argentina" in message
    assert "Kylian Mbappe Over 1.5" in message
    assert "<i>Player Shots</i>" in message


def test_format_ev_board_message_shows_american_odds_and_all_template_fields() -> None:
    leg = _leg(odds=2.5, metadata={"home_team": "Home Team", "away_team": "Away Team"})

    def provider(_leg: MarketLeg) -> ProbabilityEstimate:
        return ProbabilityEstimate(true_probability=0.6, sample_size=200, volatility=0.1)

    result = rank_ev_board([leg], provider, positive_only=False)[0]
    message = format_ev_board_message([result], title="Board", min_confidence_score=0)

    # decimal 2.5 -> American +150 (see ev_engine_core.decimal_to_american)
    assert "+150" in message
    assert "Implied" in message
    assert "True Prob:" in message
    assert "Edge (EV):" in message
    assert "AI Score:" in message
    assert f"{result.confidence_score}/100" in message
