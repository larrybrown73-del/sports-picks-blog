"""
Lightweight Telegram push notifications for the EV engine.

Deliberately dependency-light: a single `requests.post` call against the
public Telegram Bot HTTP API (https://api.telegram.org/bot<token>/sendMessage),
no telegram SDK. `send_telegram_message` knows nothing about EV/soccer at
all -- it's a reusable plain-text/HTML sender. `format_ev_board_message`
is the soccer-EV-specific formatting layer on top of it.
"""

from __future__ import annotations

import os
from html import escape as _escape_html
from typing import Any, Sequence

from ev_engine_core import DEFAULT_ENV_FILE, EVResult, MarketLeg, decimal_to_american, load_env_file

TELEGRAM_API_BASE = "https://api.telegram.org"
TELEGRAM_BOT_TOKEN_ENV_VAR = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV_VAR = "TELEGRAM_CHAT_ID"

# Telegram hard-rejects a sendMessage text over 4096 UTF-16 code units; stay
# comfortably under that so multi-byte characters (e.g. emoji) can't push a
# message over the real limit.
MAX_MESSAGE_LENGTH = 3500

DEFAULT_MIN_CONFIDENCE_SCORE = 90


class TelegramConfigError(RuntimeError):
    """Raised when the bot token / chat id can't be resolved -- never silently no-ops instead."""


def _require_requests() -> Any:
    try:
        import requests  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without requests installed
        raise ImportError(
            "The 'requests' package is required for Telegram notifications. Install it with: pip install requests"
        ) from exc
    return requests


def get_telegram_credentials(*, env_file: str | os.PathLike[str] | None = DEFAULT_ENV_FILE) -> tuple[str, str]:
    """
    Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment, falling
    back to loading them from `.env.local` at the repo root (see
    `ev_engine_core.DEFAULT_ENV_FILE` -- resolved from this package's own
    location, not the process's current working directory, so this works
    the same whether it's run interactively from the repo root or launched
    from anywhere else, e.g. a Task Scheduler entry).
    """

    if env_file is not None and not (os.getenv(TELEGRAM_BOT_TOKEN_ENV_VAR) and os.getenv(TELEGRAM_CHAT_ID_ENV_VAR)):
        load_env_file(env_file)

    bot_token = os.getenv(TELEGRAM_BOT_TOKEN_ENV_VAR)
    chat_id = os.getenv(TELEGRAM_CHAT_ID_ENV_VAR)
    if not bot_token or not chat_id:
        raise TelegramConfigError(
            f"Missing {TELEGRAM_BOT_TOKEN_ENV_VAR}/{TELEGRAM_CHAT_ID_ENV_VAR}. "
            "Set them in .env.local or the real environment before sending Telegram alerts."
        )
    return bot_token, chat_id


def send_telegram_message(
    text: str,
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "HTML",
    disable_web_page_preview: bool = True,
    session: Any | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """
    POST one message to the Telegram Bot API's sendMessage endpoint.

    Raises (rather than silently swallowing) both HTTP-level failures and
    Telegram's own `"ok": false` application-level failures -- a bet ticket
    that quietly never arrives is worse than a loud crash the scheduler's
    caller can log and act on.

    Messages longer than MAX_MESSAGE_LENGTH are split on line boundaries and
    sent as multiple messages, in order; the return value is the response
    for the LAST chunk sent.
    """

    if bot_token is None or chat_id is None:
        resolved_token, resolved_chat_id = get_telegram_credentials()
        bot_token = bot_token or resolved_token
        chat_id = chat_id or resolved_chat_id

    requests_module = _require_requests()
    http_client = session or requests_module
    url = f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage"

    last_response: dict[str, Any] = {}
    for chunk in _chunk_message(text):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        response = http_client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API rejected the message: {data}")
        last_response = data

    return last_response


def _chunk_message(text: str, *, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_length = 0
    for line in text.split("\n"):
        # +1 accounts for the newline that will separate this line from the next.
        added_length = len(line) + 1
        if current_length + added_length > max_length and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_length = 0
        current_lines.append(line)
        current_length += added_length
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


# Internal market_type codes (assigned by ev_engine_core's team-odds
# flattener) -> a human-readable label for display. Player-prop legs don't
# need an entry here: their market_type is TheStatsAPI's own market "name"
# field (e.g. "Player Shots"), already display-ready as-is.
_MARKET_TYPE_LABELS = {
    "match_odds": "Match Result",
    "btts": "Both Teams to Score",
    "total_goals": "Total Goals",
    "match_corners": "Total Corners",
    "asian_handicap": "Asian Handicap",
}


def _market_label(market_type: str | None) -> str:
    if not market_type:
        return "Market"
    if market_type in _MARKET_TYPE_LABELS:
        return _MARKET_TYPE_LABELS[market_type]
    # Fallback for anything unmapped (a future market type, a test fixture,
    # ...) -- humanize the raw token instead of leaking a snake_case string
    # like "corner_kicks" straight into a user-facing message.
    return market_type.replace("_", " ").strip().title()


def _match_context_line(leg: MarketLeg) -> str:
    """
    The \u26bd line: WHICH game this bet belongs to. Read from
    leg.metadata's home_team/away_team (present on every leg the ingestion
    layer builds -- see flatten_thestatsapi_match_odds/player_odds) rather
    than leg.team/leg.entity_name, which identify the SIDE of the bet, not
    the match -- using them here is what caused the old message to show a
    team/player's name twice.
    """

    home = leg.metadata.get("home_team")
    away = leg.metadata.get("away_team")
    if home and away:
        return f"{home} vs {away}"
    return leg.team or leg.entity_name or "Match"


def _format_line_value(value: float, *, signed: bool = False) -> str:
    return f"{value:+g}" if signed else f"{value:g}"


def _bet_selection_line(leg: MarketLeg) -> str:
    """
    The \U0001F525 line: the actual pick. Deliberately rebuilt from
    structured leg fields (side/line/team/entity_name) rather than reused
    from leg.selection -- leg.selection exists for identity/grouping
    (rank_ev_board keys legs off it), not display, and for match_odds /
    asian_handicap it's LITERALLY just the team name already shown on the
    match-context line above it (e.g. selection="Switzerland" when
    leg.team=="Switzerland"), which is exactly the repeated-name clutter
    this format is meant to remove.
    """

    market = leg.market_type

    if market == "match_odds":
        if leg.side == "draw":
            return "Draw"
        return f"{leg.team or leg.selection} to Win"

    if market == "btts":
        return (leg.side or leg.selection).title()

    if market in {"total_goals", "match_corners"}:
        side_label = (leg.side or "").title()
        line_label = _format_line_value(leg.line) if leg.line is not None else ""
        return f"{side_label} {line_label}".strip() or leg.selection

    if market == "asian_handicap":
        line_label = _format_line_value(leg.line, signed=True) if leg.line is not None else ""
        return f"{leg.team or ''} {line_label}".strip() or leg.selection

    if leg.entity_name:
        # Player prop: leg.side is the prop's over/under direction, leg.line
        # is the number -- e.g. entity_name="Kylian Mbappe", side="over",
        # line=1.5 -> "Kylian Mbappe Over 1.5".
        side_label = (leg.side or "").title()
        line_label = _format_line_value(leg.line) if leg.line is not None else ""
        detail = f"{side_label} {line_label}".strip()
        return f"{leg.entity_name} {detail}".strip() if detail else leg.selection

    return leg.selection


def format_ev_board_message(
    results: Sequence[EVResult],
    *,
    title: str,
    min_confidence_score: int = DEFAULT_MIN_CONFIDENCE_SCORE,
) -> str:
    """
    Renders EVResult rows with `confidence_score > min_confidence_score` as
    an HTML-formatted, mobile-scannable message, highest AI Score first:

        \u26bd <b>[Match or Team Name]</b>
        \U0001F525 <b>[Bet Selection]</b> (<i>[Market Name]</i>)
        \U0001F4B0 <b>Odds:</b> [American Odds] ([Implied Prob]% Implied)
        \U0001F4C8 <b>True Prob:</b> [True Prob]%
        \u26a1 <b>Edge (EV):</b> +[EV]%
        \U0001F3AF <b>AI Score:</b> [Score]/100

    The match-context and bet-selection lines are built from structured leg
    fields, not string concatenation of leg.selection -- see
    `_match_context_line`/`_bet_selection_line` for why that distinction
    matters (it's what stops a team/player's name from being printed twice).
    """

    top_bets = sorted(
        (r for r in results if r.confidence_score > min_confidence_score),
        key=lambda r: r.confidence_score,
        reverse=True,
    )

    lines = [f"<b>{_escape_html(title)}</b>"]
    if not top_bets:
        lines.append(f"No bets scored above {min_confidence_score} today.")
        return "\n".join(lines)

    for result in top_bets:
        leg = result.leg
        american_odds = decimal_to_american(result.decimal_odds)
        lines.append("")
        lines.append(f"\u26bd <b>{_escape_html(_match_context_line(leg))}</b>")
        lines.append(
            f"\U0001F525 <b>{_escape_html(_bet_selection_line(leg))}</b> "
            f"(<i>{_escape_html(_market_label(leg.market_type))}</i>)"
        )
        lines.append(
            f"\U0001F4B0 <b>Odds:</b> {american_odds:+d} ({result.implied_probability * 100:.1f}% Implied)"
        )
        lines.append(f"\U0001F4C8 <b>True Prob:</b> {result.true_probability * 100:.1f}%")
        lines.append(f"\u26a1 <b>Edge (EV):</b> {result.ev_per_unit * 100:+.1f}%")
        lines.append(f"\U0001F3AF <b>AI Score:</b> {result.confidence_score}/100")

    return "\n".join(lines)
