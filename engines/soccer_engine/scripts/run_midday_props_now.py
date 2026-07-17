"""Midday full board with the expanded AI Score >= 70 alert threshold."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "midday_props_manual.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

from ev_engine_core import decimal_to_american  # noqa: E402
from scheduler import MIN_CONFIDENCE_SCORE_FOR_ALERT, SoccerEvScheduler  # noqa: E402
from telegram_notifier import (  # noqa: E402
    _bet_selection_line,
    _market_label,
    _match_context_line,
    format_ev_board_message,
    send_telegram_message,
)

PLAYER_MARKET_HINTS = (
    "player",
    "goalscorer",
    "shots",
    "assists",
    "cards",
    "tackles",
    "saves",
    "passes",
)


def _is_player_prop(result) -> bool:
    leg = result.leg
    if leg.entity_id or leg.entity_name:
        return True
    market = (leg.market_type or "").lower()
    return any(token in market for token in PLAYER_MARKET_HINTS)


def _print_prop(r) -> None:
    leg = r.leg
    american = decimal_to_american(r.decimal_odds)
    downside = (r.downside_analysis or {}).get("main_dependency_risk") or ""
    print()
    print(f"TIER:  {r.tier}  royal={r.royal_approved}")
    print(f"MATCH: {_match_context_line(leg)}")
    print(f"PICK:  {_bet_selection_line(leg)}  ({_market_label(leg.market_type)})")
    print(f"ODDS:  {american:+d}  ({r.implied_probability * 100:.1f}% Implied)")
    print(f"TRUE:  {r.true_probability * 100:.1f}%")
    print(f"EV:    {r.ev_per_unit * 100:+.1f}%")
    print(f"AI:    {r.confidence_score}/100")
    print(
        f"GATES: form={r.form_validated} hit={r.hit_rate_ok} xi={r.lineup_confirmed} "
        f"corr={r.contradiction_clear} mkt_div={r.market_divergence}"
    )
    if downside:
        print(f"WHY?:  {downside}")


def main() -> None:
    sched = SoccerEvScheduler()
    print(f"include_player_props={sched._include_player_props}")
    print(f"min_ai_score_threshold={MIN_CONFIDENCE_SCORE_FOR_ALERT} (>=)")
    if not sched._include_player_props:
        raise SystemExit("SOCCER_ENGINE_INCLUDE_PLAYER_PROPS is off -- aborting.")

    today = datetime.now(timezone.utc).date()
    matches = sched._fetch_and_ensure_models(today)
    print(f"matches today: {len(matches)}")

    full_results = []
    for match in matches:
        home = (match.get("home_team") or {}).get("name")
        away = (match.get("away_team") or {}).get("name")
        model = sched._models_by_competition.get(match.get("competition_id"))
        if model is None:
            print(f"  skip {match.get('id')}: {home} vs {away} (no model)")
            continue
        board = sched._grade_board(match, model, include_player_props=True)
        print(f"  graded {home} vs {away}: {len(board)} legs")
        full_results.extend(board)

    sched._mark_early_props_sync_completed_today()

    message = format_ev_board_message(
        full_results,
        title="\U0001F451 Royal Picks Board (AI >= 70)",
        min_confidence_score=MIN_CONFIDENCE_SCORE_FOR_ALERT,
    )
    send_telegram_message(message)
    print("Telegram message sent.")

    qualifying = sorted(
        (r for r in full_results if r.confidence_score >= MIN_CONFIDENCE_SCORE_FOR_ALERT),
        key=lambda r: r.confidence_score,
        reverse=True,
    )
    player_qualifying = [r for r in qualifying if _is_player_prop(r)]
    print()
    print("=" * 72)
    print(
        f"ALL PLAYS AI >= {MIN_CONFIDENCE_SCORE_FOR_ALERT}: {len(qualifying)} "
        f"({len(player_qualifying)} player props)"
    )
    print("=" * 72)
    for r in qualifying:
        _print_prop(r)

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
