"""Run today's model projections against live MLB odds (The Odds API)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import requests

import baseball_data
from backtest import (
    DEFAULT_PREDICTION_SEASONS,
    EDGE_THRESHOLD,
    _memoize_obp_lookups,
    load_frozen_models,
)
from config import DEFAULT_ROLLING_WINDOW, MAX_ODDS_CAP, MIN_PROBABILITY_FLOOR
from momentum import apply_team_streak_bonus
from market.calculations import compute_wager_metrics
from slate_evaluation import append_slate_evaluation_log, evaluate_game
from starter_baseline import fetch_starter_eras, pitching_mismatch_veto

THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
SPORTSBOOK_KEYS = "draftkings,fanduel,betmgm"


def _require_odds_api_key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing THE_ODDS_API_KEY. Set it in baseball-predictor/.env (see baseball-props-model/.env.example)."
        )
    return key


@dataclass
class SlatePick:
    away_name: str
    home_name: str
    play: str
    book: str
    american_odds: int
    model_prob: float
    market_prob: float
    edge_pct: float
    ev_pct: float
    ev_per_unit: float | None
    confidence_tier: str | None
    pred_home_runs: float
    pred_away_runs: float
    quarter_kelly_pct: float
    temperature: str
    wind: str
    bullpen_status: str


def _normalize_team_name(team_name: str) -> str:
    replacements = {
        "d-backs": "diamondbacks",
        "diamond backs": "diamondbacks",
        "oakland athletics": "athletics",
    }
    normalized = (
        team_name.lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )
    return replacements.get(normalized, normalized)


def _teams_match(api_home: str, api_away: str, home_name: str, away_name: str) -> bool:
    return (
        _normalize_team_name(api_home) == _normalize_team_name(home_name)
        and _normalize_team_name(api_away) == _normalize_team_name(away_name)
    )


def fetch_all_mlb_odds() -> list[dict]:
    params = {
        "apiKey": _require_odds_api_key(),
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": SPORTSBOOK_KEYS,
    }
    response = requests.get(THE_ODDS_API_URL, params=params, timeout=30)
    response.raise_for_status()
    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"The Odds API requests remaining: {remaining}", flush=True)
    return response.json()


def _find_odds_event(odds_games: list[dict], home_name: str, away_name: str) -> dict | None:
    for event in odds_games:
        if _teams_match(
            event.get("home_team", ""),
            event.get("away_team", ""),
            home_name,
            away_name,
        ):
            return event
    return None


def _best_h2h_prices(event: dict) -> dict[str, tuple[int, str]]:
    best: dict[str, tuple[int, str]] = {}
    for bookmaker in event.get("bookmakers", []):
        book_title = bookmaker.get("title", "Unknown")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue
                price = int(price)
                key = _normalize_team_name(name)
                if key not in best or price > best[key][0]:
                    best[key] = (price, book_title)
    return best


def _ev_pct(model_prob: float, american_odds: int, stake: float = 100.0) -> float:
    if american_odds > 0:
        payout = stake * american_odds / 100
    else:
        payout = stake * 100 / abs(american_odds)
    ev = (model_prob * payout) - ((1 - model_prob) * stake)
    return ev / stake * 100


def _passes_moneyline_guardrails(model_prob: float, american_odds: int) -> bool:
    if model_prob < MIN_PROBABILITY_FLOOR:
        return False
    if american_odds > MAX_ODDS_CAP:
        return False
    return True


def evaluate_slate(
    game_date: date | None = None,
    *,
    write_log: bool = False,
) -> list[SlatePick]:
    game_date = game_date or date.today()
    slate = baseball_data.fetch_games_for_date(game_date)
    if not slate:
        return []

    _memoize_obp_lookups()
    models = load_frozen_models()
    seasons = list(DEFAULT_PREDICTION_SEASONS)
    if game_date.year not in seasons:
        seasons = sorted(set(seasons + [game_date.year]))
    history = baseball_data.games_for_prediction(seasons)
    if history.empty:
        raise RuntimeError("No historical games available for predictions.")

    odds_games = fetch_all_mlb_odds()
    picks: list[SlatePick] = []
    log_rows: list[dict[str, str]] = []

    for game in slate:
        home_name = game["home_name"]
        away_name = game["away_name"]
        try:
            evaluated = evaluate_game(
                game, game_date, history, models, window=DEFAULT_ROLLING_WINDOW
            )
        except Exception as exc:
            print(f"  Skip {away_name} @ {home_name}: {exc}", flush=True)
            continue

        event = _find_odds_event(odds_games, home_name, away_name)
        if event is None:
            print(f"  No odds board match: {away_name} @ {home_name}", flush=True)
            continue

        best_prices = _best_h2h_prices(event)
        home_key = _normalize_team_name(home_name)
        away_key = _normalize_team_name(away_name)

        home_prob = apply_team_streak_bonus(
            evaluated.home_prob, int(game["home_id"]), game_date
        )
        away_prob = apply_team_streak_bonus(
            evaluated.away_prob, int(game["away_id"]), game_date
        )

        home_sp_era, away_sp_era = fetch_starter_eras(
            int(game["game_id"]), season=game_date.year
        )

        candidates: list[SlatePick] = []
        for team_name, key, prob, backing_home in (
            (home_name, home_key, home_prob, True),
            (away_name, away_key, away_prob, False),
        ):
            if key not in best_prices:
                continue
            our_era = home_sp_era if backing_home else away_sp_era
            opp_era = away_sp_era if backing_home else home_sp_era
            if pitching_mismatch_veto(our_sp_era=our_era, opponent_sp_era=opp_era):
                print(
                    f"  Pitching mismatch veto: {team_name} "
                    f"(our ERA {our_era}, opp ERA {opp_era})",
                    flush=True,
                )
                continue
            american_odds, book = best_prices[key]
            if not _passes_moneyline_guardrails(prob, american_odds):
                continue
            metrics = compute_wager_metrics(prob, american_odds)
            market_prob = metrics.implied_prob or 0.0
            edge = (metrics.edge_pct or 0.0) / 100.0
            ev_per_unit = metrics.ev_per_unit or 0.0
            qk = metrics.fractional_kelly_pct or 0.0
            candidates.append(
                SlatePick(
                    away_name=away_name,
                    home_name=home_name,
                    play=team_name,
                    book=book,
                    american_odds=american_odds,
                    model_prob=prob,
                    market_prob=market_prob,
                    edge_pct=edge * 100,
                    ev_pct=ev_per_unit * 100.0,
                    ev_per_unit=ev_per_unit,
                    confidence_tier=metrics.confidence_tier,
                    pred_home_runs=evaluated.home_runs,
                    pred_away_runs=evaluated.away_runs,
                    quarter_kelly_pct=qk,
                    temperature=evaluated.temperature,
                    wind=evaluated.wind,
                    bullpen_status=evaluated.bullpen_status,
                )
            )

        if not candidates:
            continue

        best = max(candidates, key=lambda row: row.edge_pct)
        if (
            best.edge_pct > EDGE_THRESHOLD * 100
            and best.quarter_kelly_pct > 0
            and _passes_moneyline_guardrails(best.model_prob, best.american_odds)
        ):
            picks.append(best)
            log_rows.append(
                {
                    "Date": game_date.isoformat(),
                    "Matchup": f"{best.away_name} @ {best.home_name}",
                    "Temp": best.temperature,
                    "Wind": best.wind,
                    "Bullpen_Status": best.bullpen_status,
                    "Edge_Pct": f"{best.edge_pct:+.2f}",
                    "Play": best.play,
                    "Result": "",
                }
            )

    picks.sort(key=lambda row: row.edge_pct, reverse=True)
    if write_log:
        append_slate_evaluation_log(log_rows)
    return picks


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    today = date.today()
    write_log = "--log" in sys.argv
    print(f"--- Live Odds Slate ({today.isoformat()}) ---", flush=True)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print(f"Books: {SPORTSBOOK_KEYS.replace(',', ', ')}\n", flush=True)

    try:
        picks = evaluate_slate(today, write_log=write_log)
    except requests.RequestException as exc:
        raise SystemExit(f"Odds API request failed: {exc}") from exc

    if not picks:
        print("No moneyline plays with model edge above 3% vs best available live odds.")
        return

    print(
        f"{'Matchup':<38} {'Play':<20} {'Book':<10} {'Line':>6} {'Model%':>7} "
        f"{'Edge':>7} {'Temp':>6} {'Wind':<16} {'Bullpen':<24}"
    )
    print("-" * 140)
    for pick in picks:
        matchup = f"{pick.away_name} @ {pick.home_name}"
        wind = pick.wind[:15]
        bullpen = pick.bullpen_status[:23]
        print(
            f"{matchup:<38} {pick.play:<20} {pick.book:<10} {pick.american_odds:>+6d} "
            f"{pick.model_prob * 100:>6.1f}% {pick.edge_pct:>+6.2f}% {pick.temperature:>6} "
            f"{wind:<16} {bullpen:<24}"
        )
        print(
            f"    Pred runs: {pick.home_name} {pick.pred_home_runs:.2f} - "
            f"{pick.away_name} {pick.pred_away_runs:.2f}"
        )


if __name__ == "__main__":
    main()
