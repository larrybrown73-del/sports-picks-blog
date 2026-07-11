from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backtest import (
    backtest_summary,
    compute_ev_columns,
    devig_group,
    implied_probability_column,
    legs_to_frame,
)
from ev_engine_core import MarketLeg, evaluate_leg, expected_value, odds_to_decimal


def _sample_legs() -> list[MarketLeg]:
    return [
        MarketLeg(
            game_id="mt_10", market_type="match_odds", selection="Spain", odds=1.90,
            odds_format="decimal", sportsbook="Bet365", team="Spain", side="home", sample_size=40,
        ),
        MarketLeg(
            game_id="mt_10", market_type="match_odds", selection="Draw", odds=3.60,
            odds_format="decimal", sportsbook="Bet365", team=None, side="draw", sample_size=40,
        ),
        MarketLeg(
            game_id="mt_10", market_type="match_odds", selection="Portugal", odds=4.20,
            odds_format="decimal", sportsbook="Bet365", team="Portugal", side="away", sample_size=40,
        ),
        # American-quoted leg mixed into the same dataset on purpose, to
        # prove the vectorized odds conversion handles mixed formats.
        MarketLeg(
            game_id="mt_11", market_type="player_shots", selection="Lamine Yamal 2+ Shots",
            odds=-135, odds_format="american", sportsbook="DraftKings", entity_name="Lamine Yamal",
            line=1.5, sample_size=20,
        ),
        # Deliberately missing odds -- must be dropped, never imputed.
        MarketLeg(
            game_id="mt_12", market_type="btts", selection="BTTS Yes", odds=float("nan"),
            odds_format="decimal", sportsbook="Bet365",
        ),
    ]


def test_legs_to_frame_drops_rows_with_missing_odds() -> None:
    frame = legs_to_frame(_sample_legs())
    assert len(frame) == 4  # the NaN-odds leg for mt_12 must be dropped
    assert "mt_12" not in frame["game_id"].to_numpy()
    assert frame["line"].dtype == np.float64
    assert frame["odds"].dtype == np.float64


def test_implied_probability_column_matches_scalar_engine() -> None:
    legs = _sample_legs()
    frame = legs_to_frame(legs)
    vectorized = implied_probability_column(frame)

    for idx, row in frame.iterrows():
        scalar_decimal = odds_to_decimal(row["odds"], row["odds_format"])
        expected = 1.0 / scalar_decimal
        assert math.isclose(vectorized.loc[idx], expected, rel_tol=1e-12)


def test_compute_ev_columns_matches_scalar_evaluate_leg() -> None:
    legs = [leg for leg in _sample_legs() if not math.isnan(leg.odds)]
    frame = legs_to_frame(legs)

    def provider(leg: MarketLeg):
        # Simple stand-in projection model: a fixed edge over market-implied
        # probability so both the scalar and vectorized paths get a
        # realistic, non-degenerate true probability to grade against.
        from ev_engine_core import ProbabilityEstimate, implied_probability

        base = implied_probability(leg.odds, leg.odds_format)
        return ProbabilityEstimate(true_probability=min(0.95, base + 0.05))

    true_probs = pd.Series(
        [provider(leg).true_probability for leg in legs],
        index=frame.index,
        name="true_probability",
    )

    graded = compute_ev_columns(frame, true_probs, stake=1.0)

    for i, leg in enumerate(legs):
        scalar_result = evaluate_leg(leg, provider, stake=1.0)
        row = graded.iloc[i]
        assert math.isclose(row["decimal_odds"], scalar_result.decimal_odds, rel_tol=1e-12)
        assert math.isclose(row["implied_probability"], scalar_result.implied_probability, rel_tol=1e-12)
        assert math.isclose(row["ev"], scalar_result.ev, rel_tol=1e-12)
        assert math.isclose(row["ev_per_unit"], scalar_result.ev_per_unit, rel_tol=1e-12)
        assert bool(row["positive_ev"]) == scalar_result.positive_ev


def test_compute_ev_columns_rejects_misaligned_index() -> None:
    frame = legs_to_frame([leg for leg in _sample_legs() if not math.isnan(leg.odds)])
    bad_index_probs = pd.Series([0.5] * len(frame), index=range(100, 100 + len(frame)))

    try:
        compute_ev_columns(frame, bad_index_probs)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for misaligned true_probability index")


def test_devig_group_sums_to_one_per_market() -> None:
    frame = legs_to_frame([leg for leg in _sample_legs() if leg.market_type == "match_odds"])
    frame = frame.assign(implied_probability=implied_probability_column(frame))

    fair = devig_group(frame, ["game_id", "market_type", "sportsbook"])
    assert math.isclose(fair.sum(), 1.0, rel_tol=1e-9)


def test_backtest_summary_reports_positive_ev_only() -> None:
    legs = [leg for leg in _sample_legs() if not math.isnan(leg.odds)]
    frame = legs_to_frame(legs)

    from ev_engine_core import ProbabilityEstimate, implied_probability

    def provider(leg: MarketLeg):
        base = implied_probability(leg.odds, leg.odds_format)
        return ProbabilityEstimate(true_probability=min(0.95, base + 0.10))

    true_probs = pd.Series(
        [provider(leg).true_probability for leg in legs], index=frame.index
    )
    graded = compute_ev_columns(frame, true_probs)
    summary = backtest_summary(graded)

    assert summary["total_legs_considered"] == len(graded)
    assert summary["positive_ev_legs"] > 0
    assert summary["avg_ev_per_unit_positive_ev"] > 0
