import math

import pandas as pd

from baseball_props.analysis.market_metrics import enrich_edge_sheet_with_market_metrics
from baseball_props.analysis.edge_sheets import PASS_NO_DATA
from baseball_props.analysis.market_metrics import (
    american_to_decimal,
    compute_ev,
    confidence_tier,
    fractional_kelly,
)
from baseball_props.config import KELLY_MAX_STAKE_PCT


def test_american_to_decimal_minus_110() -> None:
    assert math.isclose(american_to_decimal(-110), 1.909090909, rel_tol=1e-4)


def test_compute_ev_at_minus_110_fifty_five_pct() -> None:
    decimal = american_to_decimal(-110)
    assert decimal is not None
    ev = compute_ev(0.55, decimal)
    assert ev is not None
    assert ev > 0


def test_confidence_tier_boundaries() -> None:
    assert confidence_tier(5.0) == "Tier-1 High Conviction"
    assert confidence_tier(4.9) == "Tier-2 Moderate"
    assert confidence_tier(3.0) == "Tier-2 Moderate"
    assert confidence_tier(2.9) == "Tier-3 Speculative"
    assert confidence_tier(0.5) == "Below Threshold"
    assert confidence_tier(None) is None


def test_fractional_kelly_capped_at_max_stake() -> None:
    decimal = american_to_decimal(-110)
    assert decimal is not None
    kelly = fractional_kelly(0.75, decimal, fraction=1.0, max_stake_pct=KELLY_MAX_STAKE_PCT)
    assert kelly is not None
    assert kelly <= KELLY_MAX_STAKE_PCT


def test_enrich_edge_sheet_no_nan_on_no_data_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "probability_pct": None,
                "edge_pct": None,
                "recommendation": PASS_NO_DATA,
            }
        ]
    )
    enriched = enrich_edge_sheet_with_market_metrics(df)
    assert enriched.iloc[0]["ev_per_unit"] is None
    assert enriched.iloc[0]["confidence_tier"] is None
    assert not enriched["ev_per_unit"].apply(lambda x: isinstance(x, float) and math.isnan(x)).any()
