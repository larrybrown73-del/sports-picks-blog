from __future__ import annotations

import pandas as pd

from baseball_props.analysis.edge_sheets import (
    aggregate_top_conviction,
    build_batter_tb_edge_sheet,
    build_pitcher_outs_edge_sheet,
)
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

def compute_conviction_plays(
    projected: pd.DataFrame,
    pitcher_outs: pd.DataFrame,
    prop_lines: pd.DataFrame,
    *,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Rank model vs market edges for batter total bases and pitcher outs by Edge %.
    """
    if prop_lines.empty:
        logger.warning("No prop lines available for conviction ranking")
        return pd.DataFrame(
            columns=[
                "player_name",
                "market",
                "model_value",
                "market_line",
                "probability_pct",
                "edge_pct",
                "recommendation",
            ]
        )

    batter_sheet = build_batter_tb_edge_sheet(projected, prop_lines)
    pitcher_sheet = build_pitcher_outs_edge_sheet(pitcher_outs, prop_lines)
    return aggregate_top_conviction(batter_sheet, pitcher_sheet, top_n=top_n)
