from backtest import _select_best_side_bet, compute_market_edge


def test_select_best_side_picks_higher_edge_when_both_playable() -> None:
    """Only one side should be selected even if both exceed the edge threshold."""
    _, home_edge, _ = compute_market_edge(0.62, -110)
    _, away_edge, _ = compute_market_edge(0.58, +145)

    assert home_edge > 0.03
    assert away_edge > 0.03

    selected = _select_best_side_bet(
        0.62,
        -110,
        True,
        0.58,
        +145,
        False,
    )
    assert selected is not None
    odds, _ = selected
    assert odds == +145 if away_edge > home_edge else -110


def test_select_best_side_returns_none_below_threshold() -> None:
    selected = _select_best_side_bet(
        0.51,
        -110,
        True,
        0.49,
        -110,
        False,
        edge_threshold=0.03,
    )
    assert selected is None
