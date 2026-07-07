from baseball_props.analysis.edge_sheet_health import (
    EdgeSheetHealthReport,
    record_edge_skip,
    safe_edge_eval,
)
from baseball_props.data.data_health import DataHealthReport


def test_record_edge_skip_increments_counts() -> None:
    report = EdgeSheetHealthReport()
    record_edge_skip(report, "batter_no_market_quote", player="Player A")
    record_edge_skip(report, "batter_no_market_quote", player="Player B")
    record_edge_skip(report, "batter_name_unmatched", player="Player C")

    assert report.skip_counts["batter_no_market_quote"] == 2
    assert report.skip_counts["batter_name_unmatched"] == 1


def test_edge_health_merge_into_data_health_dict() -> None:
    base = DataHealthReport()
    base.record_missing("ingest", detail="test warning")
    edge = EdgeSheetHealthReport(report=base)
    record_edge_skip(edge, "pitcher_no_quote", player="Pitcher X")

    payload = edge.to_dict()
    assert "edge_skip_counts" in payload
    assert payload["edge_skip_counts"]["pitcher_no_quote"] == 1
    assert payload["warning_count"] >= 1


def test_safe_edge_eval_returns_default_on_error() -> None:
    report = EdgeSheetHealthReport()

    def boom() -> int:
        raise ValueError("fail")

    result = safe_edge_eval("test_label", boom, default=0, report=report)
    assert result == 0
    assert report.skip_counts["test_label"] == 1


def test_edge_health_merge_reports() -> None:
    left = EdgeSheetHealthReport()
    right = EdgeSheetHealthReport()
    record_edge_skip(left, "a", player="p1")
    record_edge_skip(right, "a", player="p2")
    record_edge_skip(right, "b", player="p3")

    left.merge(right)
    assert left.skip_counts["a"] == 2
    assert left.skip_counts["b"] == 1
