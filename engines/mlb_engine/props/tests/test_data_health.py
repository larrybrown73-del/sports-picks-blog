from __future__ import annotations

import pandas as pd
import pytest

from baseball_props.data.data_health import DataHealthReport, safe_feature_slice


def test_safe_feature_slice_returns_default_on_exception() -> None:
    report = DataHealthReport()

    def _fail() -> dict[str, float]:
        raise RuntimeError("network down")

    value = safe_feature_slice("test_feature", _fail, default={"woba": 0.31}, report=report)
    assert value == {"woba": 0.31}
    assert report.fallback_counts.get("test_feature") == 1
    assert any("Missing-Data Warning" in w for w in report.warnings)


def test_safe_feature_slice_returns_default_on_empty() -> None:
    report = DataHealthReport()
    value = safe_feature_slice(
        "empty_profile",
        lambda: pd.DataFrame(),
        default=pd.DataFrame([{"player_id": "1"}]),
        report=report,
    )
    assert len(value) == 1
    assert report.fallback_counts.get("empty_profile") == 1


def test_data_health_report_merge() -> None:
    left = DataHealthReport()
    right = DataHealthReport()
    right.record_missing("statcast")
    left.merge(right)
    assert "statcast" in left.missing_slices
