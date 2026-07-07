from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from baseball_props.data.injuries import (
    apply_injury_rust_multiplier,
    fetch_active_injuries,
    injury_rust_multiplier,
    lookup_injury,
    normalize_injury_name,
    parse_injury_html,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fantasypros_injuries_sample.html"
TODAY = date(2025, 6, 27)


def test_parse_injury_html_fixture() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    result = parse_injury_html(html, today=TODAY)

    assert "Jordan Lawlar" in result
    assert result["Jordan Lawlar"]["status"] == "IL10"
    assert result["Jordan Lawlar"]["injury"] == "Hamstring"
    assert result["Jordan Lawlar"]["days_off_il"] == 0

    assert result["Nick Kurtz"]["status"] == "Active"
    assert result["Nick Kurtz"]["injury"] == "Hip Flexor Strain"
    assert result["Nick Kurtz"]["days_off_il"] == 3

    assert result["Jacob Wilson"]["status"] == "DTD"
    assert result["Jacob Wilson"]["days_off_il"] == 0


def test_fetch_active_injuries_network_failure_returns_empty() -> None:
    with patch(
        "baseball_props.data.injuries.requests.get",
        side_effect=requests.RequestException("offline"),
    ):
        assert fetch_active_injuries() == {}


def test_fetch_active_injuries_parses_response() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    mock_response = type("R", (), {"text": html, "raise_for_status": lambda self: None})()
    with patch(
        "baseball_props.data.injuries.requests.get",
        return_value=mock_response,
    ):
        result = fetch_active_injuries(today=TODAY)
    assert len(result) == 3


def test_injury_rust_multiplier_il_is_zero() -> None:
    record = {"status": "IL10", "days_off_il": 0}
    assert injury_rust_multiplier(record) == 0.0
    assert apply_injury_rust_multiplier(1.95, record) == 0.0


def test_injury_rust_multiplier_returning_active() -> None:
    record = {"status": "Active", "days_off_il": 3}
    expected = 0.75 + 0.25 * 3 / 14
    assert injury_rust_multiplier(record) == pytest.approx(expected)
    assert apply_injury_rust_multiplier(2.0, record) == pytest.approx(2.0 * expected)


def test_injury_rust_multiplier_dtd_zero_days_unchanged() -> None:
    record = {"status": "DTD", "days_off_il": 0}
    assert injury_rust_multiplier(record) == 1.0


def test_lookup_injury_normalized_match() -> None:
    html = FIXTURE.read_text(encoding="utf-8")
    lookup = parse_injury_html(html, today=TODAY)
    assert lookup_injury("Nick Kurtz", lookup) is not None
    assert lookup_injury("nick kurtz", lookup) is not None
    assert lookup_injury("Nick Kurtz Jr.", lookup) is not None
    assert lookup_injury("Unknown Player", lookup) is None


def test_normalize_injury_name_strips_suffix() -> None:
    assert normalize_injury_name("Nick Kurtz Jr.") == normalize_injury_name("Nick Kurtz")
