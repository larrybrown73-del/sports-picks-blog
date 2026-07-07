from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

from baseball_props.config import INJURY_RUST_DAYS_TO_FULL, INJURY_RUST_MIN_MULTIPLIER
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_INJURY_URL = "https://www.fantasypros.com/mlb/injuries/"
_REQUEST_TIMEOUT = 20.0
_USER_AGENT = (
    "Mozilla/5.0 (compatible; baseball-props-model/1.0; +https://github.com/local/mlb-props)"
)

_RETURNING_STATUSES = frozenset({"active", "dtd"})


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def _parse_updated_days(updated_text: str, *, today: date | None = None) -> int:
    """Parse 'Jun 27' style dates; return days since update."""
    ref = today or date.today()
    text = updated_text.strip()
    if not text:
        return 0
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return max((ref - parsed.date()).days, 0)
        except ValueError:
            continue
    for fmt in ("%b %d", "%B %d"):
        try:
            parsed = datetime.strptime(f"{text} {ref.year}", f"{fmt} %Y")
            if parsed.date() > ref:
                parsed = parsed.replace(year=ref.year - 1)
            return max((ref - parsed.date()).days, 0)
        except ValueError:
            continue
    return 0


def _days_off_il(status: str, updated_text: str, *, today: date | None = None) -> int:
    normalized = status.strip().lower()
    if normalized in _RETURNING_STATUSES:
        return _parse_updated_days(updated_text, today=today)
    return 0


def parse_injury_html(html: str, *, today: date | None = None) -> dict[str, dict[str, Any]]:
    """Parse FantasyPros injury board HTML into a player-keyed dict."""
    soup = BeautifulSoup(html, "html.parser")
    injuries: dict[str, dict[str, Any]] = {}

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) != 5:
                continue
            player = _normalize_name(cells[0].get_text(strip=True))
            position = cells[1].get_text(strip=True)
            status = cells[2].get_text(strip=True)
            injury = cells[3].get_text(strip=True)
            updated = cells[4].get_text(strip=True)
            if not player or not status:
                continue
            injuries[player] = {
                "status": status,
                "injury": injury,
                "days_off_il": _days_off_il(status, updated, today=today),
                "position": position,
                "updated": updated,
            }

    return injuries


def fetch_active_injuries(
    url: str = DEFAULT_INJURY_URL,
    *,
    today: date | None = None,
    data_health: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Scrape active MLB injuries from a public injury board.

    Returns a dict keyed by player name. On network or parse failure, returns {}.
    """
    from baseball_props.data.data_health import DataHealthReport, safe_feature_slice

    report = data_health if data_health is not None else DataHealthReport()

    def _fetch() -> dict[str, dict[str, Any]]:
        response = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        injuries = parse_injury_html(response.text, today=today)
        logger.info("Fetched %d injury records from %s", len(injuries), url)
        return injuries

    return safe_feature_slice(
        "fantasypros_injuries",
        _fetch,
        default={},
        report=report,
        empty_check=lambda value: not value,
    )


def normalize_injury_name(name: str) -> str:
    """Normalize player names for cross-source injury matching."""
    n = name.lower().strip()
    n = re.sub(r"[^a-z\s]", "", n)
    n = re.sub(r"\s(jr|sr|ii|iii|iv)$", "", n).strip()
    return n


def is_il_status(status: str) -> bool:
    """True when the player is on an injured list (IL10, IL15, IL60, etc.)."""
    return status.strip().lower().startswith("il")


def injury_rust_multiplier(record: dict[str, Any] | None) -> float:
    """
    Return a TB multiplier in [0.0, 1.0] based on injury status.

    IL → 0.0; recently activated (days_off_il > 0) → linear rust ramp; else 1.0.
    """
    if not record:
        return 1.0
    status = str(record.get("status", ""))
    if is_il_status(status):
        return 0.0
    days_off = int(record.get("days_off_il") or 0)
    if days_off <= 0:
        return 1.0
    rust_span = max(INJURY_RUST_DAYS_TO_FULL, 1.0)
    ramp = INJURY_RUST_MIN_MULTIPLIER + (
        (1.0 - INJURY_RUST_MIN_MULTIPLIER) * days_off / rust_span
    )
    return min(1.0, ramp)


def apply_injury_rust_multiplier(proj_tb: float, record: dict[str, Any] | None) -> float:
    """Scale projected total bases by the injury rust multiplier."""
    return proj_tb * injury_rust_multiplier(record)


def lookup_injury(
    player_name: str,
    injury_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find an injury record by normalized player name."""
    if not injury_lookup or not player_name:
        return None
    normalized_index = {
        normalize_injury_name(name): record for name, record in injury_lookup.items()
    }
    return normalized_index.get(normalize_injury_name(player_name))
