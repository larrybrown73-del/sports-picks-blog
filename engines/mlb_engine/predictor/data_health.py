"""Thin data-health wrapper for predictor feature fetches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

MISSING_DATA_PREFIX = "Missing-Data Warning"


@dataclass
class DataHealthReport:
    warnings: list[str] = field(default_factory=list)

    def record_missing(self, label: str, *, detail: str = "using neutral baseline") -> None:
        message = f"{MISSING_DATA_PREFIX}: {label} — {detail}"
        if message not in self.warnings:
            self.warnings.append(message)
        logger.warning("%s: %s — %s", MISSING_DATA_PREFIX, label, detail)


def safe_feature_fetch(
    feature_name: str,
    fetch_fn: Callable[[], T],
    *,
    fallback: T,
    context: dict[str, str] | None = None,
) -> T:
    """Catch exceptions / empty profiles; log warning; return neutral baseline."""
    del context
    try:
        result = fetch_fn()
    except Exception as exc:
        logger.warning(
            "%s: %s — error (%s); using neutral baseline",
            MISSING_DATA_PREFIX,
            feature_name,
            exc,
        )
        return fallback
    if result is None:
        logger.warning(
            "%s: %s — empty profile; using neutral baseline",
            MISSING_DATA_PREFIX,
            feature_name,
        )
        return fallback
    return result
