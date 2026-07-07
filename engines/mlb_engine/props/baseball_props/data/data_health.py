from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import pandas as pd

from baseball_props.logging_utils import get_logger, log_once

logger = get_logger(__name__)

T = TypeVar("T")

MISSING_DATA_PREFIX = "Missing-Data Warning"


@dataclass
class DataHealthReport:
    warnings: list[str] = field(default_factory=list)
    fallback_counts: dict[str, int] = field(default_factory=dict)
    missing_slices: list[str] = field(default_factory=list)

    def record_missing(self, label: str, *, detail: str = "using league-average baseline") -> None:
        message = f"{MISSING_DATA_PREFIX}: {label} — {detail}"
        self.missing_slices.append(label)
        self.fallback_counts[label] = self.fallback_counts.get(label, 0) + 1
        if message not in self.warnings:
            self.warnings.append(message)
        log_once(
            f"missing_data_{label}",
            logger,
            logging.WARNING,
            "%s: %s — %s",
            MISSING_DATA_PREFIX,
            label,
            detail,
        )

    def merge(self, other: DataHealthReport | None) -> None:
        if other is None:
            return
        for warning in other.warnings:
            if warning not in self.warnings:
                self.warnings.append(warning)
        for label in other.missing_slices:
            if label not in self.missing_slices:
                self.missing_slices.append(label)
        for key, count in other.fallback_counts.items():
            self.fallback_counts[key] = self.fallback_counts.get(key, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "warnings": list(self.warnings),
            "fallback_counts": dict(self.fallback_counts),
            "missing_slices": list(self.missing_slices),
            "warning_count": len(self.warnings),
        }


def safe_feature_slice(
    label: str,
    fn: Callable[[], T],
    *,
    default: T,
    report: DataHealthReport | None = None,
    empty_check: Callable[[T], bool] | None = None,
) -> T:
    """Run feature extraction; on error or empty result return default and record warning."""
    try:
        result = fn()
    except Exception as exc:
        if report is not None:
            report.record_missing(label, detail=f"error ({exc}); using league-average baseline")
        else:
            logger.warning("%s: %s failed: %s", MISSING_DATA_PREFIX, label, exc)
        return default

    is_empty = False
    if empty_check is not None:
        is_empty = empty_check(result)
    elif isinstance(result, pd.DataFrame):
        is_empty = result.empty
    elif result is None:
        is_empty = True

    if is_empty and report is not None:
        report.record_missing(label, detail="empty profile; using league-average baseline")
        return default
    if is_empty:
        return default
    return result
