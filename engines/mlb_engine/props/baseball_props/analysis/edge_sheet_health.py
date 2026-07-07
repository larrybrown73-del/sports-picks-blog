from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

from baseball_props.data.data_health import DataHealthReport

T = TypeVar("T")


@dataclass
class EdgeSheetHealthReport:
    report: DataHealthReport = field(default_factory=DataHealthReport)
    skip_counts: dict[str, int] = field(default_factory=dict)

    def record_skip(self, label: str, *, player: str = "", detail: str = "") -> None:
        record_edge_skip(self, label, player=player, detail=detail)

    def merge(self, other: EdgeSheetHealthReport | None) -> None:
        if other is None:
            return
        self.report.merge(other.report)
        for label, count in other.skip_counts.items():
            self.skip_counts[label] = self.skip_counts.get(label, 0) + count

    def to_dict(self) -> dict:
        merged = self.report.to_dict()
        merged["edge_skip_counts"] = dict(self.skip_counts)
        return merged


def record_edge_skip(
    report: EdgeSheetHealthReport | None,
    label: str,
    *,
    player: str = "",
    detail: str = "",
) -> None:
    if report is None:
        return
    report.skip_counts[label] = report.skip_counts.get(label, 0) + 1
    if detail:
        report.report.record_missing(
            label,
            detail=f"{player}: {detail}" if player else detail,
        )


def safe_edge_eval(
    label: str,
    fn: Callable[[], T],
    *,
    default: T,
    report: EdgeSheetHealthReport | None = None,
) -> T:
    """Run edge evaluation; on error return default and record skip."""
    try:
        return fn()
    except Exception as exc:
        record_edge_skip(report, label, detail=f"error ({exc})")
        return default
