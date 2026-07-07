from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from baseball_props.analysis.edge_sheets import SKIP_DISPLAY_RECOMMENDATIONS
from baseball_props.config import (
    HITS_PROP_PRIMARY_LINE,
    HITS_PROP_TARGET_LINES,
    PARLAY_LEGS_PER_TICKET,
    PARLAY_MAX_PLAYER_EXPOSURE,
    PARLAY_TICKET_COUNT,
)


@dataclass(frozen=True)
class PropLeg:
    player_id: str
    player_name: str
    game_id: str
    market: str
    line: float
    recommendation: str
    edge_pct: float
    ev_per_unit: float | None = None
    confidence_tier: str | None = None


@dataclass
class ParlayTicket:
    ticket_id: int
    legs: list[PropLeg]
    combined_edge_proxy: float


def _leg_key(leg: PropLeg) -> tuple[str, str, float, str]:
    return (leg.player_id, leg.market, leg.line, leg.recommendation)


def legs_from_batter_sheet(batter_sheet: pd.DataFrame) -> list[PropLeg]:
    """Convert filtered batter edge sheet rows into PropLeg candidates."""
    legs: list[PropLeg] = []
    if batter_sheet is None or batter_sheet.empty:
        return legs

    candidates = batter_sheet[
        batter_sheet["market_line"].notna()
        & batter_sheet["edge_pct"].notna()
        & batter_sheet["market_line"].isin(HITS_PROP_TARGET_LINES)
        & (batter_sheet["recommendation"] == "Over")
    ].copy()
    if "recommendation" in candidates.columns:
        candidates = candidates[~candidates["recommendation"].isin(SKIP_DISPLAY_RECOMMENDATIONS)]
    if "verdict" in candidates.columns:
        candidates = candidates[candidates["verdict"].fillna("Play") == "Play"]

    for _, row in candidates.iterrows():
        edge = row.get("edge_pct")
        if edge is None or pd.isna(edge) or float(edge) <= 0:
            continue
        player_id = str(row.get("player_id") or row.get("player_name", ""))
        legs.append(
            PropLeg(
                player_id=player_id,
                player_name=str(row.get("player_name", "")),
                game_id=str(row.get("game_id", "")),
                market=str(row.get("market", "batter_hits")),
                line=float(row.get("market_line", HITS_PROP_PRIMARY_LINE)),
                recommendation=str(row.get("recommendation", "Over")),
                edge_pct=float(edge),
                ev_per_unit=float(row["ev_per_unit"])
                if pd.notna(row.get("ev_per_unit"))
                else None,
                confidence_tier=str(row.get("confidence_tier"))
                if pd.notna(row.get("confidence_tier"))
                else None,
            )
        )
    return legs


def build_diversified_tickets(
    legs: list[PropLeg] | pd.DataFrame,
    *,
    ticket_count: int = PARLAY_TICKET_COUNT,
    legs_per_ticket: int = PARLAY_LEGS_PER_TICKET,
    max_player_exposure: int = PARLAY_MAX_PLAYER_EXPOSURE,
) -> list[ParlayTicket]:
    """
    Build diversified parlay tickets from TB Play legs or a batter edge sheet.

    Greedy assignment prefers highest-edge legs while capping identical-leg exposure
    and favoring different games per ticket when possible.
    """
    if isinstance(legs, pd.DataFrame):
        leg_list = legs_from_batter_sheet(legs)
    else:
        leg_list = list(legs)

    def _tier_rank(tier: str | None) -> int:
        order = {
            "Tier-1 High Conviction": 3,
            "Tier-2 Standard": 2,
            "Tier-3 Marginal": 1,
        }
        return order.get(tier or "", 0)

    candidates = sorted(
        [leg for leg in leg_list if leg.edge_pct > 0 and leg.recommendation == "Over"],
        key=lambda leg: (_tier_rank(leg.confidence_tier), leg.edge_pct),
        reverse=True,
    )
    if not candidates:
        return []

    exposure: dict[tuple[str, str, float, str], int] = {}
    tickets: list[list[PropLeg]] = [[] for _ in range(ticket_count)]

    for ticket_idx in range(ticket_count):
        used_games: set[str] = set()
        for leg in candidates:
            if len(tickets[ticket_idx]) >= legs_per_ticket:
                break
            key = _leg_key(leg)
            if exposure.get(key, 0) >= max_player_exposure:
                continue
            if leg.game_id in used_games:
                continue
            tickets[ticket_idx].append(leg)
            exposure[key] = exposure.get(key, 0) + 1
            used_games.add(leg.game_id)

    for ticket_idx in range(ticket_count):
        if len(tickets[ticket_idx]) >= legs_per_ticket:
            continue
        for leg in candidates:
            if len(tickets[ticket_idx]) >= legs_per_ticket:
                break
            key = _leg_key(leg)
            if exposure.get(key, 0) >= max_player_exposure:
                continue
            if any(_leg_key(existing) == key for existing in tickets[ticket_idx]):
                continue
            tickets[ticket_idx].append(leg)
            exposure[key] = exposure.get(key, 0) + 1

    result: list[ParlayTicket] = []
    ticket_id = 1
    for ticket_legs in tickets:
        if len(ticket_legs) < legs_per_ticket:
            continue
        combined = sum(leg.edge_pct for leg in ticket_legs) / len(ticket_legs)
        result.append(
            ParlayTicket(
                ticket_id=ticket_id,
                legs=ticket_legs,
                combined_edge_proxy=round(combined, 2),
            )
        )
        ticket_id += 1
    return result


def tickets_to_records(tickets: list[ParlayTicket]) -> list[dict[str, object]]:
    """Serialize parlay tickets for JSON export."""
    out: list[dict[str, object]] = []
    for ticket in tickets:
        out.append(
            {
                "ticket_id": ticket.ticket_id,
                "combined_edge_proxy": ticket.combined_edge_proxy,
                "legs": [
                    {
                        "game_id": leg.game_id,
                        "player_id": leg.player_id,
                        "player_name": leg.player_name,
                        "market": leg.market,
                        "line": leg.line,
                        "recommendation": leg.recommendation,
                        "edge_pct": leg.edge_pct,
                    }
                    for leg in ticket.legs
                ],
            }
        )
    return out
