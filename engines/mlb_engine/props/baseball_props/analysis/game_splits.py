from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from baseball_props.data.statcast_pitches import get_pitch_locations_for_player
from baseball_props.pipeline.slate_run import SlateRunResult


def _safe_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _pitch_locations(player_id: str, role: str, *, include: bool) -> list[dict[str, Any]]:
    if not include or not player_id:
        return []
    cached = get_pitch_locations_for_player(str(player_id), "pitcher" if role == "pitcher" else "batter")
    return [dict(row) for row in cached]


def _edge_market_fields(edge_row: pd.Series | None) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "probability_pct": None,
        "ev_per_unit": None,
        "confidence_tier": None,
        "confidence_score": None,
        "kelly_fraction": None,
        "fractional_kelly_pct": None,
        "suggested_stake": None,
        "decimal_odds": None,
        "true_probability": None,
        "verdict": None,
        "warnings": None,
        "data_warnings": None,
    }
    if edge_row is None:
        return empty
    return {
        "probability_pct": _safe_float(edge_row.get("probability_pct")),
        "ev_per_unit": _safe_float(edge_row.get("ev_per_unit")),
        "confidence_tier": edge_row.get("confidence_tier"),
        "confidence_score": edge_row.get("confidence_score"),
        "kelly_fraction": _safe_float(edge_row.get("kelly_fraction")),
        "fractional_kelly_pct": _safe_float(edge_row.get("fractional_kelly_pct")),
        "suggested_stake": _safe_float(edge_row.get("suggested_stake")),
        "decimal_odds": _safe_float(edge_row.get("decimal_odds")),
        "true_probability": _safe_float(edge_row.get("true_probability")),
        "verdict": edge_row.get("verdict"),
        "warnings": edge_row.get("warnings"),
        "data_warnings": edge_row.get("data_warnings"),
    }


def _situational_fields(row: pd.Series) -> dict[str, Any]:
    tag = row.get("travel_rest_tag")
    if tag is None or (isinstance(tag, float) and math.isnan(tag)):
        tag = "Neutral"
    return {
        "travel_rest_tag": str(tag),
        "umpire_modifier": _safe_float(row.get("run_environment_modifier")) or 1.0,
        "injury_status": row.get("injury_status"),
        "data_warnings": [],
    }


def _split_lookup(
    splits: pd.DataFrame,
    player_id: str,
) -> dict[str, float | None]:
    rows = splits[splits["player_id"].astype(str) == str(player_id)]
    out: dict[str, float | None] = {
        "vs_lhp_woba": None,
        "vs_rhp_woba": None,
        "vs_lhp_wrc_plus": None,
        "vs_rhp_wrc_plus": None,
    }
    for _, row in rows.iterrows():
        split = str(row.get("split", ""))
        if split == "vs_lhp":
            out["vs_lhp_woba"] = _safe_float(row.get("woba"))
            out["vs_lhp_wrc_plus"] = _safe_float(row.get("wrc_plus"))
        elif split == "vs_rhp":
            out["vs_rhp_woba"] = _safe_float(row.get("woba"))
            out["vs_rhp_wrc_plus"] = _safe_float(row.get("wrc_plus"))
    return out


def _pitcher_split_lookup(
    splits: pd.DataFrame,
    pitcher_id: str,
) -> dict[str, dict[str, float | None]]:
    rows = splits[splits["pitcher_id"].astype(str) == str(pitcher_id)]
    out: dict[str, dict[str, float | None]] = {
        "vs_lhb": {"woba_allowed": None, "bf": None},
        "vs_rhb": {"woba_allowed": None, "bf": None},
    }
    for _, row in rows.iterrows():
        split = str(row.get("split", ""))
        if split not in out:
            continue
        out[split] = {
            "woba_allowed": _safe_float(row.get("woba_allowed")),
            "bf": _safe_float(row.get("bf")),
        }
    return out


def _sp_payload(
    *,
    sp_id: str,
    hand: str,
    name: str,
    pitcher_outs: pd.DataFrame,
    pitcher_edges: pd.DataFrame | None,
    pitcher_platoon: pd.DataFrame,
    game_id: str,
    include_pitch_locations: bool,
) -> dict[str, Any]:
    outs_row = pitcher_outs[
        (pitcher_outs["game_id"].astype(str) == str(game_id))
        & (pitcher_outs["pitcher_id"].astype(str) == str(sp_id))
    ]
    proj_outs = None
    if not outs_row.empty:
        proj_outs = _safe_float(outs_row.iloc[0].get("proj_outs"))

    edge_pct = None
    recommendation = None
    market_line = None
    if pitcher_edges is not None and not pitcher_edges.empty:
        edge_row = pitcher_edges[
            (pitcher_edges["game_id"].astype(str) == str(game_id))
            & (pitcher_edges["pitcher_name"].astype(str) == str(name))
        ]
        if edge_row.empty and not outs_row.empty:
            edge_row = pitcher_edges[
                (pitcher_edges["game_id"].astype(str) == str(game_id))
                & (
                    pitcher_edges["pitcher_name"].astype(str)
                    == str(outs_row.iloc[0].get("pitcher_name", ""))
                )
            ]
        if not edge_row.empty:
            edge_pct = _safe_float(edge_row.iloc[0].get("edge_pct"))
            recommendation = edge_row.iloc[0].get("recommendation")
            market_line = _safe_float(edge_row.iloc[0].get("market_line"))
            market_fields = _edge_market_fields(edge_row.iloc[0])
        else:
            market_fields = _edge_market_fields(None)
    else:
        market_fields = _edge_market_fields(None)

    platoon = _pitcher_split_lookup(pitcher_platoon, sp_id)
    return {
        "pitcher_id": sp_id,
        "name": name,
        "hand": hand,
        "proj_outs": proj_outs,
        "market_line": market_line,
        "edge_pct": edge_pct,
        "recommendation": recommendation,
        "vs_lhb": platoon["vs_lhb"],
        "vs_rhb": platoon["vs_rhb"],
        "pitch_locations": _pitch_locations(sp_id, "pitcher", include=include_pitch_locations),
        **market_fields,
    }


def _batter_row(
    row: pd.Series,
    splits: pd.DataFrame,
    edges: pd.DataFrame | None,
    *,
    include_pitch_locations: bool,
) -> dict[str, Any]:
    player_id = str(row.get("player_id", ""))
    player_name = str(row.get("player_name", player_id))
    game_id = str(row.get("game_id", ""))
    split_info = _split_lookup(splits, player_id)

    edge_pct = None
    recommendation = None
    market_line = None
    edge_series: pd.Series | None = None
    if edges is not None and not edges.empty:
        edge_row = edges[
            (edges["game_id"].astype(str) == game_id)
            & (edges["player_name"].astype(str) == player_name)
        ]
        if not edge_row.empty:
            edge_series = edge_row.iloc[0]
            edge_pct = _safe_float(edge_series.get("edge_pct"))
            recommendation = edge_series.get("recommendation")
            market_line = _safe_float(edge_series.get("market_line"))

    return {
        "player_id": player_id,
        "player_name": player_name,
        "lineup_slot": int(row.get("lineup_slot", 0) or 0),
        "bat_hand": str(row.get("bat_hand", "R")),
        "batter_hand_active": str(row.get("batter_hand_active", row.get("bat_hand", "R"))),
        "split_key": str(row.get("split_key", "")),
        "vs_lhp_woba": split_info["vs_lhp_woba"],
        "vs_rhp_woba": split_info["vs_rhp_woba"],
        "active_woba": _safe_float(row.get("matchup_woba")),
        "effective_woba": _safe_float(row.get("effective_woba")),
        "wrc_plus": _safe_float(row.get("wrc_plus_split")),
        "sp_vs_hand_woba": _safe_float(
            row.get("opp_sp_platoon_woba_allowed", row.get("opp_sp_woba_allowed"))
        ),
        "proj_hits": _safe_float(row.get("proj_hits")),
        "proj_tb": _safe_float(row.get("proj_total_bases")),
        "hits_market_line": market_line,
        "market_line": market_line,
        "edge_pct": edge_pct,
        "recommendation": recommendation,
        "pitch_locations": _pitch_locations(player_id, "batter", include=include_pitch_locations),
        **_edge_market_fields(edge_series),
        **_situational_fields(row),
    }


def build_game_split_export(
    result: SlateRunResult,
    *,
    include_pitch_locations: bool = True,
) -> list[dict[str, Any]]:
    """Build per-game matchup split payloads for canvas / JSON export."""
    games = result.frames["slate_games"]
    projected = result.projected
    splits = result.frames["matchup_splits"]
    pitcher_platoon = result.frames.get("pitcher_platoon_splits", pd.DataFrame())
    vegas = result.frames["vegas_totals"]
    pitcher_outs = result.pitcher_outs
    batter_edges = result.batter_edge_sheet
    pitcher_edges = result.pitcher_edge_sheet
    pitcher_names = {
        str(r["pitcher_id"]): str(r["pitcher_name"])
        for _, r in pitcher_outs.iterrows()
        if pd.notna(r.get("pitcher_name"))
    }

    vegas_by_game = (
        {str(r["game_id"]): r for _, r in vegas.iterrows()} if not vegas.empty else {}
    )

    export: list[dict[str, Any]] = []
    for _, game in games.iterrows():
        game_id = str(game["game_id"])
        away_team = str(game["away_team_id"])
        home_team = str(game["home_team_id"])
        label = f"{away_team} @ {home_team}"

        game_proj = projected[projected["game_id"].astype(str) == game_id]
        away_lineup = game_proj[game_proj["team_id"].astype(str) == away_team].sort_values(
            "lineup_slot"
        )
        home_lineup = game_proj[game_proj["team_id"].astype(str) == home_team].sort_values(
            "lineup_slot"
        )

        vegas_row = vegas_by_game.get(game_id)
        game_total = _safe_float(vegas_row["game_total"]) if vegas_row is not None else None

        sp_home_id = str(game.get("sp_home_id", ""))
        sp_away_id = str(game.get("sp_away_id", ""))
        home_sp_name = pitcher_names.get(sp_home_id, f"SP {sp_home_id}" if sp_home_id else "TBD")
        away_sp_name = pitcher_names.get(sp_away_id, f"SP {sp_away_id}" if sp_away_id else "TBD")

        sample_row = game_proj.iloc[0] if not game_proj.empty else None
        umpire_name = "TBD"
        umpire_mod = 1.0
        away_rest = 1
        home_rest = 1
        if sample_row is not None:
            umpire_name = str(sample_row.get("umpire_name", "TBD"))
            umpire_mod = _safe_float(sample_row.get("run_environment_modifier")) or 1.0
        away_rows = game_proj[game_proj["team_id"].astype(str) == away_team]
        home_rows = game_proj[game_proj["team_id"].astype(str) == home_team]
        if not away_rows.empty and pd.notna(away_rows.iloc[0].get("days_rest")):
            away_rest = int(away_rows.iloc[0]["days_rest"])
        if not home_rows.empty and pd.notna(home_rows.iloc[0].get("days_rest")):
            home_rest = int(home_rows.iloc[0]["days_rest"])

        data_warnings: list[str] = []
        if result.meta.get("data_health"):
            data_warnings = list(result.meta["data_health"].get("warnings", []))[:5]

        export.append(
            {
                "game_id": game_id,
                "label": label,
                "away_team": away_team,
                "home_team": home_team,
                "game_total": game_total,
                "umpire": {
                    "name": umpire_name,
                    "zone_size_modifier": _safe_float(
                        sample_row.get("zone_size_modifier") if sample_row is not None else None
                    )
                    or 1.0,
                    "run_environment_modifier": umpire_mod,
                },
                "travel_rest": {
                    "away_days_rest": away_rest,
                    "home_days_rest": home_rest,
                },
                "data_health_warnings": data_warnings,
                "away_sp": _sp_payload(
                    sp_id=sp_away_id,
                    hand=str(game.get("sp_away_hand", "R")),
                    name=away_sp_name,
                    pitcher_outs=pitcher_outs,
                    pitcher_edges=pitcher_edges,
                    pitcher_platoon=pitcher_platoon,
                    game_id=game_id,
                    include_pitch_locations=include_pitch_locations,
                ),
                "home_sp": _sp_payload(
                    sp_id=sp_home_id,
                    hand=str(game.get("sp_home_hand", "R")),
                    name=home_sp_name,
                    pitcher_outs=pitcher_outs,
                    pitcher_edges=pitcher_edges,
                    pitcher_platoon=pitcher_platoon,
                    game_id=game_id,
                    include_pitch_locations=include_pitch_locations,
                ),
                "away_lineup": [
                    _batter_row(
                        r,
                        splits,
                        batter_edges,
                        include_pitch_locations=include_pitch_locations,
                    )
                    for _, r in away_lineup.iterrows()
                ],
                "home_lineup": [
                    _batter_row(
                        r,
                        splits,
                        batter_edges,
                        include_pitch_locations=include_pitch_locations,
                    )
                    for _, r in home_lineup.iterrows()
                ],
            }
        )

    export.sort(key=lambda g: g["label"])
    return export


def filter_conviction_plays(
    plays: list[dict[str, Any]],
    *,
    market_line: float | None = None,
    side: str | None = None,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Subset conviction rows and re-rank by absolute edge."""
    filtered = list(plays)
    if market_line is not None:
        filtered = [
            play
            for play in filtered
            if play.get("line") is not None
            and abs(float(play["line"]) - float(market_line)) < 1e-6
        ]
    if side is not None:
        side_norm = side.strip().lower()
        filtered = [
            play
            for play in filtered
            if str(play.get("rec") or "").strip().lower() == side_norm
        ]
    filtered.sort(key=lambda play: abs(float(play.get("edge") or 0.0)), reverse=True)
    if top_n is not None and top_n >= 0:
        filtered = filtered[:top_n]
    return filtered


def _summary_stats_for_plays(plays: list[dict[str, Any]]) -> dict[str, Any]:
    tier_counts: dict[str, int] = {}
    total_stake = 0.0
    for play in plays:
        tier = str(play.get("confidence_tier") or "Unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        stake = play.get("suggested_stake")
        if stake is not None:
            total_stake += float(stake)
    return {
        "play_count": len(plays),
        "tier_counts": tier_counts,
        "total_suggested_exposure": round(total_stake, 2),
    }


def build_betting_intel_export(
    result: SlateRunResult,
    *,
    slate_date: date | None = None,
    market_line: float | None = None,
    side: str | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Build conviction + summary payload for canvas betting-intel tab."""
    from baseball_props.analysis.parlay_builder import tickets_to_records

    conviction = result.conviction
    plays: list[dict[str, Any]] = []
    if conviction is not None and not conviction.empty:
        for _, row in conviction.iterrows():
            plays.append(
                {
                    "player": row.get("player_name"),
                    "market": row.get("market"),
                    "model_value": _safe_float(row.get("model_value")),
                    "line": _safe_float(row.get("market_line")),
                    "prob": _safe_float(row.get("probability_pct")),
                    "edge": _safe_float(row.get("edge_pct")),
                    "rec": row.get("recommendation"),
                    "ev_per_unit": _safe_float(row.get("ev_per_unit")),
                    "confidence_tier": row.get("confidence_tier"),
                    "kelly_fraction": _safe_float(row.get("kelly_fraction")),
                    "suggested_stake": _safe_float(row.get("suggested_stake")),
                }
            )

    plays = filter_conviction_plays(
        plays,
        market_line=market_line,
        side=side,
        top_n=top_n,
    )
    play_summary = _summary_stats_for_plays(plays)

    data_health = result.meta.get("data_health") or {}
    resolved_date = slate_date
    if resolved_date is None and "game_date" in result.frames.get("slate_games", pd.DataFrame()).columns:
        games_df = result.frames["slate_games"]
        if not games_df.empty:
            try:
                resolved_date = date.fromisoformat(str(games_df.iloc[0]["game_date"])[:10])
            except ValueError:
                resolved_date = None
    if resolved_date is None:
        resolved_date = date.today()

    parlay_tickets = tickets_to_records(result.parlay_tickets or [])
    return {
        "slate_date": resolved_date.isoformat(),
        "conviction_plays": plays,
        "parlay_tickets": parlay_tickets,
        "summary_stats": {
            **play_summary,
            "data_health_warning_count": data_health.get("warning_count", 0),
            "parlay_ticket_count": len(parlay_tickets),
        },
        "data_health_warnings": data_health.get("warnings", []),
    }
