"""Export today's value picks as JSON for canvas display."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from performance_check import get_upcoming_value_picks


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    game_date = date.today()
    picks = get_upcoming_value_picks(game_date)

    rows = [
        {
            "matchup": f"{pick.away_name} @ {pick.home_name}",
            "play": pick.play,
            "line": pick.american_odds,
            "edgePct": round(pick.edge_pct, 2),
            "modelWinProb": round(pick.model_win_prob * 100, 1),
            "qkPct": round(pick.quarter_kelly_pct, 2),
            "confidence": pick.confidence_label,
            "homeRuns": round(pick.pred_home_runs, 2),
            "awayRuns": round(pick.pred_away_runs, 2),
            "totalRuns": round(pick.pred_home_runs + pick.pred_away_runs, 2),
            "homeWinProb": round(pick.home_win_prob * 100, 1),
            "awayWinProb": round(pick.away_win_prob * 100, 1),
            "temp": pick.temperature,
            "wind": pick.wind,
            "bullpen": pick.bullpen_status,
        }
        for pick in picks
    ]

    payload = {"date": game_date.isoformat(), "pickCount": len(rows), "picks": rows}
    text = json.dumps(payload, indent=2)
    if out_path:
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path} ({len(rows)} picks)")
    else:
        print(text)


if __name__ == "__main__":
    main()
