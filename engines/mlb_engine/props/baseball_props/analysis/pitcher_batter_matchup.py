"""Bridge hitter discipline with opposing pitcher style scalars (predictor engine)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from baseball_props.analysis.hitter_discipline import (
    BatterDisciplineProfile,
    is_elite_discipline,
    is_erratic_swinger,
)
from baseball_props.config import PREDICTOR_PATH

logger = logging.getLogger(__name__)


def _load_predictor_pitcher_matchup():
    predictor_root = Path(PREDICTOR_PATH)
    if not predictor_root.is_dir():
        return None

    path_str = str(predictor_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

    from pitcher_matchup import (  # type: ignore[import-not-found]
        fetch_pitcher_season_profile,
        is_ground_ball_pitcher,
        is_power_pitcher,
        is_velo_struggler,
        pitcher_runs_allowed_scalar,
    )
    from config import (  # type: ignore[import-not-found]
        PATIENT_LINEUP_ADVANTAGE,
        VELO_DOMINANCE_SCALAR,
    )

    return {
        "fetch_pitcher_season_profile": fetch_pitcher_season_profile,
        "is_ground_ball_pitcher": is_ground_ball_pitcher,
        "is_power_pitcher": is_power_pitcher,
        "is_velo_struggler": is_velo_struggler,
        "pitcher_runs_allowed_scalar": pitcher_runs_allowed_scalar,
        "PATIENT_LINEUP_ADVANTAGE": PATIENT_LINEUP_ADVANTAGE,
        "VELO_DOMINANCE_SCALAR": VELO_DOMINANCE_SCALAR,
    }


def apply_pitcher_hitter_matchup(
    adjusted_proj: float,
    prob_multiplier: float,
    *,
    opponent_pitcher_id: str | None,
    discipline: BatterDisciplineProfile,
    batting_mlb_team_id: int | None,
    season: int,
    adjustments: dict[str, float],
    warnings: list[str],
) -> tuple[float, float]:
    """
    Layer opposing SP stability and style matchups on hitter discipline.

    Pitcher guardrails run first; patient-eye and hero-swing profiles then
    compound with ground-ball and velo-dominance styles before edge export.
    """
    if not opponent_pitcher_id:
        return adjusted_proj, prob_multiplier

    bridge = _load_predictor_pitcher_matchup()
    if bridge is None:
        warnings.append(
            "Missing-Data Warning: predictor_pitcher_matchup — skipping SP style interaction"
        )
        return adjusted_proj, prob_multiplier

    try:
        pitcher_id = int(opponent_pitcher_id)
    except (TypeError, ValueError):
        return adjusted_proj, prob_multiplier

    pitcher = bridge["fetch_pitcher_season_profile"](pitcher_id, season=season)
    if pitcher is None:
        return adjusted_proj, prob_multiplier

    proj = float(adjusted_proj)
    prob = float(prob_multiplier)

    stability_scalar, stability_tag = bridge["pitcher_runs_allowed_scalar"](pitcher)
    if stability_tag:
        proj *= stability_scalar
        adjustments[f"pitcher_{stability_tag}"] = stability_scalar

    if bridge["is_ground_ball_pitcher"](pitcher) and is_elite_discipline(discipline):
        patient_scalar = float(bridge["PATIENT_LINEUP_ADVANTAGE"])
        proj *= patient_scalar
        prob *= patient_scalar
        adjustments["gb_pitcher_discipline_synergy"] = patient_scalar

    if bridge["is_power_pitcher"](pitcher):
        velo_scalar = float(bridge["VELO_DOMINANCE_SCALAR"])
        velo_triggered = False
        if is_erratic_swinger(discipline):
            velo_triggered = True
            adjustments["velo_erratic_synergy"] = velo_scalar
        elif batting_mlb_team_id is not None and bridge["is_velo_struggler"](
            batting_mlb_team_id, season
        ):
            velo_triggered = True
            adjustments["velo_team_struggle_synergy"] = velo_scalar
        if velo_triggered:
            proj *= velo_scalar
            prob *= velo_scalar

    return proj, prob
