import pandas as pd
import pytest
from datetime import date

from baseball_props.analysis.batter_projection import (
    marcels_regress,
    platoon_hitting_rate,
    project_batter_total_bases,
)
from baseball_props.config import (
    INJURY_RUST_DAYS_TO_FULL,
    INJURY_RUST_MIN_MULTIPLIER,
    LEAGUE_AVG,
    LEAGUE_SLG,
    LEAGUE_TB_PER_GAME,
    REGRESSION_PA_STABILIZATION,
    TB_PER_SLG_PA,
    TB_PER_WOBA_PA,
)
from baseball_props.environment.parks import get_park_scoring_factor


def test_mock_slate_hard_hit_splits_vary_by_player() -> None:
    from baseball_props.data.ingest import build_slate_context, load_slate_frames

    context = build_slate_context(load_slate_frames(source="mock"))
    hard_hit = context.player_games["matchup_hard_hit_pct"].dropna()
    assert hard_hit.nunique() > 1


def test_platoon_uses_wrc_plus_split_over_global_woba() -> None:
    df = pd.DataFrame(
        [
            {
                "matchup_wrc_plus": 120.0,
                "matchup_woba": 0.280,
                "effective_woba": 0.320,
            }
        ]
    )
    rate = platoon_hitting_rate(df)
    expected = (120.0 / 100.0) * LEAGUE_AVG["woba"]
    assert rate.iloc[0] == pytest.approx(expected)
    assert rate.iloc[0] != pytest.approx(0.320)


def test_marcels_regress_uses_300_pa_stabilization() -> None:
    player = pd.Series([0.400])
    regressed = marcels_regress(
        player,
        LEAGUE_AVG["woba"],
        pd.Series([100.0]),
    )
    weight = 100.0 / REGRESSION_PA_STABILIZATION
    expected = weight * 0.400 + (1 - weight) * LEAGUE_AVG["woba"]
    assert regressed.iloc[0] == pytest.approx(expected)
    assert REGRESSION_PA_STABILIZATION == 300.0


def test_marcels_full_weight_at_300_pa() -> None:
    player = pd.Series([0.400])
    regressed = marcels_regress(
        player,
        LEAGUE_AVG["woba"],
        pd.Series([300.0]),
    )
    assert regressed.iloc[0] == pytest.approx(0.400)


def test_park_factor_boosts_total_bases_at_fenway() -> None:
    base_row = {
        "player_id": "H001",
        "park_id": "ORC",
        "matchup_wrc_plus": 100.0,
        "matchup_iso": LEAGUE_AVG["iso"],
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.0,
        "season_pa": 500,
    }
    fen_row = {**base_row, "park_id": "FEN"}
    orc = project_batter_total_bases(pd.DataFrame([base_row]))
    fen = project_batter_total_bases(pd.DataFrame([fen_row]))

    assert fen.iloc[0]["proj_total_bases"] > orc.iloc[0]["proj_total_bases"]
    assert fen.iloc[0]["park_tb_factor"] == pytest.approx(get_park_scoring_factor("FEN"))
    assert orc.iloc[0]["park_tb_factor"] == pytest.approx(get_park_scoring_factor("ORC"))


def test_low_sample_pa_reduces_extreme_total_bases() -> None:
    high_skill = {
        "park_id": "ORC",
        "matchup_wrc_plus": 140.0,
        "matchup_iso": 0.280,
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.5,
        "season_pa": 40,
    }
    full_sample = {**high_skill, "season_pa": 400}
    low = project_batter_total_bases(pd.DataFrame([high_skill]))
    high = project_batter_total_bases(pd.DataFrame([full_sample]))
    assert low.iloc[0]["proj_total_bases"] < high.iloc[0]["proj_total_bases"]


def test_low_sample_regresses_toward_one_tb_per_game() -> None:
    row = {
        "park_id": "ORC",
        "matchup_wrc_plus": 150.0,
        "matchup_iso": 0.300,
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.5,
        "season_pa": 0,
    }
    result = project_batter_total_bases(pd.DataFrame([row]))
    park = get_park_scoring_factor("ORC")
    assert result.iloc[0]["proj_total_bases"] == pytest.approx(LEAGUE_TB_PER_GAME * park)


def test_high_iso_low_pa_regresses_iso_heavily() -> None:
    row = {
        "park_id": "ORC",
        "matchup_woba": 0.350,
        "matchup_iso": 0.300,
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.0,
        "season_pa": 60,
    }
    result = project_batter_total_bases(pd.DataFrame([row]))
    weight = 60.0 / REGRESSION_PA_STABILIZATION
    expected_iso = weight * 0.300 + (1 - weight) * LEAGUE_AVG["iso"]
    assert result.iloc[0]["regressed_iso"] == pytest.approx(expected_iso)
    assert result.iloc[0]["regressed_iso"] < 0.300


def test_rookie_low_pa_dampens_extreme_skill_tb_spike() -> None:
    row = {
        "park_id": "ORC",
        "matchup_wrc_plus": 155.0,
        "matchup_iso": 0.320,
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.5,
        "season_pa": 80,
    }
    result = project_batter_total_bases(pd.DataFrame([row]))
    park = get_park_scoring_factor("ORC")

    assert result.iloc[0]["skill_tb"] > 2.5
    assert result.iloc[0]["proj_total_bases"] < 2.0
    raw_regressed = marcels_regress(
        pd.Series([result.iloc[0]["skill_tb"]]),
        LEAGUE_TB_PER_GAME,
        pd.Series([80.0]),
    ).iloc[0]
    assert result.iloc[0]["regressed_game_tb"] == pytest.approx(raw_regressed, rel=1e-3)
    assert result.iloc[0]["proj_total_bases"] == pytest.approx(
        result.iloc[0]["regressed_game_tb"] * park, rel=1e-3
    )


def test_total_bases_formula_components() -> None:
    row = {
        "park_id": "ORC",
        "matchup_woba": 0.350,
        "matchup_iso": LEAGUE_AVG["iso"],
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.0,
        "season_pa": REGRESSION_PA_STABILIZATION,
    }
    result = project_batter_total_bases(pd.DataFrame([row]))
    regressed_woba = result.iloc[0]["regressed_woba"]
    regressed_iso = result.iloc[0]["regressed_iso"]
    regressed_slg = result.iloc[0]["regressed_slg"]
    park = get_park_scoring_factor("ORC")

    contact_tb = regressed_woba * 4.0 * TB_PER_WOBA_PA
    power_ratio = regressed_iso / LEAGUE_AVG["iso"]
    skill_tb = (contact_tb * power_ratio + regressed_slg * 4.0 * TB_PER_SLG_PA) / 2.0
    regressed_game_tb = marcels_regress(
        pd.Series([skill_tb]),
        LEAGUE_TB_PER_GAME,
        pd.Series([REGRESSION_PA_STABILIZATION]),
    ).iloc[0]
    expected = regressed_game_tb * park

    assert result.iloc[0]["skill_tb"] == pytest.approx(skill_tb, rel=1e-3)
    assert result.iloc[0]["regressed_game_tb"] == pytest.approx(regressed_game_tb, rel=1e-3)
    assert result.iloc[0]["proj_total_bases"] == pytest.approx(expected, rel=1e-3)


def _base_tb_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "player_id": "P001",
        "player_name": "Healthy Player",
        "park_id": "ORC",
        "matchup_wrc_plus": 100.0,
        "matchup_iso": LEAGUE_AVG["iso"],
        "opp_sp_woba_allowed": LEAGUE_AVG["woba"],
        "proj_pa": 4.0,
        "season_pa": 500,
    }
    row.update(overrides)
    return row


def test_project_batter_total_bases_applies_injury_lookup() -> None:
    from pathlib import Path

    from baseball_props.data.injuries import parse_injury_html

    fixture = Path(__file__).parent / "fixtures" / "fantasypros_injuries_sample.html"
    injury_lookup = parse_injury_html(fixture.read_text(encoding="utf-8"), today=date(2025, 6, 27))

    healthy = project_batter_total_bases(
        pd.DataFrame([_base_tb_row(player_name="Healthy Player")]),
        injury_lookup=injury_lookup,
    )
    il_player = project_batter_total_bases(
        pd.DataFrame([_base_tb_row(player_name="Jordan Lawlar")]),
        injury_lookup=injury_lookup,
    )
    returning = project_batter_total_bases(
        pd.DataFrame([_base_tb_row(player_name="Nick Kurtz")]),
        injury_lookup=injury_lookup,
    )
    dtd = project_batter_total_bases(
        pd.DataFrame([_base_tb_row(player_name="Jacob Wilson")]),
        injury_lookup=injury_lookup,
    )

    healthy_tb = float(healthy.iloc[0]["proj_total_bases"])
    assert float(il_player.iloc[0]["proj_total_bases"]) == 0.0
    assert float(il_player.iloc[0]["injury_multiplier"]) == 0.0

    rust_mult = INJURY_RUST_MIN_MULTIPLIER + (1 - INJURY_RUST_MIN_MULTIPLIER) * 3 / INJURY_RUST_DAYS_TO_FULL
    assert float(returning.iloc[0]["injury_multiplier"]) == pytest.approx(rust_mult)
    assert float(returning.iloc[0]["proj_total_bases"]) == pytest.approx(healthy_tb * rust_mult, rel=1e-3)

    assert float(dtd.iloc[0]["injury_multiplier"]) == 1.0
    assert float(dtd.iloc[0]["proj_total_bases"]) == pytest.approx(healthy_tb, rel=1e-3)
