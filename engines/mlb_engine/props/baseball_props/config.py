from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# 2024-ish league average anchors (rate stats)
LEAGUE_AVG: dict[str, float] = {
    "woba": 0.313,
    "iso": 0.145,
    "k_pct": 0.223,
    "bb_pct": 0.085,
    "wrc_plus": 100.0,
    "hard_hit_pct": 0.389,
}

STRICT_MODE: bool = False

# Rolling baseline blend weights (must sum to 1.0)
DEFAULT_W14: float = 0.45
DEFAULT_W30: float = 0.35
DEFAULT_W_SEASON: float = 0.20

# Team plate appearances ≈ implied_runs * this factor (calibrated ~8.5 for MLB)
LEAGUE_PA_PER_RUN: float = 8.5

# Fallback implied team runs when Vegas totals are missing for a game
LEAGUE_AVG_IMPLIED_RUNS: float = 4.2

# Batting-order slot weights (share of team PA opportunity)
SLOT_PA_WEIGHTS: dict[int, float] = {
    1: 1.12,
    2: 1.08,
    3: 1.05,
    4: 1.03,
    5: 1.00,
    6: 0.97,
    7: 0.94,
    8: 0.91,
    9: 0.88,
}

# Environment adjustment sensitivity (phase 1 stubs)
TEMP_BASELINE_F: float = 72.0
TEMP_RUNS_FACTOR_PER_DEG: float = 0.003
WIND_OUT_FACTOR_PER_MPH: float = 0.015

# wOBA * PA * factor → fair total bases for market comparison
TB_PER_WOBA_PA: float = 1.60

# Marcels stabilization: full weight on player rates at this many season PA
REGRESSION_PA_STABILIZATION: float = 300.0
# Backward-compatible alias
REGRESSION_PA_FULL: float = REGRESSION_PA_STABILIZATION

# Conservative total-bases anchor for unproven (low-PA) hitters
LEAGUE_TB_PER_GAME: float = 1.0

# Hits per PA anchor (~0.24 league-wide; used for batter hits projection)
LEAGUE_HITS_PER_PA: float = 0.235
LEAGUE_HITS_PER_GAME: float = 0.88

# Slugging proxy scale: league SLG ≈ wOBA + ISO → ~2 TB over 4 PA
LEAGUE_SLG: float = LEAGUE_AVG["woba"] + LEAGUE_AVG["iso"]
TB_PER_SLG_PA: float = 2.0 / (LEAGUE_SLG * 4.0)

# Pitcher pitch-count simulation (~15.6 pitches per inning at league average)
LEAGUE_PITCHES_PER_OUT: float = 5.3
PITCHES_PER_WALK: float = 5.5
PITCHES_PER_STRIKEOUT: float = 4.8
LEAGUE_STARTER_OUTS: float = 17.0
FALLBACK_STARTER_OUTS: float = 15.0
FALLBACK_RELIEF_OUTS: float = 3.0
MAX_PROJ_OUTS: float = 27.0
MIN_PITCH_EFFICIENCY: float = 3.5
MAX_PITCH_EFFICIENCY: float = 8.0
HOOK_FLOOR_MIN_OUTS: float = 15.0
HOOK_FLOOR_MIN_PITCHES: float = 85.0
HOOK_FLOOR_MAX_PITCHES: float = 105.0
MARQUEE_STARTER_OUTS: float = 16.5

# Injury rust: minimum TB multiplier for recently activated players; full recovery at N days
INJURY_RUST_MIN_MULTIPLIER: float = 0.75
INJURY_RUST_DAYS_TO_FULL: float = 14.0

# Normal-model uncertainty for prop edge sheets (continuous stat vs market line)
EDGE_TB_SIGMA: float = 0.55
EDGE_HITS_SIGMA: float = 0.42
EDGE_OUTS_SIGMA: float = 1.75

# Hits prop guardrail thresholds (Over 0.5 / Over 1.5 from market aggregator)
HITS_PROP_TARGET_LINES: tuple[float, ...] = (0.5, 1.5)
HITS_PROP_PRIMARY_LINE: float = 1.5
HITS_CONTACT_ROLLING_GAMES: int = 15
HITS_CONTACT_K_PCT_MAX: float = 0.18
HITS_CONTACT_PCT_FLOOR: float = 0.78
HITS_BABIP_FLOOR: float = 0.280
HITS_CONTACT_BONUS_MULTIPLIER: float = 1.06
HITS_LINEUP_TOP_SLOT: int = 4
HITS_LINEUP_SLOT_PENALTY: float = 0.75
HITS_WEATHER_TEMP_BOOST_F: float = 75.0
HITS_WEATHER_BONUS_MULTIPLIER: float = 1.05
HITS_PARK_HIT_BONUS_THRESHOLD: float = 1.03
HITS_BULLPEN_FATIGUE_BONUS: float = 0.04
HITS_MIN_ADJUSTED_EDGE_PCT: float = 3.0

# Over 1.5 TB prop filter thresholds (deprecated — use HITS_PROP_*)
TB_PROP_TARGET_LINE: float = 1.5
TB_FILTER_TARGET_LINE: float = TB_PROP_TARGET_LINE  # backward-compatible alias
TB_LINEUP_TOP_SLOT: int = 4
TB_LINEUP_TOP_SLOT_MAX: int = TB_LINEUP_TOP_SLOT  # backward-compatible alias
TB_LINEUP_SLOT_PENALTY: float = 0.75
TB_XBH_ROLLING_GAMES: int = 15
TB_XBH_MIN_RATE_PER_PA: float = 0.08
TB_XBH_MIN_RATE: float = TB_XBH_MIN_RATE_PER_PA  # backward-compatible alias
TB_SINGLES_DOMINANT_RATIO: float = 0.70
TB_WEATHER_TEMP_BOOST_F: float = 75.0
TB_WEATHER_BONUS_MULTIPLIER: float = 1.05
TB_PARK_POWER_BONUS_THRESHOLD: float = 1.03
TB_BULLPEN_FATIGUE_BONUS: float = 0.04
TB_MIN_ADJUSTED_EDGE_PCT: float = 3.0
TB_BULLPEN_FATIGUE_THRESHOLD: float = 0.65

# Legacy env bonus names (used by existing bullpen module)
TB_ENV_TEMP_BONUS_F: float = TB_WEATHER_TEMP_BOOST_F
TB_ENV_TEMP_BONUS: float = TB_WEATHER_BONUS_MULTIPLIER
TB_ENV_PARK_POWER_THRESHOLD: float = TB_PARK_POWER_BONUS_THRESHOLD
TB_ENV_PARK_POWER_BONUS: float = TB_WEATHER_BONUS_MULTIPLIER

# Parlay ticket diversification
PARLAY_TICKET_COUNT: int = 3
PARLAY_LEGS_PER_TICKET: int = 2
PARLAY_MAX_PLAYER_EXPOSURE: int = 2
PARLAY_MAX_IDENTICAL_LEGS: int = PARLAY_MAX_PLAYER_EXPOSURE
PARLAY_DEFAULT_TICKET_COUNT: int = PARLAY_TICKET_COUNT
PARLAY_DEFAULT_LEGS: int = PARLAY_LEGS_PER_TICKET

# Market intelligence: EV, Kelly, confidence tiers
KELLY_FRACTION: float = 0.25
KELLY_MAX_STAKE_PCT: float = 0.02
DEFAULT_BANKROLL: float = 1000.0
CONFIDENCE_TIERS: list[tuple[float, str]] = [
    (5.0, "Tier-1 High Conviction"),
    (3.0, "Tier-2 Moderate"),
    (1.0, "Tier-3 Speculative"),
]
TIER_KELLY_MULTIPLIERS: dict[str, float] = {
    "Tier-1 High Conviction": 0.50,
    "Tier-2 Moderate": 0.25,
    "Tier-3 Speculative": 0.125,
    "Below Threshold": 0.0,
    "No Bet": 0.0,
}
# Backward-compatible aliases for baseball_props.market.calculations
FRACTIONAL_KELLY: float = KELLY_FRACTION
MAX_BET_PCT: float = KELLY_MAX_STAKE_PCT * 100.0
MIN_PLAYABLE_EV: float = 0.0
TIER1_MIN_EDGE_PCT: float = 5.0
TIER2_MIN_EDGE_PCT: float = 3.0
TIER3_MIN_EDGE_PCT: float = 0.5

# Travel/rest logistics (48-hour schedule lookback)
LOGISTICS_LOOKBACK_HOURS: int = 48
TRAVEL_REST_B2B_PENALTY: float = 0.98
TRAVEL_REST_LONG_MILES: float = 1500.0
TRAVEL_REST_LONG_MILES_PENALTY: float = 0.97
TRAVEL_REST_TZ_DELTA_THRESHOLD: int = 2
TRAVEL_REST_TZ_PENALTY: float = 0.98

# Default slate ingest mode (live requires API keys in .env)
DEFAULT_SLATE_SOURCE: Literal["mock", "live"] = "live"

# baseball-predictor bridge for optional live bullpen fatigue import
_DEFAULT_PREDICTOR = Path(__file__).resolve().parents[2] / "predictor"
PREDICTOR_PATH: str = os.environ.get("BASEBALL_PREDICTOR_PATH", str(_DEFAULT_PREDICTOR))


@dataclass(frozen=True)
class RollingWeights:
    w14: float = DEFAULT_W14
    w30: float = DEFAULT_W30
    w_season: float = DEFAULT_W_SEASON

    def __post_init__(self) -> None:
        total = self.w14 + self.w30 + self.w_season
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Rolling weights must sum to 1.0, got {total}")


DEFAULT_ROLLING_WEIGHTS = RollingWeights()

RATE_METRICS: list[str] = ["woba", "iso", "k_pct", "bb_pct"]
PROBABILITY_METRICS: list[str] = ["k_pct", "bb_pct"]
ADVANCED_SPLIT_METRICS: list[str] = ["wrc_plus", "hard_hit_pct"]
