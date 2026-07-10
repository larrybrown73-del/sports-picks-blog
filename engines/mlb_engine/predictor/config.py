"""Central configuration for baseball-predictor slate evaluation and backtests."""

from __future__ import annotations

# Backtest / value-pick thresholds
EDGE_THRESHOLD = 0.03
QUARTER_KELLY = 0.25
FRACTIONAL_KELLY = QUARTER_KELLY
MAX_BET_PCT = 5.0
TIER1_MIN_EDGE_PCT = 5.0
TIER2_MIN_EDGE_PCT = 3.0
TIER3_MIN_EDGE_PCT = 0.5
MIN_PLAYABLE_EV = 0.0
DEFAULT_SAMPLE_SIZE = 600
DEFAULT_PREDICTION_SEASONS = [2024, 2025, 2026]
DEFAULT_ROLLING_WINDOW = 10
MOCK_MONEYLINE_VIG = 1.05

# Weather / park run environment (post-model adjustment)
WEATHER_HOT_TEMP_F = 85
WEATHER_COLD_TEMP_F = 55
RUN_ENV_BOOST_MULTIPLIER = 1.06
RUN_ENV_REDUCE_MULTIPLIER = 0.94
RUN_ENV_NEUTRAL_MULTIPLIER = 1.0

# Bullpen fatigue / late-inning adjustments (innings 7-9 run layer)
BULLPEN_LOOKBACK_DAYS = 3
BULLPEN_FATIGUE_WINDOW_HOURS = 48
BULLPEN_FATIGUE_PITCH_THRESHOLD = 35
BULLPEN_FATIGUE_WIN_PROB_PENALTY = 0.03
BULLPEN_MIN_RELIEF_APPEARANCES = 1

# Dead Arm / Lockdown late-inning scalars (applied before win probability)
OVERWORKED_BULLPEN_PENALTY: float = 1.18
RESTED_ELITE_BONUS: float = 0.88
OVERWORKED_CONSECUTIVE_DAYS: int = 2
OVERWORKED_PITCH_THRESHOLD_3D: int = 45
ELITE_BULLPEN_TOP_N: int = 10
BULLPEN_HIGH_LEVERAGE_ARMS: int = 2
LATE_INNING_RUN_SHARE: float = 0.33

# Optional per-game evaluation log
SLATE_EVALUATION_LOG = "slate_evaluation_log.csv"

# Moneyline defensive guardrails
MIN_PROBABILITY_FLOOR: float = 0.40
MAX_ODDS_CAP: int = 160

# Team momentum (hot hand)
TEAM_STREAK_MIN_WINS: int = 3
TEAM_STREAK_BONUS: float = 1.03

# Full-season pitcher stability guardrails (post-model run adjustment)
BABIP_LUCK_THRESHOLD: float = 0.320
BABIP_LUCK_ERA_CEILING: float = 3.60
BABIP_LUCK_BONUS: float = 0.95

SMOKE_MIRRORS_ERA_CEILING: float = 3.50
SMOKE_MIRRORS_WHIP_FLOOR: float = 1.35
SMOKE_MIRRORS_BABIP_CEILING: float = 0.270
REGRESSION_PENALTY: float = 1.15

# Power pitcher vs. velo-struggling lineups (offensive run penalty)
POWER_PITCHER_VELO_FLOOR: float = 95.5
VELO_MATCHUP_FASTBALL_MPH: float = 95.0
VELO_DOMINANCE_SCALAR: float = 0.88
VELO_STRUGGLE_BOTTOM_N: int = 10

# Ground-ball pitcher vs. patient lineup (offensive run boost)
GROUND_BALL_PITCHER_GB_PCT: float = 50.0
PATIENT_LINEUP_BB_PCT: float = 9.5
PATIENT_LINEUP_PITCHES_PER_PA: float = 3.95
PATIENT_LINEUP_ADVANTAGE: float = 1.12

# Hitter plate discipline and lineup slot factors
DISCIPLINE_BONUS_ELITE_BB_PCT: float = 12.0
DISCIPLINE_BONUS: float = 1.04
ERRATIC_SWINGER_K_PCT: float = 26.0
ERRATIC_SWINGER_BB_PCT: float = 6.5
ERRATIC_SWINGER_PENALTY: float = 0.85
PREMIUM_SLOT_MAX: int = 4
PREMIUM_SLOT_SCALAR: float = 1.05
BOTTOM_ORDER_PENALTY: float = 0.80

# Starter rest and rotation hierarchy (runs-allowed layer)
SHORT_REST_PENALTY: float = 1.12
OPTIMAL_REST_MIN_DAYS: int = 5
OPTIMAL_REST_MAX_DAYS: int = 6
OPTIMAL_REST_BONUS: float = 0.95
RUST_MIN_DAYS: int = 9
RUST_PENALTY: float = 1.05
TOP_OF_ROTATION_SCALAR: float = 0.90
BACK_END_STARTER_PENALTY: float = 1.15
ACE_RUN_SUPPRESSION_FACTOR: float = 0.92
STARTER_REST_LOOKBACK_DAYS: int = 120

# Ace dominance score (gates TOP_OF_ROTATION_SCALAR behind bat-missing profile)
TRUE_ACE_K_BB_PCT: float = 20.0
TRUE_ACE_WHIP_MAX: float = 1.05
ELITE_ACE_SCALAR: float = 0.82
CONTACT_STARTER_K_BB_PCT: float = 14.0
CONTACT_STARTER_MAX_BONUS: float = 0.95
ACE_DOMINANCE_MIN_BATFERS_FACED: int = 100

# Team defense cross-reference for contact starters (innings eaters only)
GOLD_GLOVE_BOOST: float = 0.98
POOR_DEFENSE_PENALTY: float = 1.12
ELITE_DEFENSE_TOP_N: int = 5
POOR_DEFENSE_BOTTOM_N: int = 5
DEFENSE_OAA_MIN_ATTEMPTS: int = 100
