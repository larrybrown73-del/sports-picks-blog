from __future__ import annotations

# Run-scoring index vs league average (1.00 = neutral). Approximate 2024 park factors.
PARK_SCORING_FACTORS: dict[str, float] = {
    # Mock slate park codes
    "FEN": 1.05,
    "ORC": 0.92,
    # MLB venue IDs (home stadium)
    "1": 1.02,  # Angel Stadium
    "2": 0.98,  # Chase Field
    "3": 1.05,  # Fenway Park
    "4": 0.97,  # Great American Ball Park
    "5": 0.99,  # Progressive Field
    "6": 1.12,  # Coors Field
    "7": 0.96,  # Dodger Stadium
    "8": 0.92,  # Oracle Park (legacy)
    "9": 1.00,  # Kauffman Stadium
    "10": 1.02,  # Oakland Coliseum
    "11": 1.00,  # Petco Park
    "12": 1.01,  # Tropicana Field
    "13": 1.00,  # Globe Life Field
    "14": 1.03,  # Rogers Centre
    "15": 1.01,  # Wrigley Field
    "16": 0.99,  # Guaranteed Rate Field
    "17": 1.00,  # Minute Maid Park
    "18": 1.02,  # Nationals Park
    "19": 1.12,  # Coors Field (alt)
    "22": 1.00,  # Citi Field
    "23": 1.01,  # Citizens Bank Park
    "2394": 0.96,  # Dodger Stadium
    "2395": 0.92,  # Oracle Park
    "2681": 1.00,  # Truist Park
    "2682": 1.01,  # LoanDepot Park
    "2683": 1.00,  # Target Field
    "2862": 1.02,  # Busch Stadium
    "2889": 1.01,  # PNC Park
    "3289": 1.03,  # Yankee Stadium
    "3312": 1.00,  # T-Mobile Park
    "3313": 1.03,  # Yankee Stadium (alt)
    "4169": 1.00,  # American Family Field
    "4705": 1.01,  # Comerica Park
    "680": 1.00,  # Camden Yards
}


def get_park_scoring_factor(park_id: str) -> float:
    """Return run-scoring multiplier for a park; 1.0 when unknown."""
    key = str(park_id).strip()
    return PARK_SCORING_FACTORS.get(key, 1.0)


# MLB venue IDs with fixed or typical closed roofs (non-outdoor for weather bonuses).
DOMED_PARK_IDS: frozenset[str] = frozenset(
    {
        "2",  # Chase Field
        "12",  # Tropicana Field
        "13",  # Globe Life Field
        "14",  # Rogers Centre
        "17",  # Minute Maid Park
        "2682",  # loanDepot Park
        "4169",  # American Family Field
    }
)


def is_outdoor_venue(park_id: str) -> bool:
    """Return True when the venue is treated as outdoor for weather TB bonuses."""
    return str(park_id).strip() not in DOMED_PARK_IDS
