"""MLB stadium coordinates for Open-Meteo weather lookups."""

from __future__ import annotations

# Home ballpark coordinates keyed by MLB team id (home_id).
TEAM_STADIUM_COORDS: dict[int, tuple[float, float]] = {
    108: (33.800293, -117.882690),   # Los Angeles Angels
    109: (33.445302, -112.066687),   # Arizona Diamondbacks
    110: (39.283819, -76.621611),    # Baltimore Orioles
    111: (42.346676, -71.097217),    # Boston Red Sox
    112: (41.948438, -87.655332),    # Chicago Cubs
    113: (39.097389, -84.506831),    # Cincinnati Reds
    114: (41.496211, -81.685234),    # Cleveland Guardians
    115: (39.755891, -104.994178),   # Colorado Rockies
    116: (42.339020, -83.048520),    # Detroit Tigers
    117: (29.757268, -95.355455),    # Houston Astros
    118: (39.051565, -94.480559),    # Kansas City Royals
    119: (34.073851, -118.240089),   # Los Angeles Dodgers
    120: (25.778056, -80.219722),    # Miami Marlins
    121: (40.757087, -73.845848),    # New York Mets
    133: (37.751594, -122.200546),   # Oakland Athletics
    134: (40.446903, -80.005636),    # Pittsburgh Pirates
    135: (32.707174, -117.156877),   # San Diego Padres
    136: (47.591423, -122.332490),   # Seattle Mariners
    137: (37.778214, -122.389256),   # San Francisco Giants
    138: (38.622566, -90.192845),    # St. Louis Cardinals
    139: (27.768226, -82.653392),    # Tampa Bay Rays
    140: (32.747300, -97.083100),    # Texas Rangers
    141: (43.641778, -79.389045),    # Toronto Blue Jays
    142: (44.981829, -93.277891),    # Minnesota Twins
    143: (39.906057, -75.166472),    # Philadelphia Phillies
    144: (33.890667, -84.467722),    # Atlanta Braves
    145: (41.829902, -87.633640),    # Chicago White Sox
    146: (38.873010, -77.007432),    # Washington Nationals
    147: (40.829643, -73.926175),    # New York Yankees
    158: (43.028034, -87.971272),    # Milwaukee Brewers
}

# Override coordinates for specific venues (neutral sites, temporary parks, etc.).
VENUE_COORDS: dict[int, tuple[float, float]] = {
    14: (43.641778, -79.389045),     # Rogers Centre
    3289: (40.829643, -73.926175),   # Yankee Stadium
    4705: (41.948438, -87.655332),   # Wrigley Field
    680: (39.755891, -104.994178),   # Coors Field
}


def get_stadium_coords(home_team_id: int, venue_id: int | None = None) -> tuple[float, float] | None:
    """Return (latitude, longitude) for a game at the home team's ballpark."""
    if venue_id is not None and venue_id in VENUE_COORDS:
        return VENUE_COORDS[venue_id]
    return TEAM_STADIUM_COORDS.get(home_team_id)
