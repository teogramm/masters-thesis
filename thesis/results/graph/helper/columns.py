from typing import Literal

import polars as pl

from thesis.model.enums import Location


pretty_names_cols = {
    "speed": "Speed (m/s)",
    "acc": "Acceleration (m/s2)",
    "relative_apex_pos": "Position relative to apex in travel direction (m)",
    "long_pos": "Longitudinal position (m)",
    "direction": "Direction",
    "dist_right": "Distance to the right edge of the path in travel direction (m)",
    "dist_left": "Distance to the left edge of the path in travel direction (m)",
    "cuts_corner": "Cuts corner",
    "width": "Path width (m)",
    "curvature": "Curvature",
    "lat_pos": "Distance from centre line (m)",
    "elevation": "Elevation (m)",
    "following_time_headway": "Headway to the bike in front (s)",
    "following_lateral_deviation": "Lateral distance to the bike in front (m)",
    "following_long_dist": "Longitudinal distance to the bike in front (m)",
    "following_speed_difference": "Relative speed difference (m/s)",
    "meeting_lateral_dist": "Lateral distance (m)",
    "location": "Location",
    "time_of_day": "Time of day",
    "primary_type": "Type"
}
"""Names to use when displaying a column in a graph."""


pretty_names_location = {
    Location.RIDDARHUSKAJEN: "Riddarhuskajen",
    Location.RIDDARHOLMSBRON_N: "Riddarholmsbron N",
    Location.RIDDARHOLMSBRON_S: "Riddarholmsbron S"
}
"""Full names for the locations."""

XColType = Literal["long_pos", "relative_apex_pos"]
"""Available values for X columns in graphs."""

YColType = Literal["speed", "acc", "lat_pos"]
"""Available values for Y columns in graphs, for which filtering has been implemented."""


def ensure_single_value(trajectories: pl.DataFrame, col: str) -> None:
    """
    Ensures that the given column in the given dataframe has only a single value.
    If not, a ValueError is raised.
    :raises ValueError: If the given column has more than one value.
    """
    if trajectories.select(pl.col(col).n_unique()).item() > 1:
        raise ValueError(f"More than one value in the {col} column")
