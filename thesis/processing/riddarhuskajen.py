import polars as pl
import numpy as np

from thesis.model.enums import Location


def calculate_centerline_curvature(rk: pl.LazyFrame) -> pl.LazyFrame:
    """
    Uses datapoints along the centre line to calculate the curvature of the trajectory.
    :param rk: Dataframe with ``location``, ``long_pos``, ``lat_pos``, ``X`` and ``Y`` columns.
    :return: DataFrame with ``location``, ``long_pos`` and ``curvature`` columns.
    """
    # Keep only trajectories from Riddarhuskajen

    rk = rk.filter(pl.col("location") == Location.RIDDARHUSKAJEN, ~pl.col("issue"))

    min_long_pos = 0
    max_long_pos = 40
    all_long_pos = np.arange(min_long_pos, max_long_pos + 0.1, step=0.5)
    center_trajectory = pl.DataFrame({
        "lat_pos": [0.0] * len(all_long_pos),
        "long_pos": all_long_pos
    })
    center_trajectory = center_trajectory.lazy().join(rk.with_columns(
        pl.col("lat_pos").round(2)
    ).select("X", "Y", "long_pos", "lat_pos"),
                                                      on=["long_pos", "lat_pos"], how="left")
    center_trajectory = center_trajectory.group_by("long_pos", "lat_pos").agg(
        pl.col("X", "Y").mean()
    )
    center_trajectory = (center_trajectory.with_columns(location=pl.lit(Location.RIDDARHUSKAJEN), ID=1)
                         .sort(by="long_pos").with_row_index(name="time"))
    center_trajectory = calculate_curvature(center_trajectory)
    center_trajectory = center_trajectory.select("location", "long_pos", "X", "Y", "curvature")
    return center_trajectory


def calculate_curvature(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates the curvature at each datapoint.
    :param trajectories: DataFrame with ``location``, ``ID``, ``time``, ``X`` and ``Y`` columns.
    :return: Input DataFrame with ``curvature`` column.
    """
    curvature = trajectories.group_by("location", "ID").agg(
        pl.col("time").sort(),
        pl.col("X").sort_by("time").map_batches(np.gradient).alias("xd"),
        pl.col("Y").sort_by("time").map_batches(np.gradient).alias("yd"),
        pl.col("X").sort_by("time").map_batches(np.gradient).map_batches(np.gradient).alias("xdd"),
        pl.col("Y").sort_by("time").map_batches(np.gradient).map_batches(np.gradient).alias("ydd")
    ).explode("time", "xd", "yd", "xdd", "ydd")

    trajectories = trajectories.join(curvature, how="left", on=["location", "time", "ID"])
    trajectories = trajectories.with_columns(
        curvature=((pl.col("xd") * pl.col("ydd")) - (pl.col("yd") * pl.col("xdd")) /
                   (((pl.col("xd").pow(2)) + pl.col("yd").pow(2)).pow(1.5))),
    ).drop("xd", "xdd", "yd", "ydd")

    return trajectories


def calculate_position_relative_to_apex(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds the position relative to the curve's apex in the trajectory's direction of travel, when the trajectory is
    inside the path.
    For trajectories outside the path or not in Riddarhuskajen, null is added.
    :param trajectories: DataFrame with ``location``, ``long_pos`` and ``in_path`` columns.
    :return: Input Dataframe with the ``relative_apex_pos`` column.
    """
    all_trajectories = trajectories
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN)

    apex_long_pos = 21.5

    trajectories = trajectories.with_columns(pl.col("long_pos").alias("relative_apex_pos"))
    # Since longitudinal position is increasing northbound, make the southbound trajectories relative to that point
    trajectories = trajectories.with_columns(pl.when(pl.col("direction") == "Southbound")
                                             .then(pl.col("long_pos").max() - pl.col("relative_apex_pos"))
                                             .otherwise(pl.col("relative_apex_pos")).alias("relative_apex_pos"))
    # Remove values for observations not in the path
    trajectories = trajectories.with_columns(pl.when(~pl.col("in_path")).then(None).
                                             otherwise(pl.col("relative_apex_pos")).alias("relative_apex_pos"))
    trajectories = trajectories.with_columns(pl.col("relative_apex_pos") - apex_long_pos)

    all_trajectories = all_trajectories.join(trajectories.select("location", "time", "ID", "relative_apex_pos"),
                                             how="left", on=["location", "time", "ID"], validate="1:1")

    return all_trajectories


def calculate_cuts_corner(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates which trajectories in Riddarhuskajen cut the corner.
    :param trajectories: DataFrame with ``location``, ``ID``, ``direction``, ``long_pos`` and ``lat_pos`` columns.
    :return: Mapping of each ``location`` and ``ID`` to a cuts_corner value 
    """
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN)
    cuts_corner_northbound = ((pl.col("direction") == "Northbound") &
                              (pl.col("long_pos").is_between(11, 23)) &
                              (pl.col("lat_pos") > 0).any() &
                              (pl.col("lat_pos") < 0))
    cuts_corner_southbound = ((pl.col("direction") == "Southbound") &
                              (pl.col("long_pos").is_between(14, 26)) &
                              (pl.col("lat_pos") < 0).any() &
                              (pl.col("lat_pos") > 0))
    trajectories = trajectories.group_by("location", "ID").agg(
        (cuts_corner_northbound | cuts_corner_southbound).any().alias("cuts_corner")
    )
    return trajectories


def calculate_crossings_into_opposite_lane(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    crossings = trajectories.sort("time")
    crosses_from_east_to_west = (pl.col("lat_pos").shift(-1) < 0) & (pl.col("lat_pos") > 0)
    crosses_from_west_to_east = (pl.col("lat_pos").shift(-1) > 0) & (pl.col("lat_pos") < 0)
    crosses_lane = crosses_from_west_to_east | crosses_from_east_to_west
    # Get the earliest crossing into the opposite lane, so we can start from when the trajectory was in the correct
    #direction
    first_crossing_into_opposite_lane = pl.col("time").filter(
        (crosses_from_east_to_west & (pl.col("direction") == "Northbound")) |
        (crosses_from_west_to_east & (pl.col("direction") == "Southbound"))
    ).min().over("location", "ID")
    
    # After the first time the trajectory crosses into the opposite lane, it either crosses back into the correct
    # lane (at least once)
    crossings = ((crossings.filter(pl.col("time") >= first_crossing_into_opposite_lane,
                                   pl.col("lat_pos") != 0)
                  ).group_by(
        "location", "ID"
    ).agg(
        pl.col("long_pos").filter(crosses_lane).gather_every(2, 0).alias("crossing_start"),
        pl.col("time").filter(crosses_lane).gather_every(2, 0).alias("crossing_start_time"),
        pl.col("long_pos").filter(crosses_lane).gather_every(2, 1).alias("crossing_end"),
        pl.col("time").filter(crosses_lane).gather_every(2, 1).alias("crossing_end_time"),
    ).filter(pl.col("crossing_start").list.len() > 0,
             pl.col("crossing_end").list.len() > 0)
    .with_columns(
        pl.col("crossing_start", "crossing_end", "crossing_start_time", "crossing_end_time").list
        .gather(
            pl.int_ranges(pl.min_horizontal(pl.col("crossing_start").list.len(),
                                                pl.col("crossing_end").list.len())))
    )).explode("crossing_start", "crossing_end", "crossing_start_time", "crossing_end_time")
    
    # Calculate the distance travelled by taking all the observations in between each crossing
    crossings = crossings.join_where(trajectories.select("location", "ID", "dist", "time", "lat_pos"),
                                     pl.col("location") == pl.col("location_right"),
                                     pl.col("ID") == pl.col("ID_right"),
                                     pl.col("time").is_between(pl.col("crossing_start_time"),
                                                                           pl.col("crossing_end_time")))
    crossings = crossings.group_by("location", "ID", "crossing_start", "crossing_end").agg(
        pl.col("dist").sum().round(1).alias("crossing_dist"),
        pl.col("lat_pos").max().round(2).alias("crossing_max_lat_pos"),
    ).group_by("location", "ID").agg(
        pl.struct("crossing_start", "crossing_end", "crossing_dist", "crossing_max_lat_pos").alias("line_crossing_info")
    )
    
    return crossings
