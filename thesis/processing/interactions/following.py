from datetime import timedelta
from typing import Optional

import polars as pl

from thesis.processing.interpolation import interpolate_for_long_pos


def _calculate_following_trajectories(trajectories: pl.LazyFrame, headway_threshold: Optional[int] = 5) -> pl.LazyFrame:
    # To find if a trajectory is constrained by any trajectory in the same direction for every timestamp, we check the
    # trajectories that had the same longitudinal position in the last 5s.
    # Those are the trajectories that the bike is following.
    # As the same timestamp can appear multiple times, but the result is the same, keep only unique rows
    following = trajectories.rolling(period=timedelta(seconds=headway_threshold), index_column="time",
                                     group_by=["location", "direction", "long_pos"], closed="left").agg(
        pl.col("ID").unique().alias("following_ids")
    ).unique()

    # Join the main trajectories dataframe and for each entry remove its ID number if it appears in the following IDs
    trajectories = trajectories.join(following, on=["location", "direction", "long_pos", "time"],
                                     how="left").with_columns(
        pl.col("following_ids").list.set_difference(pl.col("ID"))
    )

    return trajectories


def calculate_following_parameters(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates:
    
    * ID of the trajectory being followed (``following_id``)
    * Following longitudinal distance (``following_long_dist``)
    * Lateral deviation (``following_lateral_deviation``)
    * Time headway (``following_time_headway``)
    * Speed difference (``following_speed_difference``)
    relative to the bike that is following.
    For example, a negative speed difference means that the bike ahead is slower.
    
    In addition the ``following_interpolated`` column indicates that the headway for the bike ahead have been
    was interpolated (in case of a missing exact longitudinal position).
    
    Overtakes affect the sign of the values.
    For example, a negative following distance means that the bike that the bike that is followed is actually behind.
    A positive headway means that the bike ahead passed the same longitudinal position before us, while a negative
    headway means that the bike ahead passes the same longitudinal position after.
    
    A positive lateral deviation indicates that the bike ahead is to the right of the bike behind, while a negative
    lateral deviation indicates that the bike ahead is to the left of the bike behind.
    :param trajectories: DataFrame with ``following_ids`` column.
    :return: DataFrame containing for each ``location``, ``ID`` and ``time`` a list of structs with the above
    information in the ``following_info`` columns.
    To use, explode and unnest the ``following_info`` column.
    """
    following = trajectories.select("location", "ID", "time", "following_ids", "lat_pos", "long_pos", "direction",
                                    "speed", "X", "Y", "in_path")
    following = following.explode("following_ids").rename({"following_ids": "following_id"})
    # For each datapoint, we want to add where the followed trajectory was at that moment
    # Keep direction as it is necessary during processing, following and followed trajectories have the same value
    # anyway.
    following = (following.join(following.drop("following_id"), how="inner",
                                left_on=["location", "time", "following_id"],
                                right_on=["location", "time", "ID"], suffix="_ahead")
                 .filter(pl.col("following_id").is_not_null()))
    # Keep only trajectories in path
    following = following.filter(pl.col("in_path"), pl.col("in_path_ahead"))
    # Filter after the join, since a trajectory might be following another that is not following anything. 
    # Calculate statistics
    # In southbound trajectories, the bike ahead has a smaller long_dist.
    # long_distance = pl.when(pl.col("direction") == "Southbound").then(
    #     -(pl.col("long_pos_ahead") - pl.col("long_pos"))
    # ).otherwise(pl.col("long_pos_ahead") - pl.col("long_pos")).alias("following_long_dist")

    # In southbound trajectories both values are negative and smaller values are further from the centreline, so we
    # invert the sign again to make it left/right relative to the following cyclist.
    lateral_dev = pl.when(pl.col("direction") == "Southbound").then(
        -(pl.col("lat_pos_ahead") - pl.col("lat_pos"))
    ).otherwise(pl.col("lat_pos_ahead") - pl.col("lat_pos")).alias("following_lateral_deviation")

    # Do not add square root
    # long_dist^2 = dist^2 - lat_dist^2
    xy_dist = ((pl.col("X") - pl.col("X_ahead")).pow(2) + (pl.col("Y") - pl.col("Y_ahead")).pow(2))
    long_distance = (xy_dist - lateral_dev.pow(2)).sqrt()
    # We need to invert the sign if the bike ahead is actually behind
    invert_sign_northbound = (pl.col("direction") == "Northbound") & ((pl.col("long_pos") > pl.col("long_pos_ahead"))
                              | ((pl.col("long_pos") == pl.col("long_pos_ahead")) & (pl.col("Y") < pl.col("Y_ahead"))))
    invert_sign_southbound = (pl.col("direction") == "Southbound") & ((pl.col("long_pos") < pl.col("long_pos_ahead"))
                              | ((pl.col("long_pos") == pl.col("long_pos_ahead")) & (pl.col("Y") > pl.col("Y_ahead"))))
    long_distance = pl.when(invert_sign_southbound | invert_sign_northbound).then(-long_distance).otherwise(
        long_distance
    ).alias("following_long_dist")

    speed_diff = (pl.col("speed_ahead") - pl.col("speed")).alias("following_speed_difference")

    following = _calculate_time_headway(trajectories, following)
    following = following.with_columns(long_distance, lateral_dev, speed_diff).unique()
    
    # Remove negative
    following = following.filter(pl.col("following_long_dist") > 0,
                                 pl.col("following_time_headway") > 0)
    
    # Since each trajectory can only follow one trajectory at time, for each datapoint, keep only the closest trajectory
    # ahead
    following = following.group_by(["location", "time","ID"]).agg(
        pl.all().sort_by("following_long_dist").first()
    )
    
    
    # Collect everything into a single column 
    following = following.group_by(["location", "time", "ID"]).agg(
        pl.struct("following_id", "following_long_dist", "following_lateral_deviation",
                  "following_speed_difference", "following_time_headway").alias("following_info")
    )
    return following


def _calculate_time_headway(all_trajectories: pl.LazyFrame, following: pl.LazyFrame, interpolate=False):
    # To calculate the time headway,
    # we need to find out when the trajectory ahead crossed the same longitudinal position
    # Interpolate the trajectories ahead. We don't care about speed the function just wants a parameter.
    if interpolate:
        # Interpolate only IDs that are being followed
        all_trajectories = all_trajectories.join(following,
                                                 left_on=["location", "ID"], right_on=["location", "following_id"],
                                                 how="semi")
        # The function does not currently accept empty interpolation, so interpolate speed
        all_trajectories = interpolate_for_long_pos(all_trajectories, ["speed"])

    following = following.join(all_trajectories.select("location", "ID", "long_pos", "time"),
                               how="inner",
                               left_on=["location", "following_id", "long_pos"],
                               right_on=["location", "ID", "long_pos"], suffix="_ahead")

    # The time the trajectory ahead crossed will be lower than the time this trajectory crossed,
    # so invert the sign to make it a positive number.
    time_headway = (pl.col("time") - pl.col("time_ahead")).dt.total_milliseconds().alias("following_time_headway")
    if interpolate:
        interpolated = pl.col("interpolated_ahead").alias("following_headway_interpolated")
        following = following.with_columns(interpolated)
    return following.with_columns(time_headway)
