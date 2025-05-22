from datetime import timedelta
from typing import Optional

import polars as pl


def calculate_meeting_statistics(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculates the distance relative
    :param trajectories: Trajectories dataframe with ``meeting_ids`` column 
    :return: Dataframe with meeting_info column
    """
    trajectories = trajectories.select("location", "time", "ID", "meeting_ids", "lat_pos",
                                       "long_pos", "in_path")
    meetings = trajectories.explode("meeting_ids").rename({"meeting_ids": "meeting_id"})


    meetings = meetings.join(meetings.drop("meeting_id"), left_on=["location", "time", "meeting_id"],
                             right_on=["location", "time", "ID"], how="inner", suffix="_opposite")
    meetings = meetings.filter(pl.col("in_path") & pl.col("in_path_opposite"))
    # Keep observations where the longitudinal position of this bike is equal to the longitudinal position of the
    # opposite bike (the bikes are meeting)
    meetings = meetings.filter(pl.col("long_pos") == pl.col("long_pos_opposite"))
    meetings = meetings.drop("long_pos_opposite").rename({"long_pos": "meeting_long_pos"})
    # Due to the aggregation of long_pos into discrete values every 0.5m some meetings might create duplicate rows
    # (duplicate long_pos and/or long_pos_opposite).
    # To have just one entry per meeting, calculate the average of the rest of the columns.
    meetings = meetings.group_by("location", "ID", "meeting_id").agg(
        pl.selectors.numeric().mean()
    )
    # Calculate the distance between the bikes
    lateral_distance = (abs(pl.col("lat_pos") - pl.col("lat_pos_opposite"))).alias("meeting_lateral_dist")
    meetings = meetings.with_columns(lateral_distance)
    # lateral_distances = meetings.group_by("location", "ID").agg(pl.col("meeting_lateral_dist").sort_by("meeting_id"))
    meetings = meetings.filter(pl.col("lat_pos") * pl.col("lat_pos_opposite") < 0)
    
    # Calculate the relative positions for each meeting
    meetings = meetings.select("location", "ID", "meeting_id", "meeting_long_pos", "meeting_lateral_dist")
    meetings = meetings.group_by("location", "ID").agg(
        pl.struct("meeting_id", "meeting_long_pos", "meeting_lateral_dist").alias("meeting_info")
    )
    # trajectories = trajectories.join(meetings, on=["location", "ID"], how="inner")
    # relative_pos = (pl.col("long_pos") - pl.col("meeting_long_pos")).alias("meeting_relative_pos")
    # trajectories = trajectories.with_columns(relative_pos).group_by("location", "time", "ID").agg(
    #     pl.col("meeting_relative_pos").sort_by("meeting_id")
    # )
    
    return meetings



def _calculate_meeting_trajectories(trajectories: pl.LazyFrame, headway_threshold: Optional[int] = 5) -> pl.LazyFrame:
    meeting = trajectories.rolling(offset="0s", period=timedelta(seconds=headway_threshold), index_column="time",
                                   group_by=["location", "long_pos"], closed="both").agg(
        # For each timestamp we see if there are multiple direction values.
        # We use closed=both to include the observations directly on the timestamp, to include the exact point of the
        # meeting if two trajectories are in the same long_pos.
        (pl.col("direction").n_unique() > 1).alias("is_meeting"),
        # Keep information about all trajectories crossing the longitudinal position within the next threshold seconds.
        pl.struct(meeting_ids=pl.col("ID"), meeting_direction=pl.col("direction")).unique().alias("meeting_info")
        # Create a dataframe containing a row for each crossing at each timestamp within the threshold
    ).filter(pl.col("is_meeting")).explode("meeting_info").unnest("meeting_info")

    # Join meetings to the main trajectories dataframe and for every observation remove other observations going in
    # the same direction
    meeting = trajectories.select(
        ["location", "ID", "time", "long_pos", "direction"]
    ).join(
        # For each time and longitudinal position combination, add all the trajectories that met during the window
        # to all the trajectories crossing the longitudinal position at that time.
        meeting, on=["location", "long_pos", "time"], how="inner"
    ).filter(
        # Keep only meetings where the trajectories travel in opposite directions
        (pl.col("meeting_direction") != pl.col("direction"))
        # Remove unnecessary columns and keep only a list with the IDs of meeting trajectories
    ).drop(["meeting_direction", "is_meeting", "direction"]).group_by(["location", "ID", "time", "long_pos"]).all()

    # Add information about meeting trajectories to each datapoint (defined by id, longitudinal position and time)
    trajectories = trajectories.join(meeting, on=["location", "ID", "long_pos", "time"], how="left").with_columns(
        pl.col("meeting_ids").list.set_difference(pl.col("ID")).fill_null(pl.lit([],
                                                                                 dtype=pl.List(pl.UInt16)))
    )

    return trajectories
