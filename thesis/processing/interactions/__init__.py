from typing import Optional

import polars as pl

from thesis.processing.interactions.following import _calculate_following_trajectories
from thesis.processing.interactions.meeting import _calculate_meeting_trajectories


def calculate_following_meeting_ids(trajectories: pl.LazyFrame,
                                    headway_threshold: Optional[int] = 5) -> pl.LazyFrame:
    """
    Adds information to trajectories about meetings and followings.
    The results are calculated for the entirety of the trajectory.
    :param headway_threshold: Headway in seconds for when to consider a cyclist constrained.
    :param trajectories: DataFrame with trajectories of both bicyclists and pedestrians.
    :return: Dataframe with ``location``, ``ID``, ``meeting_ids`` and ``following_ids`` columns.
    """

    # Cast to 16-bit unsigned integer to save memory
    trajectories = trajectories.with_columns(pl.col("ID").cast(pl.UInt16))

    # Sort the trajectories by their time to apply the rolling function
    trajectories = trajectories.sort("time")

    trajectories = _calculate_meeting_trajectories(trajectories, headway_threshold)
    trajectories = _calculate_following_trajectories(trajectories, headway_threshold)

    # Convert the results from per-datapoint to per-trajectory
    trajectories = trajectories.group_by("location", "ID").agg(
        pl.col("meeting_ids").flatten().drop_nulls().unique(),
        pl.col("following_ids").flatten().drop_nulls().unique()
    )

    return trajectories


def add_constrained_type(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Uses the ``meeting_ids`` and ``following_ids`` columns to add information about the constraints.
    :return: Dataframe with ``constrained`` column with text about the constraint type
    """
    constrained_type = (pl.when(pl.col("overtakes_id").list.len() > 0).then(pl.lit("Overtakes")).when(
        pl.col("meeting_ids").list.len() > 0,
        pl.col("following_ids").list.len() > 0).then(pl.lit("Both, does not overtake"))
                        .when(pl.col("meeting_ids").list.len() > 0).then(pl.lit("Only meeting"))
                        .when(pl.col("following_ids").list.len() > 0).then(pl.lit("Only following"))
                        .otherwise(pl.lit("Unconstrained")).cast(pl.Categorical))
    return trajectories.with_columns(constrained=constrained_type)
