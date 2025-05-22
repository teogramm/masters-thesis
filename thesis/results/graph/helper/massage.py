"""
Ensures a consistent sample size across all long_pos.
"""
import numpy as np
import polars as pl

from thesis.processing.interpolation import interpolate_for_long_pos, deduplicate_by_long_pos


def smooth_sample_size(trajectories: pl.LazyFrame, interpolate_cols: list[str],
                       copy_cols: list[str]) -> pl.LazyFrame:
    """
    Ensures a consistent sample size across all longitudinal positions.
    Interpolates missing values for intermediate longitudinal positions in a trajectory.
    If a trajectory contains multiple values for the same longitudinal position, the average is taken.
    :param trajectories: DataFrame with ``long_pos`` and the columns given in the other parameters.
    :param interpolate_cols: Names of numerical columns to linearly interpolate.
    :param copy_cols: Columns whose value is copied from any other point in the trajectory.
    Should be used with columns that characterise the trajectory (e.g. ``direction`` ).
    :return: LazyFrame with each trajectory having exactly one value at each longitudinal position between its first
    and last point.
    """
    trajectories = interpolate_for_long_pos(trajectories, interpolate_cols, copy_cols)
    trajectories = deduplicate_by_long_pos(trajectories, interpolate_cols, copy_cols)
    return trajectories

def recalculate_acceleration(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Recalculates the acceleration values and stores them in the ``acc`` column.
    :param trajectories: DataFrame with ``speed``, ``time``, ``location`` and ``ID`` columns.
    :return: DataFrame with recalculated values in the ``acc'' column.
    """
    acceleration = ((pl.col("speed") - pl.col("speed").shift(1, fill_value=np.nan))/
                    ((pl.col("time") - pl.col("time").shift(1)).dt.total_milliseconds() / 1000))
    trajectories = trajectories.sort("location", "ID", "time").with_columns(acceleration.over("location", "ID").alias("acc"))
    return trajectories