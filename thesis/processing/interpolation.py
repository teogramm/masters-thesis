from typing import Optional, Iterable

import polars as pl

def interpolate_for_long_pos(trajectories: pl.LazyFrame, col_interpolated: Iterable[str],
                             col_forward: Optional[Iterable[str]] = None) -> pl.LazyFrame:
    """
    Linearly interpolate values for the given columns for missing longitudinal position values.
    The ``time`` column is always linearly interpolated.
    The ``direction`` column is always kept.
    
    Longitudinal positions are grouped every 0.5 m. This function interpolates values for the given columns for
    longitudinal positions in the middle of the trajectory; it does not add new values outside the existing area of the
    trajectory.
    
    :param col_forward: Columns whose value will be filled by the next observation
    :param trajectories: 
    :param col_interpolated: Columns whose values will be interpolated.
    :return: DataFrame with each trajectory containing a ``long_pos`` entry every 0.5 m and the ``interpolated`` column.
    """
    
    if col_forward is None:
        col_forward = []
    if "direction" not in col_forward:
        col_forward.append("direction")
    
    # For each datapoint find the minimum and maximum long_pos
    trajectory_min_long_pos = pl.col("long_pos").min()
    trajectory_max_long_pos = pl.col("long_pos").max()
    # Create values every 0.5m between the trajectory's start and end
    interpolated = trajectories.group_by(["location", "ID"]).agg(
        pl.int_range(trajectory_min_long_pos*2, trajectory_max_long_pos*2 + 1).truediv(2).alias("long_pos")
    ).explode("long_pos")
    # Join values which exist
    interpolated = interpolated.join(trajectories, on=["location", "ID", "long_pos"], how="full")
    # Interpolate within each trajectory
    interpolated = interpolated.group_by(["location", "ID"]).agg(
        pl.col("time").sort_by("long_pos").is_null().alias("interpolated"),
        pl.col(*col_interpolated).sort_by("long_pos").interpolate(),
        pl.col("time").sort_by("long_pos").interpolate(),
        pl.col("long_pos").sort(),
        pl.col(*col_forward).sort_by("long_pos").fill_null(strategy="forward")
    ).explode("time", "long_pos", "interpolated", *col_forward, *col_interpolated)    
    return interpolated

def deduplicate_by_long_pos(trajectories: pl.LazyFrame, deduplicate_mean: list[str],
                            static_cols: Iterable[str] = None) -> pl.LazyFrame:
    """
    Remove duplicate datapoints for the same ID at the same longitudinal position by taking the mean of the given
    columns.
    
    The ``time`` and ``direction`` columns are always included.
    :param static_cols: Columns that their value is filled by taking any value for the trajectory in the specific
    longitudinal position.
    :param trajectories: 
    :param deduplicate_mean: Columns to keep
    :return: 
    """

    if static_cols is None:
        static_cols = []
    if "direction" not in static_cols:
        static_cols.append("direction")
        
    if "time" not in deduplicate_mean:
        deduplicate_mean.append("time")
    
    trajectories = trajectories.group_by(["location", "ID", "long_pos"]).agg(
        pl.col(*deduplicate_mean).mean(),
        pl.col(*static_cols).first()
    )
    
    return trajectories