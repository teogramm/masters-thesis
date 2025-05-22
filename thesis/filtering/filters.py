from enum import Enum
from typing import Iterable, Union
import polars as pl

from thesis.files.filtering import SUMMARY_FILE


class PathFilter(int, Enum):
    INSIDE = 0
    """All points of trajectory are inside the path"""
    PARTIALLY = 1
    """The trajectory contains points both outside and inside the path"""
    OUTSIDE = 2
    """All points of trajectory are outside the path"""

    def expr(self) -> pl.Expr:
        return pl.col("path") == self.value
        

class TripFilter(int, Enum):
    COMPLETE = 0
    """Trajectory enters and exits through the path"""
    COMPLETE_ABNORMAL_EXIT = 1
    """Trajectory enters the path but exits outside the path"""
    INCOMPLETE = 2
    """Trajectory does not enter through the path"""
    OUTSIDE = 3
    """Trajectory is completely outside the path"""

    def expr(self) -> pl.Expr:
        return pl.col("trip") == self.value

Filter = Union[PathFilter, TripFilter]

def _do_filter(trajectories: pl.LazyFrame, summary: pl.LazyFrame, filters: Iterable[Filter]) -> pl.DataFrame:
    """
    Filters the trajectories using information in the summary dataframe, keeping the ones that match the given filters.
    :param trajectories: Trajectory data
    :param summary: Information for each trajectory ID
    :param filters: Collection of filters.
    Trajectories matching these will be kept in the results.
    :return: Dataframe containing trajectory data matching the given filters.
    """
    # Keep original columns so we can return only them
    original_columns = trajectories.columns
    
    common_columns = set(original_columns).intersection(summary.columns)
    if len(common_columns) > 0:
        print(f"Warning: columns {common_columns} appear in both trajectories and summary. This can cause problems"
              f"during filtering.")
    
    trajectories = trajectories.join(summary, on=["ID", "location"], how="left")
    for condition in filters:
        trajectories = trajectories.filter(condition.expr())
    # Keep original columns and rename back
    trajectories = trajectories.select(original_columns).collect()
    return trajectories

def apply_filters(trajectories: pl.DataFrame, filters: Iterable[Filter]) -> pl.DataFrame:
    summary = pl.scan_csv(SUMMARY_FILE)
    trajectories = _do_filter(trajectories.lazy(), summary, filters)
    
    return trajectories