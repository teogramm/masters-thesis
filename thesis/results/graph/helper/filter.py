import sys

import polars
import polars as pl

from thesis.model.enums import Location
from thesis.results.graph.helper.columns import YColType


def filter_trajectories_linear(trajectories: pl.LazyFrame, y_col: YColType | str) -> pl.LazyFrame:
    """
    Filter trajectories that will be displayed in a line graph, based on the given column.
    Can also be used to filter based on the given column, as no columns are removed from the DataFrame.
    :param trajectories: DataFrame containing the given ``y_col'' and any other columns required for filtering (vary
    depending on the column used for the filtering)
    :param y_col: If it is ``speed'', ``acc'' or ``lat_pos'' filtering is applied, depending on this column.
    :return: The returned DataFrame might contain trajectories with only some datapoints removed.
    """
    if y_col == "speed":
        return _filter_speed(trajectories)
    elif y_col == "acc":
        return _filter_acc(trajectories)
    elif y_col == "lat_pos":
        return _filter_lat_pos(trajectories)
    else:
        print(f"Warning: Filtering for {y_col} not implemented.", file=sys.stderr)
        return trajectories

def filter_for_type(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Retains trajectories with a bike type and keeps only Regular, Electric and Race bicycles.
    :param trajectories: Dataframe with ``primary_type'' column.
    :return: LazyFrame with trajectories of Regular, Electric and Race bicycles.
    """
    trajectories = trajectories.filter(pl.col("primary_type").is_not_null(),
                                       pl.col("primary_type") != "Other")
    # rk_hidden = ((pl.col("location") == Location.RIDDARHUSKAJEN) &
    #              (pl.col("primary_type").is_in(["Race", "Electric", "Regular", "Cargo", "Moped"])))
    # trajectories = trajectories.filter(rk_hidden | (pl.col("location") != Location.RIDDARHUSKAJEN))
    shown_types = ["Regular", "Electric", "Race"]
    trajectories = trajectories.filter(pl.col("primary_type").is_in(shown_types))
    return trajectories
    
def _filter_speed(trajectories: pl.LazyFrame):
    """
    Filters to apply when examining speed.
    Keep speed values between 0.5 and 10 m/s, that are inside the path and not pedestrians. 
    :param trajectories: LazyFrame with ``long_pos'', ``in_path'', ``speed'' and ``f_ped'' columns.
    """
    min_speed = 0.5
    max_speed = 10.0
    trajectories = trajectories.filter(pl.col("long_pos") > 0,
                                          pl.col("in_path"),
                                          pl.col("speed").is_not_nan(),
                                          pl.col("speed").is_between(min_speed, max_speed),
                                            ~pl.col("f_ped"))
    return trajectories

def _filter_acc(trajectories: pl.LazyFrame):
    """
    Filters to apply when examining acceleration.
    Keep acceleration values up to 5 m/s^2 that remain inside the path in their entirety.
    In addition, keep speed values between 0.5 and 10 m/s.
    Keeps trajectories that are complete and remain inside the path, that are not pedestrians or swapping.
    :param trajectories: LazyFrame with ``long_pos'', ``speed'', ``acc'', ``in_path'', ``trip'', ``swap_minor'',
    ``swap_major'' and ``f_ped'' columns.
    """
    min_acc = 0
    max_acc = 5
    trajectories = trajectories.filter(pl.col("long_pos") > 0,
                                       pl.col("speed").is_between(0.5, 10, closed="none"),
                                       pl.col("in_path"),
                                       pl.col("acc").is_not_nan(),
                                       pl.col("acc").abs().is_between(min_acc, max_acc, closed="both"),
                                       pl.col("trip") == 0,
                                       ~pl.col("swap_minor"),
                                       ~pl.col("swap_major"),
                                       ~pl.col("f_ped"))
    return trajectories

def _filter_lat_pos(trajectories: pl.LazyFrame):
    """
    Filters to apply when examining lateral position.
    Keeps trajectories inside the path, that are not swapping or pedestrians.
    :param trajectories: DataFrame with ``long_pos'', ``in_path'', ``swap_minor'', ``swap_major'' and ``f_ped'' columns.
    """
    trajectories = trajectories.filter(pl.col("long_pos") > 0,
                                       pl.col("in_path"),
                                       ~pl.col("swap_minor"),
                                       ~pl.col("swap_major"),
                                       ~pl.col("f_ped"))
    return trajectories


def filter_long_pos(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Keeps only datapoints in the longitudinal positions which have enough data.
    :param trajectories: DataFrame with ``long_pos'' and ``location'' columns.
    :return: Trajectories with some datapoints removed.
    """
    long_pos_low = {
        Location.RIDDARHUSKAJEN: 8.0,
        Location.RIDDARHOLMSBRON_S: 2.0,
        Location.RIDDARHOLMSBRON_N: 1.5
    }
    long_pos_high = {
        Location.RIDDARHUSKAJEN: 35.0,
        Location.RIDDARHOLMSBRON_S: 17.5,
        Location.RIDDARHOLMSBRON_N: 16.5
    }
    in_section = ((pl.col("long_pos") >= pl.col("location").replace_strict(long_pos_low, return_dtype=pl.Float64)) &
                  (pl.col("long_pos") <= pl.col("location").replace_strict(long_pos_high, return_dtype=pl.Float64)))
    trajectories = trajectories.filter(in_section)
    return trajectories


def filter_meetings(meetings: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filters to apply when examining meetings.
    Discards all trajectories that have any sort of issue.
    """
    meetings = meetings.filter(~pl.col("excluded"),
                                   pl.col("width").is_not_null(),
                                   pl.col("in_path"))
    bad_trajectories = meetings.filter(pl.col("excluded"))
    meetings = meetings.join(bad_trajectories, left_on=["location", "meeting_id"], right_on=["location", "ID"],
                             how="anti")
    meetings = meetings.filter(pl.col("ID") < pl.col("meeting_id"))
    # meetings = meetings.filter(pl.col("meeting_lateral_dist") > 0.7)
    return meetings