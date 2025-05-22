import numpy as np
import polars as pl
from numba import guvectorize, float64

import thesis.files.crossing_times
import thesis.files.trajectories
from thesis.model.enums import Location
from thesis.model.exprs import is_off_peak


@guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:])], "(n),(n),(m),(m)->()")
def line_intersection_index(x, y, l0, l1, res) -> None:
    """
    Given a sequence of (X,Y) pairs calculate the intersection point with the given line.
    When mapping returns a column of one-element arrays.
    
    :param x: List of X coordinates of the points.
    :param y: List of Y coordinates of the points.
    :param l0: 2-element array with (X,Y) coordinates of the first point of the  line.
    :param l1: 2-element array with (X,Y) coordinates of the second point of the  line.
    :param res: See return
    :return : One-element array with the index of the point before the intersection (intersection occurs between this 
    point and the next). -1 if no intersection occurs.
    """
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    s2 = (l1[0] - l0[0], l1[1] - l0[1])
    for i in range(len(x)-1):
        s1 = (x[i + 1] - x[i + 0], y[i + 1] - y[i + 0])

        if (-s2[0] * s1[1] + s1[0] * s2[1]) == 0:
            continue
        s = (-s1[1] * (x[i + 0] - l0[0]) + s1[0] * (y[i + 0] - l0[1])) / (-s2[0] * s1[1] + s1[0] * s2[1])
        t = (s2[0] * (y[i + 0] - l0[1]) - s2[1] * (x[i + 0] - l0[0])) / (-s2[0] * s1[1] + s1[0] * s2[1])

        if 0 <= s <= 1 and 0 <= t <= 1:
            # res[0] = x[i+0] + (t*s1[0])
            # res[1] = y[i+0] + (t*s1[1])
            res[0] = i
            return
    res[0] = -1
    # res[1] = np.nan


def calculate_crossing_times(trajectories: pl.DataFrame, l0: tuple[float, float],
                             l1: tuple[float, float]) -> pl.DataFrame:
    """
    Get the time each trajectory corsses the line given by l0, l1
    :param trajectories: 
    :param l0: (X,Y) coordinates of one edge of the line
    :param l1: (X,Y) coordinates of the other edge of the line
    :return: Dataframe with a row for each ID, crossing_time combination,
    """
    # Keep the time column to get the crossing time after finding the index
    # Group trajectories by ID and find the index of the point before the crossing
    crossing_times = (trajectories.group_by("ID").
    agg(
        # Points from the same trajectory should have the same direction
        pl.col("time"), pl.col("direction").first(),
        # Keep all dist_right as well as X,Y values for every point of the trajectory with the given ID
        pl.when(pl.col("direction") == "Southbound").then("dist_left").otherwise("dist_right").alias("dist_east"),
        pl.struct("X", "Y").
        # Find the index of the point just before the intersection
        map_batches(
            lambda x: line_intersection_index(x.struct.field("X"), x.struct.field("Y"), np.array(l0), np.array(l1)),
            return_dtype=pl.Float64)
        .alias("intersection_index"))
    # Convert the 1-element list to a simple value
    .with_columns(
        pl.col("intersection_index").explode().cast(pl.Int64)
    ))
    # Get the time corresponding to each crossing
    crossing_times = crossing_times.with_columns(
        pl.when(pl.col("intersection_index") > -1)
        .then(pl.struct(
            crossing_time=pl.col("time").list.get(pl.col("intersection_index")),
            dist_east=pl.col("dist_east").list.get(pl.col("intersection_index"))
        ))
        .otherwise(None).struct.unnest()).drop(["time", "intersection_index"])
    # Reorder columns and sort by time
    crossing_times = crossing_times.sort(by="crossing_time").select(["ID", "direction", "crossing_time", "dist_east"])
    return crossing_times


def calculate_crossing_times_riddarhuskajen() -> pl.DataFrame:
    line_points_nb = (
        (-350 / 477, 7610 / 477),
        (620 / 477, 2290 / 159)
    )

    line_points_sb_op = (
        (-620 / 477, 8230 / 477),
        (1100 / 477, 750 / 53)
    )

    trajectories = thesis.files.trajectories.open_trajectories(Location.RIDDARHUSKAJEN)

    uses_northbound_line = (pl.col("direction") == "Northbound").or_(is_off_peak("time"))

    northbound = trajectories.filter(uses_northbound_line)
    crossings_northbound = calculate_crossing_times(northbound, line_points_nb[0], line_points_nb[1])

    southbound = trajectories.filter(~uses_northbound_line)
    crossings_southbound = calculate_crossing_times(southbound, line_points_sb_op[0], line_points_sb_op[1])

    crossing_times = pl.concat([crossings_northbound, crossings_southbound])
    
    return crossing_times


def calculate_crossing_times_riddarhusbron_n() -> pl.DataFrame:
    line_points = (
        (-5900 / 941, -530 / 941),
        (-1950 / 941, -3730 / 941)
    )
    trajectories = thesis.files.trajectories.open_trajectories(Location.RIDDARHOLMSBRON_N)
    crossing_times = calculate_crossing_times(trajectories, line_points[0], line_points[1])
    return crossing_times