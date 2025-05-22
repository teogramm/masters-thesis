"""
Opens CSV files and sets the correct data types
"""
from typing import Optional

import polars as pl
from thesis.model.enums import Location
from thesis.files import BASE_PATH

TRAJECTORIES_LOCATION = BASE_PATH.joinpath("data/trajectories/source.parquet")

def open_trajectories(location: Optional[Location] = None,
                      with_pedestrians: bool = False) -> pl.DataFrame:
    trajectories = pl.scan_parquet(TRAJECTORIES_LOCATION)
    if location is not None:
        trajectories = trajectories.filter(pl.col("location") == location)
    if not with_pedestrians:
        trajectories = trajectories.filter(pl.col("type") != "Pedestrian")
    return trajectories.collect()


def open_all(with_pedestrians: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Returns three DataFrames with the trajectories at each location.
    :return: Riddarhuskajen, Riddarhusbron_north, Riddarhusbron_south
    """
    trajectories = pl.scan_parquet(TRAJECTORIES_LOCATION)
    if not with_pedestrians:
        trajectories = trajectories.filter(pl.col("type") != "Pedestrian")
    trajectories = trajectories.collect()
    return (trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN),
            trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N),
            trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_S))