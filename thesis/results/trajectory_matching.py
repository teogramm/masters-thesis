from thesis.files.crossing_times import open_crossing_times_riddarhuskajen, open_crossing_times_riddarholmsbron_n
from thesis.files.filtering import add_summary_information
from thesis.files.observation_matching import add_matches
from thesis.files.trajectories import open_trajectories
from thesis.model.exprs import time_of_day_column
from thesis.processing.observation_matching.graph import calculate_match_riddarholmsbron_n_wed, \
    calculate_match_riddarholmsbron_n_tue, \
    calculate_match_riddarhuskajen

from thesis.model.observations_periods import observation_periods_all, get_period_name, observation_periods_day
from thesis.model.enums import Location

from datetime import datetime

import polars as pl


def matches_by_time_of_day(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    trajectories = trajectories.with_columns(pl.col("time").dt.day().alias("day"),
                                             time_of_day_column("time"))
    
    trajectories = trajectories.filter(pl.col("ID").is_not_null(), pl.col("primary_type").is_not_null())
    trajectories = trajectories.group_by("day", "time_of_day").agg(
        pl.col("ID").n_unique().alias("n")
    )

    return trajectories.sort("day", "time_of_day")


def matches_by_time_of_day_and_type(location: Location) -> pl.LazyFrame:
    trajectories = open_trajectories(location).lazy()
    # Add summary information for time of day and match information
    trajectories = add_summary_information(trajectories)
    trajectories = add_matches(trajectories)

    # Keep only matched trajectories
    trajectories = trajectories.filter(pl.col("ID").is_not_null(), pl.col("primary_type").is_not_null())

    trajectories = trajectories.group_by(["time_of_day", "primary_type"]).agg(
        pl.col("ID").n_unique().alias("n")
    )

    return trajectories.sort("time_of_day", "primary_type")


def trajectories_by_period(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    trajectories = trajectories.with_columns(pl.col("time").dt.day().alias("day"),
                                             time_of_day_column("time"))
    
    statistics = trajectories.group_by("day", "time_of_day").agg(
        pl.col("ID").n_unique().alias("n")
    )

    return statistics.sort("day", "time_of_day")

def observations_by_period_and_type(observations: pl.LazyFrame) -> pl.LazyFrame:
    observations = observations.with_columns(pl.col("time").dt.day().alias("day"),
                                             time_of_day_column("observation_time"))
    
    observations = observations.group_by("day", "time_of_day", "primary_type").agg(
        pl.len().alias("n")
    )

    return observations.sort("day", "time_of_day","primary_type")


def observations_by_period(observations: pl.DataFrame) -> pl.DataFrame:
    observations = observations.with_columns(pl.col("time").dt.day().alias("day"),
                                             time_of_day_column("observation_time"))

    observations = observations.group_by("day", "time_of_day").agg(
        pl.len().alias("n")
    )

    return observations.sort("day", "time_of_day")


def total_crossings() -> pl.DataFrame:
    crossings = pl.concat([open_crossing_times_riddarhuskajen(), open_crossing_times_riddarholmsbron_n()])
    crossings = crossings.filter(pl.col("crossing_time").is_not_null())
    crossings = crossings.with_columns(pl.col("crossing_time").dt.day().alias("day"),
                                       time_of_day_column("crossing_time"))

    crossings = crossings.filter(((pl.col("location") == Location.RIDDARHUSKAJEN) & (pl.col("day") == 1)) |
                                 ((pl.col("location") == Location.RIDDARHOLMSBRON_N) & (pl.col("day").is_between(2,3))))
    
    crossings = crossings.group_by("location", "day", "time_of_day").agg(
        pl.col("ID").n_unique().alias("n")
    )
    
    # Keep only relevant stuff


    return crossings.sort("location", "day", "time_of_day")
