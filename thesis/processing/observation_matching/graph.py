"""
Methods for matching trajectories based on graphs
"""
import datetime

import numpy as np
import polars as pl

import thesis.files.crossing_times
import thesis.files.observations
from thesis.model.enums import Location
from thesis.model.observations_periods import observation_periods_day

from thesis.processing.observation_matching.common import match_uncertain, correct_skewness, fix_relative_position, \
    filter_period_observations


def graph_matching(observations: pl.DataFrame, crossings: pl.DataFrame,
                   threshold: pl.Duration) -> pl.DataFrame:
    """
    Matches observations to crossings by finding the minimum cost matching in a bipartite graph.
    :return: Includes all observations, including those not matched to any trajectory.
    """
    from scipy.optimize import linear_sum_assignment
    observations = observations.with_row_index("observation_id")

    # Keep only non-null crossings
    crossings = crossings.filter(pl.col("crossing_time").is_not_null()).with_row_index("crossing_id")

    n_crossings = crossings.select(pl.len()).item()
    n_observations = observations.select(pl.len()).item()

    def search(period: datetime.datetime):
        """
        Finds all observations within the threshold of the given period.
        :return: List of structs containing the observation id and time difference.
        """
        period_start = period - threshold
        period_end = period + threshold
        return observations.filter(pl.col("observation_time").is_between(period_start, period_end)).select(
            pl.struct("observation_id", (pl.col("observation_time") - period).alias("diff"))
        ).to_series().to_list()

    # Find all possible observations for each crossing
    crossings_with_observations = crossings.with_columns(
        matching_observations=pl.col("crossing_time").map_elements(
            lambda time: search(time), return_dtype=pl.List(pl.Struct({
                "observation_id": pl.Int64, "diff": pl.Duration
            }))
        )
    ).explode("matching_observations").unnest("matching_observations")
    #  Keep only the absolute value of millisecond difference with each observation
    crossings_with_observations = crossings_with_observations.with_columns(
        pl.col("diff").dt.total_milliseconds().cast(pl.Float64).abs()
    )
    # Each row now contains an edge with a weight
    # Create a matrix with n_crossings rows and n_observation columns
    graph = np.full((n_crossings, n_observations), np.iinfo(np.int64).max)
    # Create the edges on the graph
    for row in crossings_with_observations.iter_rows(named=True):
        crossing_id = row["crossing_id"]
        observation_id = row["observation_id"]
        weight = row["diff"]
        if observation_id is not None:
            graph[crossing_id, observation_id] = np.int64(weight)
    row_ind, col_ind = linear_sum_assignment(graph)
    # Remove any int_max edges that were used, as they are not part of the optimal solution.
    for i, (row, col) in enumerate(zip(row_ind, col_ind)):
        if graph[row, col] == np.iinfo(np.int64).max:
            col_ind[i] = -1
    # Create match dataframe, removing the maximum cost edges
    matches = pl.DataFrame({
        "crossing_id": row_ind,
        "observation_id": col_ind,
    }).filter(pl.col("observation_id") != -1)
    # Add information about crossings to observations
    matches = observations.join(matches, how="left", on="observation_id").join(crossings, how="left", on="crossing_id")
    matches = matches.drop("crossing_id", "observation_id", "direction_right")
    
    return matches


def _match_observations_graph(observations: pl.DataFrame, crossing_times: pl.DataFrame,
                              periods: list[tuple[datetime, datetime]],
                              threshold: pl.Duration = pl.duration(seconds=1, milliseconds=500)) -> pl.DataFrame:
    """
    Match the observations in the given periods to the crossing times using graphs.
    Keeps only observations in the given periods.
    :param periods: List of observation periods, used for filtering observations and calculating skewness
    :return: All observations in the given periods with information about the matched trajectories.
    """

    observations = filter_period_observations(observations, periods)
    observations = correct_skewness(observations, crossing_times, periods)

    # At first match uncertain crossings using positional information
    matched_uncertain, _ = match_uncertain(crossing_times, observations)
    matched_ids = matched_uncertain.select(pl.col("ID")).to_series()

    # Match the rest of the crossing times using just the times
    crossing_times = crossing_times.filter(~pl.col("ID").is_in(matched_ids))

    matches = []
    for period in periods:
        for direction in ("Northbound", "Southbound"):
            these_observations = observations.filter(pl.col("observation_time").is_between(period[0], period[1]),
                                                     pl.col("direction") == direction)
            these_crossings = crossing_times.filter(pl.col("crossing_time").is_between(period[0], period[1]),
                                                    pl.col("direction") == direction)
            matches.append(graph_matching(these_observations, these_crossings, threshold))
    
    matched_uncertain = matched_uncertain.select(matches[0].columns)
    matches.append(matched_uncertain)
    matches = pl.concat(matches)
    matches = matches.sort("observation_time")
    
    matches = fix_relative_position(matches)
    
    return matches


def calculate_match_riddarholmsbron_n_wed(threshold: pl.Duration = pl.duration(seconds=2)) -> pl.DataFrame:
    periods = observation_periods_day(3)

    crossing_times = thesis.files.crossing_times.open_crossing_times_riddarholmsbron_n()
    # Drop location column due to legacy code
    crossing_times = crossing_times.drop("location")

    observations = thesis.files.observations.open_processed_observations().sort("observation_time")
    observations = filter_period_observations(observations, periods)

    matched_observations = _match_observations_graph(observations, crossing_times, periods, threshold)
    matched_observations = matched_observations.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N))

    return matched_observations


def calculate_match_riddarholmsbron_n_tue(threshold: pl.Duration = pl.duration(seconds=2)) -> pl.DataFrame:
    periods = observation_periods_day(2)

    crossing_times = thesis.files.crossing_times.open_crossing_times_riddarholmsbron_n().sort("crossing_time")
    # Drop location column due to legacy code
    crossing_times = crossing_times.drop("location")
    
    observations = thesis.files.observations.open_processed_observations().sort("observation_time")


    matched_observations = _match_observations_graph(observations, crossing_times, periods, threshold)
    matched_observations = matched_observations.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N))

    return matched_observations


def calculate_match_riddarhuskajen(threshold: pl.Duration = pl.duration(seconds=2)) -> pl.DataFrame:
    crossing_times = thesis.files.crossing_times.open_crossing_times_riddarhuskajen()
    # Drop location column due to legacy code
    crossing_times = crossing_times.drop("location")

    periods = observation_periods_day(1)

    observations = thesis.files.observations.open_processed_observations().sort("observation_time")
    observations = filter_period_observations(observations, periods)

    matched_observations = _match_observations_graph(observations, crossing_times, periods, threshold)
    matched_observations = matched_observations.with_columns(location=pl.lit(Location.RIDDARHUSKAJEN))

    return matched_observations


def calculate_match_all(threshold: pl.Duration = pl.duration(seconds=2)) -> pl.DataFrame:
    """
    Matches all trajectories to observations
    :param threshold: 
    :return: Dataframe containing all observations along with their matched trajectories, if available
    """
    rk = calculate_match_riddarhuskajen(threshold)
    rb_tue = calculate_match_riddarholmsbron_n_tue(threshold)
    rb_wed = calculate_match_riddarholmsbron_n_wed(threshold)
    
    return pl.concat([rk, rb_tue, rb_wed]).sort("observation_time").with_columns(pl.col("location").cast(Location))
