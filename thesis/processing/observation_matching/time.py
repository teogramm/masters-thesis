"""
Methods for matching trajectories based on time
"""
import datetime

import polars as pl

from thesis.processing.observation_matching.common import match_uncertain, correct_skewness, filter_period_observations


def fix_duplicates(matched_observations: pl.DataFrame, crossing_times: pl.DataFrame) -> pl.DataFrame:
    """
    Attempts to reassign trajectories that have been matched to more than one observation.
    :param matched_observations: Observations that have been matched to crossings
    :param crossing_times: Crossing times for all trajectories
    :return: Matched observations with limited duplicates.
    A few duplicate trajectory IDs can still occur.
    """
    # Remove non-null duplicate entries
    matched = matched_observations.filter((~pl.col("ID").is_duplicated()) | pl.col("ID").is_null())

    duplicates = matched_observations.filter(pl.col("ID").is_duplicated(),
                                             pl.col("ID").is_not_null())
    prev_dup = -1
    while duplicates.select(pl.len()).item() != prev_dup:
        prev_dup = duplicates.select(pl.len()).item()

        # Get the crossing times which have not been matched
        remaining_crossing_times = (crossing_times.join(matched, on="ID", how="anti").
                                    join(duplicates, on="ID", how="anti").sort(by="crossing_time"))

        # Calculate the minimum time distance for each pair of duplicates
        duplicates = duplicates.with_columns(time_dist=(pl.col("observation_time") - pl.col("crossing_time")).
                                             dt.total_milliseconds()).with_columns(
            min_time_dist=pl.col("time_dist").min().over(pl.col("ID")))

        # Get the duplicates that hold the closest distance and those that do not
        close_duplicates = duplicates.filter(pl.col("time_dist") == pl.col("min_time_dist"))
        far_duplicates = duplicates.filter(pl.col("time_dist") != pl.col("min_time_dist"))

        # Drop new columns
        close_duplicates = close_duplicates.drop("min_time_dist", "time_dist")
        far_duplicates = far_duplicates.drop("time_dist", "min_time_dist")

        # The close duplicates are correct, add them to the results
        matched = pl.concat([matched, close_duplicates])

        # Try to match the far duplicates
        # Drop columns given by the matches
        far_duplicates = far_duplicates.select([
            "observation_time", "direction", "primary_type", "secondary_type", "comments", "rental", "uncertain",
            "relative_position"
        ]).sort(by="observation_time")

        # Get some new matches for the remaining duplicates and add them to the results
        new_matches = far_duplicates.join_asof(remaining_crossing_times, left_on="observation_time",
                                               right_on="crossing_time",
                                               strategy='nearest',
                                               tolerance=datetime.timedelta(seconds=1, milliseconds=500),
                                               by="direction")
        matched = pl.concat([matched, new_matches]).sort(by="observation_time")
        # Calculate if there are any duplicates now
        duplicates = matched.filter(pl.col("ID").is_duplicated(),
                                    pl.col("ID").is_not_null())
        matched = matched.filter((~pl.col("ID").is_duplicated()) | pl.col("ID").is_null())
    return matched


def match_observations_asof(observations: pl.DataFrame, crossing_times: pl.DataFrame,
                            periods: list[tuple[datetime, datetime]]) -> pl.DataFrame:
    """
    Match the observations in the given periods to the crossing times using the built-in polars asof function.
    Keeps only observations in the given periods.
    :param periods: List of observation periods, used for filtering observations and calculating skewness.
    :return: All observations in the given periods with information about the matched trajectories.
    """
    
    observations = filter_period_observations(observations, periods)
    observations = correct_skewness(observations, crossing_times, periods)

    # At first match uncertain crossings using positional information
    matched_uncertain, _ = match_uncertain(crossing_times, observations)
    matched_ids = matched_uncertain.select(pl.col("ID")).to_series()

    # Match the rest of the crossing times using just the times
    crossing_times = crossing_times.filter(~pl.col("ID").is_in(matched_ids))

    # Join observations to the closest crossing trajectory at the same direction
    matches = observations.join_asof(crossing_times, left_on="observation_time", right_on="crossing_time",
                                     strategy='nearest',
                                     tolerance=datetime.timedelta(seconds=1), by="direction")    

    # Match column order in both DataFrames and concatenate
    matched_uncertain = matched_uncertain.select(matches.columns)
    all_matched = pl.concat([matched_uncertain, matches])

    all_matched = fix_duplicates(all_matched, crossing_times)

    return all_matched
