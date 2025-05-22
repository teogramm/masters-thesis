"""
Common methods used for observation matching
"""
import datetime
from datetime import timedelta

import polars as pl

from thesis.model.enums import Relative_Position


def match_uncertain(crossing_times: pl.DataFrame, observations: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Match trajectories which have exactly the same crossing time, using their relative position, if available.
    :param crossing_times: Unfiltered crossing times
    :param observations: Unfiltered observations
    :return: 2 element tuple containing the uncertain trajectories that were matched and the uncertain trajectories
             that were not matched, respectively. 
    """
    # Find trajectories for each direction, crossing time combination
    # Save the IDs and the dist_east values of the trajectories into a struct and also the maximum distance
    uncertain = (
        crossing_times
        # Find crossings with the same direction and crossing time
        .group_by(
            pl.col("direction"), pl.col("crossing_time"))
        # For each group make a list with all the IDs and distances that have the same direction and crossing time
        # and keep the observation with the longest distance from the right of the path
        .agg(
            pl.struct("ID", "dist_east").alias("matches"),
            pl.col("dist_east").max().alias("max_dist_east")
        )
        # Keep only trajectories that have more than one matching crossing
        .filter(pl.col("matches").list.len() > 1, pl.col("crossing_time").is_not_null())
        # Create one row for each ID, dist_east combination
        .explode("matches")
        .unnest("matches"))
    # Based on which observation has the maximum dist_east, set its relative position
    uncertain = uncertain.with_columns(relative_position=pl.when(pl.col("dist_east") == pl.col("max_dist_east"))
                                       .then(pl.lit("Back"))
                                       .otherwise(pl.lit("Front").cast(Relative_Position)))
    # Drop unnecessary column, as it was used to find the relative position
    uncertain = uncertain.drop("max_dist_east")
    uncertain = uncertain.sort("crossing_time")
    # Join the uncertain trajectories into the observations by ensuring the relative position matches
    uncertain = uncertain.join_asof(observations,
                                    left_on="crossing_time", right_on="observation_time",
                                    by=["direction", "relative_position"],
                                    tolerance=datetime.timedelta(seconds=1))
    # Return the crossings which were matched and the ones that were not matched
    return (uncertain.filter(pl.col("observation_time").is_not_null()),
            uncertain.filter(pl.col("observation_time").is_null()))


def find_skewness_factor(sorted_observations: pl.DataFrame, sorted_crossing_points: pl.DataFrame) -> float:
    """
    Calculates a skewness factor based on the corresponding observations and crossing points by taking the average
    difference between the crossing and observation times.
    :param sorted_observations: Observations sorted by observation_time
    :param sorted_crossing_points: Observations sorted by crossing_time
    :return: Factor to add to observations so they are closer to the crossing times
    """
    # factor = mean (crossing_time - observation_time)
    if (sorted_observations.select(pl.col("observation_time").len()).item()
            != sorted_crossing_points.select(pl.col("crossing_time").len()).item()):
        raise ValueError("The number of crossing points does not match the number of observations")
    # Keep only the time from the datetime objects
    sorted_observations = sorted_observations.with_columns(pl.col("observation_time").dt.time())
    crossing_points = sorted_crossing_points.sort("crossing_time").with_columns(pl.col("crossing_time").dt.time())
    factor = crossing_points.select(pl.col("crossing_time")).to_series() - sorted_observations.select(
        pl.col("observation_time")).to_series()
    # Keep only the absolute difference
    factor = factor.filter(
        factor.map_elements(lambda x: abs(x.total_seconds()), return_dtype=pl.Int64) < 10
    )
    # if factor.max().dt.total_seconds() > 5:
    #     print("WARNING: skewness calculated with a factor of more than 5 seconds, this usually indicates an"
    #           "error in the provided sampels")
    return factor.mean()


def correct_skewness(observations: pl.DataFrame, crossing_points: pl.DataFrame,
                     periods: list[tuple[datetime.datetime, datetime.datetime]], samples_start=5,
                     samples_end=5) -> pl.DataFrame:
    """
    Fixes skew in observation times, so they match the crossing times for each period and direction combination.
    
    Normally, a skewness factor is calculated using the given number of crossing and observation samples from the
    start and the end of each observation period and direction combination.
    It is important that the sampled observation and crossing times correspond to the same bike. 
    
    In certain cases, different operations are applied:
    
    For certain manually specified periods and directions, custom samples are used, since there are errors in the
    first or last crossings of the period-direction combination. 
    
    For certain manually specified periods and directions, linear compensation is applied instead, to correct clock
    drift during observations for that period-direction combination.

    
    :param samples_end: How many observations to use from the end of the period.
    :param samples_start: How many observations to use from the start of the period.
    :param observations: Observations sorted by ``observation_time``.
    :param crossing_points: Crossings sorted by ``crossing_time``.
    :param periods: List of start and end datetimes for periods.
    :return: ``observations`` dataframe with skewness factors applied and sorted by ``observation_time``.
    """

    manual_matches = {
        1: {
            6: {
                "Southbound": [
                    # Observation, Crossing
                    # Start
                    ("2024-10-01T06:45:32.565000+0200", "2024-10-01T06:45:33.640000+0200"),
                    ("2024-10-01T06:45:50.717000+0200", "2024-10-01T06:45:51.880000+0200"),
                    ("2024-10-01T06:46:39.232000+0200", "2024-10-01T06:46:40.360000+0200"),
                    ("2024-10-01T06:48:27.807000+0200", "2024-10-01T06:48:28.840000+0200"),
                    ("2024-10-01T06:49:26.031000+0200", "2024-10-01T06:49:27.000000+0200"),
                    # End
                    ("2024-10-01T09:02:38.047000+0200", "2024-10-01T09:02:39.000000+0200"),
                    ("2024-10-01T09:02:39.148000+0200", "2024-10-01T09:02:40.120000+0200"),
                    ("2024-10-01T09:02:50.693000+0200", "2024-10-01T09:02:51.720000+0200"),
                    ("2024-10-01T09:03:36.405000+0200", "2024-10-01T09:03:37.400000+0200"),
                    ("2024-10-01T09:04:43.272000+0200", "2024-10-01T09:04:44.280000+0200")
                ]
            },
            16: {
                "Southbound": [
                    # Observation, Crossing
                    # Start
                    ("2024-10-01T16:00:02.502000+0200", "2024-10-01T16:00:05.320000+0200"),
                    ("2024-10-01T16:00:06.072000+0200", "2024-10-01T16:00:08.920000+0200"),
                    ("2024-10-01T16:00:33.533000+0200", "2024-10-01T16:00:36.360000+0200"),
                    ("2024-10-01T16:00:34.467000+0200", "2024-10-01T16:00:37.240000+0200"),
                    ("2024-10-01T16:00:36.236000+0200", "2024-10-01T16:00:39.080000+0200"),
                    # End
                    ("2024-10-01T18:19:11.473000+0200", "2024-10-01T18:19:14.360000+0200"),
                    ("2024-10-01T18:19:16.945000+0200", "2024-10-01T18:19:19.800000+0200"),
                    ("2024-10-01T18:19:18.714000+0200", "2024-10-01T18:19:21.560000+0200"),
                    ("2024-10-01T18:19:39.201000+0200", "2024-10-01T18:19:42.040000+0200"),
                    ("2024-10-01T18:19:41.236000+0200", "2024-10-01T18:19:44.120000+0200")
                ]
            }
        },
        2: {
            16: {
                "Northbound": [
                    # Observation, Crossing
                    # Start
                    ("2024-10-02T16:00:16.916000+0200", "2024-10-02 16:00:18.520000+02:00"),
                    ("2024-10-02T16:00:48.748000+0200", "2024-10-02 16:00:50.360000+02:00"),
                    ("2024-10-02T16:00:49.916000+0200", "2024-10-02 16:00:51.480000+02:00"),
                    ("2024-10-02T16:00:54.187000+0200", "2024-10-02 16:00:55.800000+02:00"),
                    ("2024-10-02T16:01:33.793000+0200", "2024-10-02 16:01:35.320000+02:00"),
                    # End
                    ("2024-10-02T18:18:40.242000+0200", "2024-10-02 18:18:41.800000+02:00"),
                    ("2024-10-02T18:18:59.561000+0200", "2024-10-02 18:19:01.160000+02:00"),
                    ("2024-10-02T18:19:19.314000+0200", "2024-10-02 18:19:20.920000+02:00"),
                    ("2024-10-02T18:19:24.053000+0200", "2024-10-02 18:19:25.640000+02:00"),
                    ("2024-10-02T18:19:27.656000+0200", "2024-10-02 18:19:29.240000+02:00")
                ]
            }
        },
        3: {
            16: {
                "Northbound": [
                    # Observation, Crossing
                    # Start
                    ("2024-10-03T16:00:36.936000+0200", "2024-10-03T16:00:39.080000+0200"),
                    ("2024-10-03T16:00:44.711000+0200", "2024-10-03T16:00:46.840000+0200"),
                    ("2024-10-03T16:01:21.014000+0200", "2024-10-03 16:01:23.080000+02:00"),
                    ("2024-10-03T16:01:34.027000+0200", "2024-10-03 16:01:36.040000+02:00"),
                    ("2024-10-03T16:01:51.044000+0200", "2024-10-03T16:01:53.160000+0200"),
                    # End
                    ("2024-10-03T18:17:56.165000+0200", "2024-10-03 18:17:58.280000+02:00"),
                    ("2024-10-03T18:18:19.688000+0200", "2024-10-03 18:18:21.800000+02:00"),
                    ("2024-10-03T18:18:29.164000+0200", "2024-10-03 18:18:31.240000+02:00"),
                    ("2024-10-03T18:19:11.974000+0200", "2024-10-03 18:19:14.040000+02:00"),
                    ("2024-10-03T18:19:55.551000+0200", "2024-10-03 18:19:57.720000+02:00")
                ],
                "Southbound": [
                    # Observation, Crossing
                    # Start
                    ("2024-10-03T16:00:00.079000+0200", "2024-10-03T16:00:03.080000+0200"),
                    ("2024-10-03T16:00:12.512000+0200", "2024-10-03T16:00:15.560000+0200"),
                    ("2024-10-03T16:00:15.215000+0200", "2024-10-03T16:00:18.200000+0200"),
                    ("2024-10-03T16:00:16.382000+0200", "2024-10-03T16:00:19.480000+0200"),
                    ("2024-10-03T16:00:23.156000+0200", "2024-10-03T16:00:26.280000+0200"),
                    # End
                    ("2024-10-03T18:19:43.872000+0200", "2024-10-03T18:19:45.960000+0200"),
                    ("2024-10-03T18:19:44.540000+0200", "2024-10-03T18:19:46.680000+0200"),
                    ("2024-10-03T18:19:46.341000+0200", "2024-10-03T18:19:48.520000+0200"),
                    ("2024-10-03T18:19:47.309000+0200", "2024-10-03T18:19:49.480000+0200"),
                    ("2024-10-03T18:19:56.785000+0200", "2024-10-03T18:19:58.920000+0200")
                ]
            }
        }
    }

    # For which times to perform linear compensation
    linear_compensation = {
        # Day, Hour, Direction
        (2, 6, "Northbound"),
        (3, 6, "Northbound"),
        (3, 16, "Southbound")
    }

    # Apply the skewness factor for each period and direction combination
    for period in periods:
        for direction in ("Northbound", "Southbound"):
            # Keep only observations and crossings in this period

            end_adjustment = timedelta(seconds=2)

            observation_in_period = (pl.col("observation_time").is_between(period[0], period[1]) &
                                     (pl.col("direction") == direction))
            crossing_in_period = (pl.col("crossing_time").is_between(period[0], period[1] + end_adjustment) &
                                  (pl.col("direction") == direction))

            period_observations = None
            period_crossings = None

            # If the trajectories and observations have been manually entered
            # Jank
            if period[0].hour in manual_matches[period[0].day].keys():
                if direction in manual_matches[period[0].day][period[0].hour].keys():
                    pairs = manual_matches[period[0].day][period[0].hour][direction]
                    if len(pairs) != 0:
                        # Calculate using manual entries
                        period_observations = pl.DataFrame(
                            {"observation_time": [datetime.datetime.fromisoformat(p[0]) for p in pairs]})
                        period_crossings = pl.DataFrame({"crossing_time":
                                                             [datetime.datetime.fromisoformat(p[1]) for p in pairs]})
            # If no manual entries exist, take some samples
            if period_observations is None:
                period_observations = observations.filter(observation_in_period)
                period_crossing_points = crossing_points.filter(crossing_in_period)
                # Sample the first and last observations
                period_observations = pl.concat(
                    [period_observations.head(samples_start), period_observations.tail(samples_end)]
                )
                period_crossings = pl.concat(
                    [period_crossing_points.head(samples_start), period_crossing_points.tail(samples_end)]
                )

            if (period[0].day, period[0].hour, direction) in linear_compensation:
                observations = apply_linear_compensation(observations, crossing_points, period, direction,
                                                         sample_observations=period_observations,
                                                         sample_crossings=period_crossings)
            else:
                period_factor = find_skewness_factor(period_observations, period_crossings)

                # Apply the factor only to observations within the period
                observations = observations.with_columns(
                    pl.when(observation_in_period).
                    then(pl.col("observation_time") + period_factor).
                    otherwise("observation_time"))
    # Sort again by time
    return observations.sort("observation_time")


def apply_linear_compensation(observations: pl.DataFrame, crossings: pl.DataFrame,
                              period: tuple[datetime.datetime, datetime.datetime], direction,
                              sample_observations: pl.DataFrame = None,
                              sample_crossings: pl.DataFrame = None) -> pl.DataFrame:
    """
    Applies linear compensation to observation times in the given period and direction combination to account for
    clock drift.
    
    :param observations: DataFrame of all observations.
    :param crossings: Dataframe of all crossings.
    :param period: Observations within the period to be compensated.
    Tuple with start and end of the period, respectively. 
    :param direction: Observations will be compensated only for the given direction.
    :param sample_observations: 10 row Dataframe.
    The first 5 rows contain observations from the start, and the last 5 rows contain observations from the end of the
    period.
    :param sample_crossings: 10 row Dataframe.
    The first 5 rows contain crossings from the start, and the last 5 rows contain crossings from the end of the
    period.
    :return: ``observations`` dataframe with changes only to the times of observations specified by the period and
    direction.
    """
    observation_in_period = (pl.col("observation_time").is_between(period[0], period[1]) &
                             (pl.col("direction") == direction))
    crossing_in_period = (pl.col("crossing_time").is_between(period[0], period[1]) &
                          (pl.col("direction") == direction))

    # If no samples are provided, take some
    if sample_observations is None:
        sample_observations = pl.concat([observations.filter(observation_in_period).head(5),
                                         observations.filter(observation_in_period).tail(5)])
    if sample_crossings is None:
        sample_crossings = pl.concat([crossings.filter(crossing_in_period).head(5),
                                      crossings.filter(crossing_in_period).tail(5)])

    # Calculate the drift at the start
    factor_start = (sample_crossings.select(pl.col("crossing_time").head(5)).to_series() -
                    sample_observations.select(pl.col("observation_time").head(5)).to_series()).to_numpy().mean()

    # Add the factor to all observations within the period, so there is a common starting point
    observations = observations.with_columns(pl.when(observation_in_period)
                                             .then(pl.col("observation_time") + factor_start)
                                             .otherwise(pl.col("observation_time")))

    # Add the factor to the sample observations, to calculate the drift at the end of the period
    sample_observations = sample_observations.with_columns(pl.col("observation_time") + factor_start)

    # Calculate the drift at the end of the period
    offset_max = (sample_crossings.select(pl.col("crossing_time").tail(5)).to_series() -
                  sample_observations.select(pl.col("observation_time").tail(5)).to_series()).to_numpy().mean()

    period_total_seconds = (period[1] - period[0]).total_seconds()
    seconds_since_start = (pl.col("observation_time") - period[0]).dt.total_seconds().cast(pl.Float64)

    observations = observations.with_columns(
        pl.when(observation_in_period)
        .then(pl.col("observation_time") + seconds_since_start / period_total_seconds * offset_max)
        .otherwise("observation_time"))

    return observations


def fix_relative_position(matched_observations: pl.DataFrame) -> pl.DataFrame:
    """
    Fixes any matches by using the ``relative_position`` and ``dist_east`` columns. 
    :param matched_observations: Observations with matched trajectory information.
    :return: Corrected dataframe, unsorted
    """   
    # Add index id to know which rows to replace
    matched_observations = matched_observations.with_row_index()
    # Keep only rows that have relative position and a matched trajectory
    with_relative_position = matched_observations.filter(pl.col("relative_position").is_not_null())
    # In order to correct trajectories in overtakes when no matches have occurred, set a fake value for dist_east
    # 0.6 m when observation is in front and 1.6m when it is in the back.
    null_replacement = pl.when(pl.col("dist_east").is_null() & pl.col("relative_position").is_not_null()).then(pl.when(
        pl.col("relative_position") == "Front"
    ).then(pl.lit(0.6)).otherwise(pl.lit(1.6))).otherwise(pl.col("dist_east"))
    with_relative_position = with_relative_position.with_columns(dist_east=null_replacement)
    # Columns relative to the observation and crossings
    crossing_columns = ["ID", "dist_east", "crossing_time"]
    observation_columns = [col for col in with_relative_position.columns if col not in crossing_columns]
    # Copy the observation_time column to use it in the groupby
    with_relative_position = with_relative_position.with_columns(pl.col("observation_time").alias("observation_time_gb"),
                                                                 pl.col("direction").alias("direction_gb"))
    
    # Group in 1s intervals
    # Sort the dist_east and relative_position columns in each group independently, in the end they should match
    # to the actual positions, since front observations have low dist_east and back observations have high dist_east
    with_relative_position = (with_relative_position.group_by_dynamic(index_column="observation_time_gb",
                                                                      group_by="direction_gb",
                                                                     every="1s",
                                                                     start_by="datapoint")
    # First sort all the observation columns in the group by relative_position, so the observations on the front appear
    # at the top and the observations at the back apper at the bottom. Preserve order in case of equality.
    # Then sort the crossing columns depending on dist_east, so that the observations are aligned.
                              .agg(pl.col(observation_columns).sort_by("relative_position", maintain_order=True),
                                   pl.col(crossing_columns).sort_by("dist_east")).drop("observation_time_gb",
                                                                                        "direction_gb")
                              .explode(pl.all()))
    # Remove the fake dist_east values
    with_relative_position = with_relative_position.with_columns(dist_east=pl.when(pl.col("ID").is_null()).then(
        None
    ).otherwise(pl.col("dist_east")))
    # Integrate the changes back into the main dataframe
    matched_observations = matched_observations.update(with_relative_position, on="index",
                                                       how="left", include_nulls=True)
    return matched_observations.drop("index")


def filter_period_observations(observations: pl.DataFrame, periods: list[tuple[datetime, datetime]]) -> pl.DataFrame:
    """
    Keep only observations in any of the given periods.
    :param observations: List of observations.
    :param periods: List of tuples of (start_date, end_date)
    :return: DataFrame containing only observations falling within one or more of the given periods
    """
    predicate = None
    for period in periods:
        if predicate is None:
            predicate = pl.col("observation_time").is_between(period[0], period[1])
        else:
            predicate = predicate | pl.col("observation_time").is_between(period[0], period[1])
    observations = observations.filter(predicate)
    return observations
