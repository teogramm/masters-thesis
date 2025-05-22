import subprocess

import polars as pl

from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

from thesis.files import BASE_PATH
from thesis.files.observations import OBSERVATIONS_PROCESSED_FILE_PATH
from thesis.model.enums import Direction, Primary_Type, Secondary_Type, Relative_Position

OBSERVATIONS_ORIGINAL_FILE_PATH = BASE_PATH.joinpath("data/observations/observations.txt")
OBSERVATIONS_CLEANED_FILE_PATH = BASE_PATH.joinpath("data/observations/observations-clean.txt")


def preprocess_observations():
    """
    Processes the files given by the observer.
    :return: 
    """

    # We need to convert the file of the observer from UTF-16 to UTF-8
    if OBSERVATIONS_CLEANED_FILE_PATH.exists():
        print("Observations file with correct encoding already exists. Not overwriting.")
    else:
        subprocess.run(["iconv", "-f", "UTF-16LE", "-t", "UTF8", "-o", str(OBSERVATIONS_CLEANED_FILE_PATH),
                        str(OBSERVATIONS_ORIGINAL_FILE_PATH)])
    
    OBSERVATIONS_PROCESSED_FILE_PATH.parent.mkdir(exist_ok=True)
    observations = open_original_observations()
    observations.write_parquet(OBSERVATIONS_PROCESSED_FILE_PATH)

def _get_video_start_time(day: int, period: str) -> datetime:
    tz = ZoneInfo("Europe/Stockholm")
    if period == "PM":
        return datetime(2024, 10, day, 16, 0, tzinfo=tz)
    elif period == "AM":
        if day == 1:
            return datetime(2024, 10, 1, 6, 45, tzinfo=tz)
        elif day == 2:
            return datetime(2024, 10, 2, 6, 46, tzinfo=tz)
        elif day == 3:
            return datetime(2024, 10, 3, 6, 46, tzinfo=tz)
    elif period == "OP":
        if day == 1:
            return datetime(2024, 10, 1, 11, 00, tzinfo=tz)
        else:
            return datetime(2024, 10, day, 12, 00, tzinfo=tz)
    raise ValueError(f"Unknown period {period}")


def _get_observation_start_time(day: str, period: str, start_time: str) -> datetime:
    # Get the start datetime of the video
    start_datetime = _get_video_start_time(int(day), period)
    # If the video is part two, we need to add the time of part 1
    _part_1_length = timedelta(hours=1, minutes=32, seconds=2, milliseconds=520)
    if int(start_time) in (8, 17):
        start_datetime += _part_1_length
    return start_datetime


def _get_period_start_times(observations: pl.DataFrame) -> pl.Series:
    """
    Returns a series with the start time of the period when the observation was made.
    :param observations: 
    :return: Series containing a ```datetime``` object for the period start time of the corresponding observation.
    """
    # Create a dictionary mapping each Observation name (from Observer) to a start datetime
    # Get all unique Observation values
    # available_periods = observations.select(pl.col("Observation").unique())
    available_periods = observations.select(pl.col("Observation").unique().alias("original_name")).with_columns(
        period_info=pl.col("original_name").
        str.
        extract_groups(r"(0[1-3])/10 ([PA]M|OP) .+ ([0-9]+).*").
        struct.rename_fields(["day", "period", "start_time"]))
    # Extract the necessary information for each period
    # Convert the dataframe to a 1-column dataframe with a struct containing
    # original_name (from observer), day, period, start_time for each unique observation value
    available_periods = available_periods.unnest("period_info").to_struct()
    # Create a mapping for Observation to a start_datetime
    start_time_mapping = {s["original_name"]: _get_observation_start_time(s["day"], s["period"], s["start_time"]) for s
                          in available_periods}
    # Get the start time for each observation
    start_times = observations.select(pl.col("Observation").replace_strict(start_time_mapping)).to_series()
    return start_times


def _get_deltas(observations: pl.DataFrame) -> pl.Series:
    """
    Calculate the time relative to the start time of the period.
    :param observations: 
    :return: Series with ```timedelta``` objects.
    """
    # Parse columns containing time values
    deltas = observations.select(pl.col("Time_Relative_hmsf").str.strptime(pl.Time, "%H:%M:%S%.f"))
    deltas = deltas.with_columns(pl.col("Time_Relative_hmsf").map_elements(
        lambda x: timedelta(hours=x.hour, minutes=x.minute, seconds=x.second, milliseconds=x.microsecond // 1000),
        return_dtype=timedelta)).to_series()
    return deltas


def _add_absolute_dates(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Add the ```abs_time``` column, containing the absolute time of the observation.
    :param observations: 
    :return: Dataframe with ```abs_time``` column, containing ```datetime``` objects..
    """
    start_times = _get_period_start_times(observations)
    deltas = _get_deltas(observations)
    # Integrate the start times with the delta
    observations = observations.with_columns(abs_time=(start_times + deltas))
    return observations


def _add_direction(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a ```direction``` column with values from :py:class:`model.enums.Direction_Enum`.
    
    For off-peak observations, the ```Comments``` column is used. It should be a list, containing "N" or "S" depending
    on the direction of the observation.
    :param observations: 
    :return: Dataframe including ```direction``` column.
    """
    observations = observations.with_columns(direction=pl.
                                             when(pl.col("Observation").str.contains("NB")).
                                             then(pl.lit("Northbound")).
                                             when(pl.col("Observation").str.contains("SB")).
                                             then(pl.lit("Southbound")).
                                             when(pl.col("Observation").str.contains("BD"),
                                                  pl.col("Comment").list.contains("N")).
                                             then(pl.lit("Northbound")).
                                             when(pl.col("Observation").str.contains("BD"),
                                                  pl.col("Comment").list.contains("S")).
                                             then(pl.lit("Southbound")).
                                             cast(Direction))
    # Remove N and S from the comment column
    observations = observations.with_columns(pl.col("Comment").list.set_difference(["N", "S"]))
    return observations


def _parse_bike_types(observations: pl.DataFrame) -> pl.DataFrame:
    """
    Converts the ``Subject`` and ``Behavior`` columns into primary and secondary type columns.
    Drops any rows without a Subject defined.
    :param observations: 
    :return: Dataframe with the Subject and Behavior columns replaced by ``primary_type`` and ``secondary_type``.
    """
    # Drop observations with no Subject as errors
    observations = observations.filter(pl.col("Subject").str.len_chars() > 0)
    # Primary type is always specified, secondary type is optional
    observations = observations.with_columns(pl.col("Subject").cast(Primary_Type))
    observations = observations.with_columns(pl.col("Behavior").cast(Secondary_Type, strict=False))
    observations = observations.rename({
        "Subject": "primary_type",
        "Behavior": "secondary_type"
    })
    return observations


def _parse_rental(observations: pl.DataFrame) -> pl.DataFrame:
    # Scooters are rental by default, electric are rental if specified, otherwise false
    predicate = (pl
                 .when(pl.col("primary_type") == "Electric",
                       pl.col("Comment").list.contains("Rental"))
                 .then(True)
                 .when(pl.col("primary_type") == "Scooter",
                       ~pl.col("Comment").list.contains("Private"))
                 .then(True)
                 .otherwise(False))
    # observations = observations.with_columns(rental=pl.lit(False))
    observations = observations.with_columns(rental=predicate)
    # Remove the relevant values from comments
    observations = observations.with_columns(pl.col("Comment").list.set_difference(["Rental", "Private"]))
    return observations


def _parse_uncertain(observations: pl.DataFrame) -> pl.DataFrame:
    observations = observations.with_columns(pl.col("Comment").list.contains("U").alias("uncertain"))
    observations = observations.with_columns(pl.col("Comment").list.set_difference(["U"]))
    return observations


def _parse_position(observations: pl.DataFrame) -> pl.DataFrame:
    predicate = ((pl.when(pl.col("Comment").list.contains("Front")).then(pl.lit("Front"))
                  .when(pl.col("Comment").list.contains("Back")).then(pl.lit("Back")))
                 .otherwise(None).cast(Relative_Position, strict=False))
    observations = observations.with_columns(relative_position=predicate)
    # Remove relative comments
    observations = observations.with_columns(pl.col("Comment").list.set_difference(["Front", "Back"]))
    return observations


def _parse_carrying(observations: pl.DataFrame) -> pl.DataFrame:
    predicate = pl.when(pl.col("Comment").list.contains("Carrying")).then(pl.lit(True)).otherwise(False)
    observations = observations.with_columns(carrying_something=predicate)
    # Remove relative comments
    observations = observations.with_columns(
        pl.col("Comment").list.set_difference(["Carrying", "bag", "bags", "tyres"]))
    return observations


def _parse_comments(observations: pl.DataFrame) -> pl.DataFrame:
    # Parse comments that we can
    observations = _parse_rental(observations)
    observations = _parse_uncertain(observations)
    observations = _parse_position(observations)
    observations = _parse_carrying(observations)
    # Store the remaining comments in another column
    observations = observations.with_columns(comments=pl.col("Comment").list.join(" ")).drop("Comment")
    return observations


def open_original_observations() -> pl.DataFrame:
    """
    Opens the file exported by the observer and:
    * Adds absolute date and time
    * Adds direction
    * Corrects data types
    * Parses comments
    :return: 
    """
    observations = pl.read_csv(OBSERVATIONS_CLEANED_FILE_PATH, separator=';')
    # Split comments
    observations = observations.with_columns(pl.col("Comment").str.split(" "))
    observations = _parse_bike_types(observations)
    observations = _add_direction(observations)
    observations = _add_absolute_dates(observations)
    observations = _parse_comments(observations)
    # Keep only useful columns
    observations = observations.select(["abs_time", "direction", "primary_type", "secondary_type", "comments",
                                        "rental", "uncertain", "relative_position"])
    observations = observations.sort("abs_time").rename({"abs_time": "observation_time"})
    return observations
