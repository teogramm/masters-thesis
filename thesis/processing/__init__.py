import polars as pl

from thesis.files.crossing_times import _CROSSING_TIMES_FILE
from thesis.files.infrastructure import open_width, open_elevation
from thesis.files.observation_matching import add_matches
from thesis.files.processed import (create_results_per_datapoint, create_results_per_id_file,
                                    save_results_per_datapoint,
                                    save_results_per_id, add_results_per_id, add_results_per_datapoint)
from thesis.files.trajectories import open_trajectories
from thesis.model.enums import Location
from thesis.processing.infrastructure import calculate_width, calculate_elevation
from thesis.processing.interactions import calculate_following_meeting_ids
from thesis.processing.interactions.meeting import calculate_meeting_statistics
from thesis.processing.interactions.overtake import calculate_overtakes, calculate_overtake_info
from thesis.processing.interactions.following import calculate_following_parameters
from thesis.processing.riddarhuskajen import calculate_position_relative_to_apex, calculate_cuts_corner, \
    calculate_crossings_into_opposite_lane


def save_all_results() -> None:
    """
    Produces all the results of the processing
    :return: 
    """
    trajectories = open_trajectories().lazy()
    save_matches()
    save_all_results_per_datapoint(trajectories)
    save_all_results_per_trajectory(trajectories)
    # Add is the information we previously calculated to the dataframe
    trajectories = open_trajectories().lazy()
    trajectories = add_results_per_id(trajectories)
    trajectories = add_results_per_datapoint(trajectories)
    trajectories = add_matches(trajectories)
    # Collect and convert to lazy since we are going to change the result files
    trajectories = trajectories.collect().lazy()
    save_interaction_calculations(trajectories)


def save_matches() -> None:
    from thesis.processing.observation_matching.graph import calculate_match_all
    from thesis.files.observation_matching import _MATCHES_FILE
    save_crossing_times()
    matches = calculate_match_all()
    matches.write_parquet(_MATCHES_FILE)


def save_crossing_times() -> None:
    from thesis.processing.crossing_times import calculate_crossing_times_riddarhuskajen, \
        calculate_crossing_times_riddarhusbron_n
    rk = calculate_crossing_times_riddarhuskajen().with_columns(location=pl.lit(Location.RIDDARHUSKAJEN).cast(Location))
    rb_n = calculate_crossing_times_riddarhusbron_n().with_columns(
        location=pl.lit(Location.RIDDARHOLMSBRON_N).cast(Location))
    rk.vstack(rb_n, in_place=True)
    # Sort by location and crossing time
    rk = rk.sort(by=["location", "crossing_time", "ID"])
    rk.write_parquet(_CROSSING_TIMES_FILE)


def save_all_results_per_datapoint(trajectories: pl.LazyFrame) -> None:
    index_cols = ["location", "ID", "time"]
    create_results_per_datapoint()

    result_columns = []

    trajectories = calculate_position_relative_to_apex(trajectories)
    result_columns.append("relative_apex_pos")

    width = calculate_width(trajectories, open_width())
    trajectories = trajectories.join(width, on=index_cols, how="left", validate="1:1")
    result_columns.append("width")
    elevation = calculate_elevation(trajectories, open_elevation())
    trajectories = trajectories.join(elevation, on=index_cols, how="left", validate="1:1")
    result_columns.append("elevation")

    # Keep only result and index columns for joining into the results dataframe
    trajectories = trajectories.select(result_columns + index_cols)

    trajectories = trajectories.collect()
    for result_col in result_columns:
        save_results_per_datapoint(trajectories.select(*index_cols, result_col))


def save_all_results_per_trajectory(trajectories: pl.LazyFrame) -> None:
    create_results_per_id_file()
    index_cols = ["location", "ID"]

    # Calculate interactions
    fol_meetings = calculate_following_meeting_ids(trajectories).collect()
    save_results_per_id(fol_meetings.select(*index_cols, "meeting_ids"))
    save_results_per_id(fol_meetings.select(*index_cols, "following_ids"))


    overtakes = calculate_overtakes(trajectories).collect()
    save_results_per_id(overtakes.select(*index_cols, "overtakes_id"))

    cuts_corner = calculate_cuts_corner(trajectories).collect()
    save_results_per_id(cuts_corner.select(*index_cols, "cuts_corner"))

    line_crossing_info = calculate_crossings_into_opposite_lane(trajectories).collect()
    save_results_per_id(line_crossing_info.select([*index_cols, "line_crossing_info"]))


def save_interaction_calculations(trajectories: pl.LazyFrame) -> None:
    """
    Some interaction calculations require creating the results per trajectory but save results per datapoint, so create
    a function just for them.
    :return: 
    """

    following = calculate_following_parameters(trajectories).collect()
    save_results_per_datapoint(following)
    meeting_info = calculate_meeting_statistics(trajectories).collect()
    save_results_per_id(meeting_info.select("location", "ID", "meeting_info"))
    overtake_info = calculate_overtake_info(trajectories).collect()
    save_results_per_id(overtake_info.select("location", "ID", "overtake_info"))