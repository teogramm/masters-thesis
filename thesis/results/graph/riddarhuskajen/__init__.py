import polars as pl
import altair as alt

from thesis.filtering.filters import PathFilter
from thesis.model.enums import Location

alt.data_transformers.disable_max_rows()

def make_relative_to_apex(col_name: str) -> pl.Expr:
    """
    Make the given longitudinal position values relative to the apex of the turn, depending on the travel direction.
    :param col_name: Name of the column with the longitudinal position values.
    :return: Polars expression that makes the values given column relative to the apex location.
    Requires the ``direction`` and ``location`` columns.
    """
    return (pl.when(pl.col("direction") == "Northbound", pl.col("location") == Location.RIDDARHUSKAJEN)
            .then(pl.col(col_name) - 21.5)
            .when(pl.col("direction") == "Southbound", pl.col("location") == Location.RIDDARHUSKAJEN).then(
        21.5 - pl.col(col_name)
    ).otherwise(col_name).alias(col_name))


def _filter_crossing_location(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filters applied for examining corner cutting.
    Keeps trajectories that remain completely inside the path and do not overtake anyone.
    :param trajectories: DataFrame with ``path``, ``swap_minor``, ``swap_major`` and ``overtakes_id`` columns.
    """
    trajectories = trajectories.filter(pl.col("path") == PathFilter.INSIDE,
                                       pl.col("overtakes_id").list.len() == 0,
                                       ~pl.col("swap_minor"),
                                       ~pl.col("swap_major"))
    return trajectories


def crossing_location_scatter(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Create a scatter plot of maximum lateral position onto the opposite lane depending on the position of the
    crossing.
    :param trajectories: DataFrame with ``ID``, ``direction``, ``line_crossing_info`` and ``location`` columns.
    """
    trajectories = (trajectories.select("ID", "direction", "line_crossing_info", "location")
                    .explode("line_crossing_info").unnest("line_crossing_info")).unique()

    # trajectories = trajectories.with_columns(
    #     make_relative_to_apex("crossing_start"),
    #     make_relative_to_apex("crossing_end")
    # )
    trajectories = trajectories.group_by("direction", "crossing_start", "crossing_max_lat_pos").agg(
        pl.col("ID").n_unique().alias("n")
    )
    trajectories = trajectories.collect()

    chart = alt.Chart(trajectories).mark_circle().encode(
        alt.X("crossing_start:Q", title="Position crossing onto the opposite lane relative to apex (m)"),
        alt.Y("crossing_max_lat_pos:Q", title="Position of crossing back onto the correct lane relative to apex (m)"),
        alt.Color("n:Q", title="Number of trajectories")
    )

    return chart


def crossing_length_scatter(trajectories: pl.LazyFrame, unconstrained_only=False) -> alt.TopLevelMixin:
    """
    Create a scatter plot of the distance traversed into the opposite lane depending on the position of the crossing.
    :param unconstrained_only: If only unconstrained trajectories should be included.
    The ``constrained`` column should be ``Unconstrained``.
    """
    trajectories = _filter_crossing_location(trajectories)
    trajectories = (trajectories.select("ID", "direction", "line_crossing_info", "location" , "time_of_day",
                                        "constrained")
                    .explode("line_crossing_info").unnest("line_crossing_info")).unique()

    
    if unconstrained_only:
        trajectories = trajectories.filter(pl.col("constrained") == "Unconstrained")
    else:
        trajectories = trajectories.filter(pl.col("constrained") != "Unconstrained")
    
    trajectories = trajectories.filter(pl.col("direction") == "Southbound")
    trajectories = trajectories.with_columns(pl.col("time_of_day").replace({"between_peaks": "other"}))
    trajectories = trajectories.group_by("crossing_start", "crossing_dist").agg(
        pl.col("ID").n_unique().alias("n")
    )
    trajectories = trajectories.collect()

    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(trajectories).mark_circle().encode(
        alt.X("crossing_start:Q", title="Longitudinal position of crossing onto the opposite lane (m)"),
        alt.Y("crossing_dist:Q", title="Distance driven on the opposite lane (m)"),
        alt.Color("n:Q", title="Number of trajectories"),
    )

    return chart
