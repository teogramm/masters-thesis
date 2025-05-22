import polars as pl
import altair as alt

from thesis.files.infrastructure import open_elevation, open_curvature
from thesis.model.enums import Location
from thesis.results.graph.helper.columns import pretty_names_cols
from thesis.results.graph.helper.background import InfrastructureBackgroundType, background_infrastructure
from thesis.results.graph.helper.filter import filter_for_type, filter_trajectories_linear, filter_meetings
from thesis.results.graph.linear import prepare


def infrastructure_feature(location: Location, feature: InfrastructureBackgroundType) -> alt.Chart:
    """
    Creates a chart for the given infrastructure feature.
    """
    chart = background_infrastructure(location, feature)
    chart = chart.encode(alt.OpacityValue(1))
    return chart


def meeting_distance_by_width(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Creates a boxplot for the distribution of meeting distances at every width value in the given DataFrame.
    Assigns different colours depending on the location. 
    """

    meetings = trajectories.select("location", "ID", "meeting_info", "excluded", "width", "in_path", "video").unique()
    meetings = meetings.explode("meeting_info").unnest("meeting_info")
    meetings = filter_meetings(meetings)

    # Get the width at the long pos
    meetings = meetings.drop("width") 
    meetings = meetings.join(trajectories.select("location", "ID", "long_pos", "width"),
                             left_on=["location", "ID", "meeting_long_pos"], right_on=["location", "ID", "long_pos"],
                             # Due to duplicate long_pos we might have two entries 
                             how="inner", suffix="").unique(["location", "ID", "meeting_id"])
    meetings = meetings.with_columns(pl.col("width").round(1))
    meetings = meetings.collect()
    alt.data_transformers.disable_max_rows()
    c = alt.Chart(meetings).mark_boxplot().encode(
        alt.X("width:Q").title(pretty_names_cols.get("width", None)),
        alt.Y("meeting_lateral_dist:Q").title(pretty_names_cols.get("meeting_lateral_dist", None)),
        alt.Color("location:N").title(pretty_names_cols.get("location", None))
    )
    return c

def scatter_gradient(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Creates a scatter plot with the speed of trajectories before and after the gradient.
    :param trajectories: 
    :return: 
    """
    trajectories = prepare(trajectories, "speed")
    trajectories = filter_trajectories_linear(trajectories, "speed")
    trajectories = filter_for_type(trajectories)
    alt.data_transformers.disable_max_rows()
    gradient_start = 7.0
    gradient_end = 14.0
    trajectories = trajectories.filter(pl.col("long_pos").is_in([gradient_start, gradient_end]))
    trajectories = trajectories.with_columns(pl.col("long_pos").replace_strict({
        gradient_start: "start",
        gradient_end: "end"
    }, return_dtype=pl.String))
    trajectories = trajectories.collect().pivot(
        on=["long_pos"],
        index=["ID", "primary_type"],
        values=["speed"]
    )
    trajectories = trajectories.filter(pl.col("start").is_not_null(),
                                       pl.col("end").is_not_null())
    trajectories = trajectories.with_columns(end=pl.col("end") - pl.col("start"))
    
    c = alt.Chart(trajectories).mark_circle().encode(
        alt.X(field="start", type="quantitative").title("Speed before the gradient (m/s)"),
        alt.Y(field="end", type="quantitative").title("Speed after the gradient (m/s)"),
        alt.Color(field="primary_type", type="nominal").title("Type")
    )
    return c