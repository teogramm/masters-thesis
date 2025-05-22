from typing import Literal

import altair as alt
import polars as pl
from altair.vegalite.v5.schema.channels import OpacityValue

from thesis.files.infrastructure import open_width, open_elevation, open_curvature
from thesis.model.enums import Location
from thesis.results.graph import calculate_group_statistics
from thesis.results.graph.helper.columns import YColType, XColType, pretty_names_cols
from thesis.results.graph.helper.filter import filter_long_pos

BackgroundType = Literal["infra", "mean_all", "both"]
InfrastructureBackgroundType = Literal["width", "elevation", "curvature"]


def background_mean(trajectories: pl.LazyFrame, x_col: XColType, y_col: YColType) -> alt.Chart:
    """
    Create a graph with the given parameters, for the entire trajectories dataframe (without color/faceting).
    :param trajectories: Trajectories dataframe, deduplicated and filtered.
    :param x_col: Column to place on the x-axis.
    :param y_col: Column to place on the y-axis.
    :return: Chart intended to be used as a background.
    """
    # Create a simple line chart with the mean value for all trajectories
    direction = trajectories.select(pl.col("direction").first()).collect().item()
    trajectories = trajectories.select(x_col, y_col)
    trajectories = trajectories.group_by(x_col)

    stats = calculate_group_statistics(trajectories, y_col)
    # Keep only the mean and add one column so we can set the colour
    stats = stats.with_columns(
        color=pl.lit(f"All {direction} trajectories")
    )

    chart = alt.Chart(stats).mark_line().encode(
        alt.X(field=x_col, type="quantitative").title(pretty_names_cols.get(x_col, None)),
        alt.Y(field=y_col, type="quantitative").title(pretty_names_cols.get(y_col, None)),
        alt.Color(field="color", type="nominal").title("").scale(alt.Scale(range=["grey"])),
        alt.StrokeDash(field="Measure", type="nominal"),
        OpacityValue(0.4)
    )

    return chart


def background_infrastructure(location: Location, feature: InfrastructureBackgroundType,
                              invert_x=False) -> alt.Chart:
    """
    Create a background line with the given infrastructure feature.
    :param location: Location of the feature (Riddarhuskajen, Riddarholmsbron N or S)
    :param feature: Infrastructure feature to map.
    :param invert_x: Whether to display the x-axis in descending order.
    :return: Chart with the given infrastructure feature, intended to be used as a background.
    """
    gfeature = None
    if feature == "width":
        if location == Location.RIDDARHOLMSBRON_N or location == Location.RIDDARHOLMSBRON_S:
            gfeature = open_width().filter(pl.col("location") == location)
    elif feature == "elevation":
        if location == Location.RIDDARHOLMSBRON_N:
            gfeature = open_elevation().filter(pl.col("location") == location)
    elif feature == "curvature":
        if location == Location.RIDDARHUSKAJEN:
            gfeature = open_curvature()
            return _graph_curvature(gfeature)
    if gfeature is None:
        raise ValueError(f"Unsupported feature {feature} for location {location}")
    else:
        return _graph_width_elevation(gfeature, feature, location, invert_x)


def _graph_curvature(feature: pl.DataFrame) -> alt.Chart:
    """
    Graph curvature values.
    :param feature: DataFrame with the ``curvature`` values at all longitudinal positions.
    :return: Chart mapping curvature.
    """
    feature = feature.select("location", "long_pos", "curvature").with_columns(color=pl.lit("Curvature"))
    feature = filter_long_pos(feature.lazy()).collect()
    
    c = alt.Chart(feature).mark_line().encode(
        alt.X(field="long_pos", type="quantitative"),
        alt.Y(field="curvature", type="quantitative").title("Curvature"),
        alt.Color(field="color", type="nominal").title("Infrastructure").scale(alt.Scale(range=["grey"])),
        OpacityValue(0.4)
    )

    return c


def _graph_width_elevation(feature: pl.DataFrame, feature_name: InfrastructureBackgroundType,
                           location: Location, invert_x=False) -> alt.Chart:
    """
    Crate graph with width or elevation.
    :param feature: The feature to graph, containing values at every longitudinal position.
    Columns ``<feature_name>_m_start``, ``from_long_pos`` and  ``<feature_name>_m_end``, ``to_long_pos`` contain
    the values of the given feature in a section, defined by the given longitudinal positions.
    If the feature values differ between the end and the start, intermediate values are linearly interpolated. 
    :param feature_name: The name of feature that is being graphed.
    :param location: Location of the feature (Riddarholmsbron N or S)
    :param invert_x: Whether to display the x-axis in descending order.
    :return: Chart with the given infrastructure feature, intended to be used as a background.
    """
    col_start = pl.col(f"{feature_name}_m_start")
    col_end = pl.col(f"{feature_name}_m_end")

    max_long_pos: float = feature.get_column("to_long_pos").max()

    interpolated = pl.DataFrame().with_columns(
        pl.int_range(0, (max_long_pos * 2 + 1)).truediv(2).alias("long_pos"),
        pl.lit(None, dtype=pl.Float64).alias(feature_name),
        pl.lit(None, dtype=pl.Int32).alias("order")
    )

    # Join the values for known long pos to the interpolated dataframe
    for long_pos_col, feature_col in [("from_long_pos", col_start), ("to_long_pos", col_end)]:
        if long_pos_col == "from_long_pos":
            order = 1
        else:
            order = 0
        interpolated = interpolated.vstack(feature.select(pl.col(long_pos_col).alias("long_pos"),
                                                          feature_col.alias(feature_name)).with_columns(
            order=pl.lit(order)
        ))

    interpolated = (interpolated.sort("long_pos", "order", descending=invert_x)
                    .with_columns(pl.col(feature_name).interpolate())
                    .filter(pl.col(feature_name).is_not_null())
                    .with_columns(color=pl.lit(pretty_names_cols.get(feature_name, None)),
                                  location=pl.lit(location)))
    # Roundabout way of keeping observations within the graph
    interpolated = filter_long_pos(interpolated.lazy()).collect()

    c = alt.Chart(interpolated).mark_line().encode(
        alt.X(field="long_pos", type="quantitative", scale=alt.Scale(reverse=invert_x)).title(
            pretty_names_cols.get("long_pos", None)),
        alt.Y(field=feature_name, type="quantitative").title(pretty_names_cols.get(feature_name, None)),
        alt.Color(field="color", type="nominal").title("Infrastructure").scale(alt.Scale(range=["grey"])),
        OpacityValue(0.4)
    )
    return c
