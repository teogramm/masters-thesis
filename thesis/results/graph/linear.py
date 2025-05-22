import sys
from typing import Optional, Literal, Iterable

import altair as alt
import polars as pl

from thesis.results.graph import calculate_group_statistics, \
    add_sample_size_to_colour
from thesis.results.graph.helper.columns import XColType, YColType, pretty_names_cols, ensure_single_value
from thesis.results.graph.helper.massage import smooth_sample_size, recalculate_acceleration
from thesis.results.graph.helper.filter import filter_trajectories_linear
from thesis.results.graph.helper.background import background_mean
from thesis.results.graph.helper.background import background_infrastructure, BackgroundType, \
    InfrastructureBackgroundType
from thesis.results.graph.helper.filter import filter_for_type, filter_long_pos


def prepare(trajectories: pl.LazyFrame, y_col: YColType | str,
            additional_interpolate_cols: Optional[Iterable[str]] = None) -> pl.LazyFrame:
    """
    Interpolate and deduplicate trajectories at every longitudinal position, to avoid duplicate or missing observations.
    :param trajectories: LazyFrame with complete datapoints for each trajectory.
    :param y_col: Main column to interpolate.
    :param additional_interpolate_cols: 
    :return: 
    """
    interpolate_cols = {y_col}
    if additional_interpolate_cols is not None:
        interpolate_cols.update(additional_interpolate_cols)
    if "speed" not in interpolate_cols:
        interpolate_cols.add("speed")

    copy_cols = ["in_path", "primary_type", "swap_major", "swap_minor", "constrained",
                 "f_ped", "excluded", "rental", "trip", "time_of_day"]

    trajectories = smooth_sample_size(trajectories, list(interpolate_cols),
                                      copy_cols)
    if "acc" in interpolate_cols:
        trajectories = recalculate_acceleration(trajectories)
    return trajectories


def line_graph_by_direction(trajectories: pl.LazyFrame, x_col: XColType, y_col: YColType | str,
                            background: Optional[BackgroundType] = "mean_all",
                            background_infrastructure_type: InfrastructureBackgroundType = None,
                            color: Optional[Literal["type"] | str] = None,
                            unconstrained_only=True,
                            peak_only=True) -> alt.TopLevelMixin:
    """
    Creates a graph with the given parameters faceted by direction.
    
    :param trajectories: Trajectories dataframe without any datapoints removed from the included trajectories.
    :param x_col: Column to use for the X axis.
    ``relative_apex_pos`` is supported only for Riddarhuskajen.
    :param y_col: Main column to use for the Y axis.
    Can be any other column than those given in the hint,however,
    the function might break and filtering might not be applied. 
    :param color: Column to use for grouping different trajectories.
    Creates a line of different colour for each value.  
    :param background: What is shown in the background of the chart, as a faded grey line.
    The ``mean_all`` type takes into account all trajectories in the dataset, irrespective of other parameters
    applied.
    When including both infrastructure and mean, the mean line is assigned a normal colour according to the scale
    of the other lines.
    :param background_infrastructure_type: Infrastructure feature to display in the background.
    :param unconstrained_only: Graph only unconstrained trajectories.
    :param peak_only: Keeps only trajectories for which ``time_of_day`` is not ``Off-peak``.
    :return: Two charts with the given parameters, one for Northbound and one for Southbound cyclists.
    """

    trajectories = prepare(trajectories, y_col)
    trajectories = filter_long_pos(trajectories)
    trajectories = filter_trajectories_linear(trajectories, y_col)
    if color == "type":
        color = "primary_type"
        trajectories = filter_for_type(trajectories)
    else:
        color = color

    # Ensure single value after collecting, so we avoid collecting multiple times
    trajectories = trajectories.collect()
    ensure_single_value(trajectories, "location")

    northbound = trajectories.filter(pl.col("direction") == "Northbound")
    southbound = trajectories.filter(pl.col("direction") == "Southbound")

    chart_nb = line_graph_one_direction(northbound, x_col, y_col, background, color,
                                        unconstrained_only, background_infrastructure_type,
                                        peak_only=peak_only)
    chart_sb = line_graph_one_direction(southbound, x_col, y_col,
                                        background, color,
                                        unconstrained_only, background_infrastructure_type,
                                        invert_x=True,
                                        peak_only=peak_only)

    chart = alt.hconcat(chart_nb, chart_sb).properties(
        resolve=alt.Resolve(scale=alt.LegendResolveMap(color=alt.ResolveMode('shared')))
    ).configure_legend(direction="horizontal", orient="bottom")
    return chart


def line_graph_one_direction(trajectories: pl.DataFrame, x_col: XColType, y_col: YColType | str,
                             background: Optional[BackgroundType] = None,
                             color_col: Optional[str] = None,
                             unconstrained_only=False,
                             background_infrastructure_type: InfrastructureBackgroundType = None,
                             invert_x=False,
                             peak_only=True) -> alt.TopLevelMixin:
    """
    Create a line graph for one direction.
    :param trajectories: Filtered DataFrame
    :param x_col: Column to use for the X axis.
    ``relative_apex_pos`` is supported only for Riddarhuskajen.
    :param y_col:  Column to use for the Y axis.
    :param background: What is shown in the background of the chart, as a faded grey line.
    The ``mean_all`` type takes into account all trajectories in the dataset, irrespective of other parameters
    applied.
    When including both infrastructure and mean, the mean line is assigned a normal colour according to the scale
    of the other lines.
    :param color_col: Column to use for grouping different trajectories.
    Creates a line of different colour for each value.  
    :param unconstrained_only: Graph only unconstrained trajectories.
    :param background_infrastructure_type: Infrastructure feature to display in the background.
    :param invert_x: Whether to display values in the x-axis in descending order.
    :param peak_only: Keeps only trajectories for which ``time_of_day`` is not ``Off-peak``.
    :return: One chart with the given parameters.
    """
    # This function is spaghetti paradise.
    ensure_single_value(trajectories, "direction")
    direction = trajectories.select(pl.col("direction").first()).item()

    bg_mean = None
    bg_infra = None
    match background:
        case "mean_all":
            bg_mean = background_mean(trajectories.lazy(), x_col, y_col)
            if background_infrastructure_type is not None:
                print(
                    f"Ignoring infrastructure type {background_infrastructure_type} since mean background is specified.",
                    file=sys.stderr)
        case "infra":
            if background_infrastructure_type is None:
                raise ValueError("Specified infrastructure background without giving a feature type.")
            bg_infra = background_infrastructure(trajectories.select(pl.col("location").first()).item(),
                                                 background_infrastructure_type, invert_x=invert_x)
        case "both":
            if background_infrastructure_type is None:
                raise ValueError("Specified infrastructure background without giving a feature type.")
            bg_infra = background_infrastructure(trajectories.select(pl.col("location").first()).item(),
                                                 background_infrastructure_type, invert_x=invert_x)
            bg_mean = background_mean(trajectories.lazy(), x_col, y_col)
        case None:
            pass
        case _:
            raise ValueError(f"Unknow background type: {background}")

    # Create the main chart
    select_cols = [x_col, y_col]
    # Rename the color column to 'color', so we can blend the background in case we want to
    if color_col is not None:
        trajectories = trajectories.rename(
            {color_col: "color"}
        )
        select_cols.append("color")
    group_by_cols = [x_col]
    if color_col is not None:
        group_by_cols.append("color")
    if unconstrained_only:
        trajectories = trajectories.filter(pl.col("constrained") == "Unconstrained")
    if peak_only:
        trajectories = trajectories.filter(pl.col("time_of_day") != "Off-peak")
    trajectories = trajectories.select(select_cols)
    trajectories = trajectories.group_by(group_by_cols)

    stats = calculate_group_statistics(trajectories, y_col)

    if color_col is None:
        # If we don't have a color, add a default one, so it appears on the legend
        if unconstrained_only:
            col_val = "Unconstrained trajectories"
        else:
            col_val = f"All {direction} trajectories"
        stats = stats.with_columns(
            color=pl.lit(col_val)
        )
        color_name = pretty_names_cols.get(y_col, y_col)
    else:
        color_name = pretty_names_cols.get(color_col, color_col)
    
    stats = add_sample_size_to_colour(stats, "color")

    chart = alt.Chart(stats, title=direction).mark_line().encode(
        alt.X(field=x_col, type="quantitative", scale=alt.Scale(reverse=invert_x, zero=False)).title(
            pretty_names_cols.get(x_col, None)),
        alt.Y(field=y_col, type="quantitative").title(pretty_names_cols.get(y_col, None)),
        alt.Color(field="color", type="nominal").title(color_name),
        alt.StrokeDash(field="Measure", type="nominal").legend(alt.Legend(orient="right", direction="vertical")),
    )
    
    # Add mean background, if specified.
    if bg_mean is not None:
        chart = (bg_mean + chart).resolve_scale(y="shared", color="independent")


    # Add infrastructure background, if specified.
    # Infrastructure uses a different scale than the other y value.
    if bg_infra is not None:
        resolve_y = "independent"
    else:
        resolve_y = "shared"
        
    if bg_infra is not None:
        return (bg_infra + chart).resolve_scale(y=resolve_y, color="independent").resolve_legend(color="shared")
    else:
        return chart.resolve_scale(y=resolve_y, color="independent").resolve_legend(color="shared")
