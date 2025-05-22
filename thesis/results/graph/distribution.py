from typing import Optional

import polars as pl
import numpy as np
import altair as alt
from scipy.stats import gaussian_kde

from thesis.processing.interpolation import deduplicate_by_long_pos
from thesis.results.graph.helper.columns import YColType, pretty_names_cols, ensure_single_value
from .helper.columns import pretty_names_location
from .helper.filter import filter_for_type
from .linear import prepare


def _filter(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Additional filter to apply when calculating distribution.
    """
    trajectories = trajectories.filter(pl.col("in_path"),
                                       pl.col("speed").is_between(0.5, 10))
    return trajectories


def distribution_kde_facet_direction(trajectories: pl.LazyFrame, y_col: YColType, color: Optional[str] = None,
                                     stroke_dash: Optional[str] = None, include_sample_size=False) -> alt.TopLevelMixin:
    """
    Create a kernel density estimate for the given column with the given trajectories, with one chart per direction.
    :param trajectories: Deduplicated and filtered trajectories.
    :param y_col: Column for which the distribution is calculated.
    :param color: Calculates multiple distributions for each different value of this column.
    Different values are given different colours.
    :param stroke_dash: Currently only ``constrained`` is supported.
    Uses a full or a dashed line depending on whether the trajectory is constrained or not.
    :param include_sample_size: Include the sample size in the names of each category.
    If combined with a ``color`` column that has many values and ``stroke_dash`` can result in very large legends.
    :return: Altair chart with distributions for Northbound and Southbound cyclists.
    """
    northbound = trajectories.filter(pl.col("direction") == "Northbound")
    southbound = trajectories.filter(pl.col("direction") == "Southbound")

    location = trajectories.select(pl.col("location").first()).collect().item()
    long_pos = trajectories.select(pl.col("long_pos").first()).collect().item()

    chart_nb = distribution_kde(northbound, y_col, color, stroke_dash, include_sample_size).properties(
        title="Northbound")
    chart_sb = distribution_kde(southbound, y_col, color, stroke_dash, include_sample_size).properties(
        title="Southbound")


    title = f"{pretty_names_location[location]} - Longitudinal position: {long_pos}"

    chart = alt.hconcat(chart_nb, chart_sb, title=title).resolve_scale(y="shared")
    return chart

def distribution_kde_facet_long_pos(trajectories: pl.LazyFrame, y_col: YColType,
                                    long_pos_values: list[float],
                                    color: Optional[str] = None,
                                    stroke_dash: Optional[str] = None,
                                    include_sample_size=False) -> alt.TopLevelMixin:
    """
    Create a kernel density estimate for the given column with the given trajectories, with one chart per given
    longitudinal position value.
    :param trajectories: Unfiltered and not interpolated trajectories.
    :param y_col: Column for which the distribution is calculated.
    :param long_pos_values: Longitudinal positions at which to calculate the distribution.
    :param color: Calculates multiple distributions for each different value of this column.
    Different values are given different colours.
    :param stroke_dash: Currently only ``constrained`` is supported.
    Uses a full or a dashed line depending on whether the trajectory is constrained or not.
    :param include_sample_size: Include the sample size in the names of each category.
    If combined with a ``color`` column that has many values and ``stroke_dash`` can result in very large legends.
    :return: Altair chart with distributions at the given positions.
    """
    ensure_single_value(trajectories, "direction")
    trajectories = prepare(trajectories, y_col)
    trajectories = trajectories.filter(pl.col("long_pos").is_in(long_pos_values))
    
    direction = trajectories.select("direction").first().collect().item()
    
    # Sort so the graphs are in the correct order
    trajectories = trajectories.filter(pl.col("direction") == direction).sort("long_pos",
                                                                            descending=(direction == "Southbound"))

    trajectories = trajectories.collect()
    chart = alt.HConcatChart(title=trajectories.select(pl.col("direction").first()).item())
    for long_pos_trj in trajectories.partition_by("long_pos", maintain_order=True):
        long_pos_value = long_pos_trj.select(pl.col("long_pos").first()).item()
        long_pos_chart = distribution_kde(long_pos_trj.lazy(), y_col, color, stroke_dash, include_sample_size)
        long_pos_chart = long_pos_chart.properties(title=str(long_pos_value))
        chart |= long_pos_chart
    chart = chart.resolve_scale(y="shared")
    return chart

def distribution_kde(trajectories: pl.LazyFrame, y_col: YColType, color: Optional[str] = None,
                     stroke_dash: Optional[str] = None, include_sample_size=False,
                     interpolate=False) -> alt.TopLevelMixin:
    """
    Perform a kernel density estimation for the given trajectories.
    :param trajectories: Datapoints from trajectories at one longitudinal position.
    If the color is ``long_pos`` this is not required.
    :param y_col: Column for which 
    :param color: Calculates multiple distributions for each different value of this column.
    Different values are given different colours.
    :param stroke_dash: Currently only ``constrained`` is supported.
    Uses a full or a dashed line depending on whether the trajectory is constrained or not.
    :param include_sample_size: Include the sample size in the names of each category.
    If combined with a ``color`` column that has many values and ``stroke_dash`` can result in very large legends.
    :param interpolate: Whether to deduplicate and interpolate the trajectories.
    :return: 
    """
    ensure_single_value(trajectories, "location")
    if color != "long_pos":
        ensure_single_value(trajectories, "long_pos")

    trajectories = _filter(trajectories)
    if color == "long_pos":
        # Interpolation has already been done
        static_cols = [stroke_dash] if stroke_dash is not None else []
        trajectories = deduplicate_by_long_pos(trajectories, ["speed"], static_cols)
    elif interpolate:
        trajectories = prepare(trajectories, y_col)

    if color == "primary_type":
        trajectories = filter_for_type(trajectories)

    trajectories = trajectories.select([x for x in (y_col, color, stroke_dash) if x is not None])
    trajectories = trajectories.collect()

    if stroke_dash == "constrained":
        trajectories = trajectories.vstack(
            trajectories.filter(pl.col("constrained") == False).with_columns(constrained=pl.lit(True))
        )
        trajectories = trajectories.with_columns(pl.col("constrained").replace_strict({
            False: "Unconstrained",
            True: "All trajectories"
        }, return_dtype=pl.String))

    points_start = trajectories.select(pl.col(y_col).min()).item() - trajectories.select(pl.col(y_col).std()).item()
    points_end = trajectories.select(pl.col(y_col).max()).item() + trajectories.select(pl.col(y_col).std()).item()
    graph_points = np.linspace(points_start, points_end, 100)
    color_names = []
    probs = []
    stroke_dash_values = []
    group_by_cols = [x for x in (color, stroke_dash) if x is not None]
    if len(group_by_cols) > 0:
        # For each colour create a list with its observations
        trajectories = trajectories.group_by(group_by_cols).agg(pl.col(y_col), pl.len().alias("N"))
        for row in trajectories.iter_rows(named=True):
            if include_sample_size:
                color_names.append(f"{row.get(color, "")} {row.get("N")}")
            else:
                color_name = row.get(color, None)
                if color_name is None:
                    color_name = row.get(stroke_dash)
                color_names.append(color_name)
            if stroke_dash is not None:
                stroke_dash_values.append(row[stroke_dash])
            probs.append(gaussian_kde(np.array(row[y_col])).evaluate(graph_points))
    else:
        kde = gaussian_kde(trajectories.get_column(y_col).to_numpy())
        color_names.append(y_col)
        probs.append(kde.evaluate(graph_points))

    # Very bad solution but will do rn
    if color is None:
        color = stroke_dash

    data = {
        color: color_names,
        y_col: [graph_points] * len(color_names),
        "PDF": probs,
    }
    if stroke_dash is not None:
        data[stroke_dash] = stroke_dash_values
    df = pl.DataFrame(data).explode(y_col, "PDF")

    pdf = alt.Chart(df).mark_line().encode(
        alt.X(field=y_col, type="quantitative").title(pretty_names_cols.get(y_col, None)),
        alt.Y(field="PDF", type="quantitative"),
        color=alt.Color(field=color, type="nominal").title(
            pretty_names_cols.get(color, None)) if color is not None else alt.Undefined,
        strokeDash=alt.StrokeDash(field=stroke_dash, type="nominal") if stroke_dash is not None else alt.Undefined
    )
    return pdf
