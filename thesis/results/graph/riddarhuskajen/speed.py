from typing import Sequence, Optional

import polars as pl
import altair as alt

from thesis.model.enums import Location, Primary_Type_Literal
from thesis.results.graph.linear import line_graph_by_direction
from thesis.results.graph.speed import _filter
from thesis.util.image_coord_transforms import polygons_to_image


def by_corner_cutting(trajectories_rk: pl.LazyFrame,
                      speed_min: float = 0.5, speed_max: float = 15.0) -> alt.TopLevelMixin:
    """
    Requires ``relative_apex_pos`` and ``cuts_corner`` column.
    """
    # Filter observations with LongPos equal to 0, as they are often outide the coordinate system
    # In addition keep only points in the path and which have speed information
    trajectories_rk = _filter(trajectories_rk, speed_min, speed_max)

    chart = create_line_chart(trajectories_rk, "relative_apex_pos", "speed", "cuts_corner", "direction")

    return chart

def by_corner_cutting_and_time_of_day(trajectories_rk: pl.LazyFrame,
                      speed_min: float = 0.5, speed_max: float = 15.0) -> alt.TopLevelMixin:
    """
    Requires ``relative_apex_pos`` and ``cuts_corner`` column.
    """
    # Filter observations with LongPos equal to 0, as they are often outide the coordinate system
    # In addition keep only points in the path and which have speed information
    trajectories_rk = _filter(trajectories_rk, speed_min, speed_max)

    chart = create_line_chart(trajectories_rk, "relative_apex_pos", "speed", "cuts_corner", "direction",
                              facet_col="time_of_day")

    return chart


def by_region(trajectories_rk: pl.DataFrame, hull: pl.DataFrame, direction: str) -> None:
    trajectories_rk = trajectories_rk.filter(pl.col("long_pos") > 0, pl.col("in_path"),
                                             pl.col("speed").is_not_nan(),
                                             pl.col("speed") < 10, pl.col("speed") > 0.5,
                                             pl.col("direction") == direction)
    is_east = pl.when(pl.col("lat_pos") > 0).then(True).otherwise(False).alias("is_east")
    trajectories_rk = trajectories_rk.with_columns(is_east)

    trajectories_rk = (trajectories_rk.select(["long_pos", "speed", "is_east"])
                       .group_by(["long_pos", "is_east"])
                       .agg(pl.col("speed").mean().alias("mean_speed"),
                            pl.col("speed").std().alias("sdev_speed"),
                            pl.len().alias("n")))

    base = alt.Chart(trajectories_rk).transform_calculate(
        ymin="datum.mean_speed-datum.sdev_speed",
        ymax="datum.mean_speed+datum.sdev_speed",
    )

    lines = base.mark_line(interpolate="basis").encode(
        alt.X("long_pos:Q").title("Longitudinal position (m)"),
        alt.Y("mean_speed:Q").title("Speed (m/s)"),
        alt.Color("is_east:N").title("Mean speed")
    )
    lines.show()

    hull = hull.join(trajectories_rk.select(["long_pos", "is_east", "mean_speed"]),
                     on=["long_pos", "is_east"], how="inner")
    hulls = hull.get_column("hull").to_numpy()
    xp = [[p[0] for p in h] for h in hulls]
    yp = [[p[1] for p in h] for h in hulls]
    w = hull.get_column("mean_speed").to_numpy()
    polygons_to_image(Location.RIDDARHUSKAJEN, xp, yp, w)
