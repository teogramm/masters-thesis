import altair as alt
import polars as pl

def line(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Create a line plot of the mean speed for each curvature value, rounded to 3 decimal places. 
    :return: Line graph.
    """
    trajectories = trajectories.filter(pl.col("speed").is_not_nan(),
                                       pl.col("speed").is_between(0.5, 10),
                                       pl.col("long_pos").is_between(5, 35),
                                       pl.col("curvature").is_between(-0.4,0.4),
                                       ~pl.col("issue"))
    
    trajectories = trajectories.with_columns(pl.col("curvature").round(3).abs()).filter(
        ~pl.col("curvature").is_between(-0.001,0.001)
    ).group_by("curvature").agg(
        pl.col("speed").mean()
    )
    trajectories = trajectories.collect()
    c = alt.Chart(trajectories).mark_line().encode(
        alt.X("curvature:Q"),
        alt.Y("speed:Q")
    )
    return c