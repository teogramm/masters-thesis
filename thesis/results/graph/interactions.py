import altair as alt
import polars as pl

from thesis.model.enums import Location
from thesis.results.graph.helper.columns import pretty_names_cols


def _filter(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter trajectories for examining interactions
    """
    trajectories = trajectories.join(trajectories.filter(pl.col("excluded")), how="anti",
                                     left_on=["following_id"], right_on=["ID"])

    trajectories = trajectories.filter(pl.col("in_path"),
                                       ~pl.col("swap_minor"),
                                       ~pl.col("swap_major"),
                                       ~pl.col("excluded"),
                                       ~pl.col("estimated"),
                                       pl.col("following_speed_difference").abs() < 5)

    # Get a subsection of the path to avoid tracking errors at the edges
    long_pos_low = {
        Location.RIDDARHUSKAJEN: 13.0,
        Location.RIDDARHOLMSBRON_S: 4.0,
        Location.RIDDARHOLMSBRON_N: 2.0
    }
    long_pos_high = {
        Location.RIDDARHUSKAJEN: 35.0,
        Location.RIDDARHOLMSBRON_S: 14.0,
        Location.RIDDARHOLMSBRON_N: 15.0
    }
    in_section = ((pl.col("long_pos") >= pl.col("location").replace_strict(long_pos_low, return_dtype=pl.Float64)) &
                  (pl.col("long_pos") <= pl.col("location").replace_strict(long_pos_high, return_dtype=pl.Float64)))
    # Remove datapoints from a few trajectories that come extremely close to each other due to tracking errors
    extremely_close = ((pl.col("following_long_dist") < 0.5) &
                       (pl.col("following_lateral_deviation").abs() < 0.7)).any().over("ID")
    # Remove few datapoints that have any points with very big lateral deviations
    lat_dev_limit = {
        Location.RIDDARHUSKAJEN: 2.1,
        Location.RIDDARHOLMSBRON_S: 1.45,
        Location.RIDDARHOLMSBRON_N: 1.6
    }
    over_limit = (pl.col("following_lateral_deviation").abs() >
                  pl.col("location").replace_strict(lat_dev_limit, return_dtype=pl.Float64))
    # Get around some bug in polars
    trajectories = trajectories.with_columns(over_limit.alias("over_limit"))
    trajectories = trajectories.filter(
        in_section,
        ~extremely_close,
        ~pl.col("over_limit").any().over("ID")
    )

    trajectories = trajectories.drop("over_limit")

    return trajectories


def _filter_overtakes(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Apply additional filters  
    """
    trajectories = trajectories.join(trajectories.filter(pl.col("excluded")), how="anti",
                                     left_on=["overtake_id"], right_on=["ID"])
    trajectories = trajectories.filter(pl.col("in_path"),
                                       pl.col("overtakes_id").list.len() > 0,
                                       ~pl.col("swap_minor"),
                                       ~pl.col("swap_major"),
                                       ~pl.col("excluded"))
    return trajectories
    

def following_long_dist_lat_dev(trajectories: pl.LazyFrame,
                                max_follow_dist: float = 5) -> alt.TopLevelMixin:
    """
    Create graph comparing longitudinal distance to lateral deviation.
    """
    trajectories = trajectories.explode("following_info").unnest("following_info")

    trajectories = _filter(trajectories)

    trajectories = trajectories.select("following_long_dist", "following_lateral_deviation",
                                       "following_speed_difference", "following_time_headway", "following_id",
                                       "ID")

    trajectories = trajectories.filter(pl.col("following_long_dist").is_between(0, max_follow_dist))

    alt.data_transformers.disable_max_rows()
    trajectories = trajectories.collect()
    print(f"Following pairs: {trajectories.select("ID", "following_id").n_unique()}")
    n = len(trajectories)
    c = alt.Chart(trajectories).mark_point(size=1).encode(
        alt.X("following_long_dist:Q").title(pretty_names_cols["following_long_dist"]),
        alt.Y("following_lateral_deviation:Q").title(pretty_names_cols["following_lateral_deviation"]),
        alt.Color("following_speed_difference:Q").title(pretty_names_cols["following_speed_difference"]).scale(
            scheme="redyellowgreen")
    )

    return c


def following_headway_lat_dev(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Create graph comparing following time headway to lateral deviation. 
    """
    trajectories = trajectories.explode("following_info").unnest("following_info")

    trajectories = _filter(trajectories)

    trajectories = trajectories.select("following_long_dist", "following_lateral_deviation",
                                       "following_speed_difference", "following_time_headway")

    trajectories = trajectories.with_columns(
        pl.col("following_time_headway").truediv(1000)
    )

    trajectories = trajectories.filter(pl.col("following_time_headway").is_between(0, 2))

    trajectories = trajectories.collect()
    alt.data_transformers.enable("vegafusion")
    c = alt.Chart(trajectories).mark_point(size=1).encode(
        alt.X("following_time_headway:Q").title(pretty_names_cols["following_time_headway"]),
        alt.Y("following_lateral_deviation:Q").title(pretty_names_cols["following_lateral_deviation"]),
        alt.Color("following_speed_difference:Q").scale(scheme="redyellowgreen").title(
            pretty_names_cols["following_speed_difference"]),
    )

    return c


def overtake_histogram(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Create histogram of the longitudinal positions of overtakes.
    """
    trajectories = trajectories.select("location", "ID", "direction", "primary_type", "overtake_info",
                                       "excluded", "swap_minor", "swap_major", "in_path", "overtakes_id").unique()
    trajectories = trajectories.explode("overtake_info").unnest("overtake_info")
    trajectories = _filter_overtakes(trajectories).collect()

    bin_start = trajectories.select(pl.col("overtake_long_pos").min().alias("breakpoint")).to_series()
    breakpoints = trajectories.get_column("overtake_long_pos").hist().get_column("breakpoint")
    breakpoints = breakpoints.append(bin_start).sort()

    long_pos_sb = trajectories.filter(pl.col("direction") == "Southbound").get_column("overtake_long_pos").hist(
        bins=breakpoints, include_category=False
    ).with_columns(
        direction=pl.lit("Southbound")
    )
    long_pos_nb = trajectories.filter(pl.col("direction") == "Northbound").get_column("overtake_long_pos").hist(
        bins=breakpoints, include_category=False
    ).with_columns(direction=pl.lit("Northbound"))

    long_pos = pl.concat([long_pos_sb, long_pos_nb]).sort("breakpoint")
    long_pos = long_pos.with_columns(
        bin_start=pl.col("breakpoint").shift(2, fill_value=bin_start.item()),
    ).rename({"breakpoint": "bin_end"})

    chart = alt.Chart(long_pos).mark_bar().encode(
        alt.X("bin_start", bin="binned").title("Longitudinal position of overtake (m)"),
        alt.X2("bin_end").title("Longitudinal position of overtake (m)"),
        alt.Y("count").title("Number of overtakes"),
        alt.Color("direction").title("Direction")
    )
    return chart
