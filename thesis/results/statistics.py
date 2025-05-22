import numpy as np
import polars as pl
import altair as alt
from altair import Tooltip

from thesis.filtering.filters import PathFilter
from thesis.model.enums import Location
from thesis.results.graph.helper.filter import filter_trajectories_linear

alt.renderers.enable("browser")
def departures_per_hour(trajectories: pl.DataFrame, aggregation_minutes: int = 15,
                        time_start: pl.Time = None, time_end: pl.Time = None):
    # Keep the first record of each observation
    grouped = trajectories.group_by("ID").first().sort(by="Time").group_by_dynamic("Time", every=f"{aggregation_minutes}m").agg(pl.len())
    # Optional filtering
    grouped: pl.DataFrame = grouped.filter(pl.col("Time").dt.time().is_between(time_start, time_end))
    grouped = grouped.with_columns(date=pl.col("Time").dt.date().cast(pl.String), time_only=pl.col("Time").dt.time().cast(pl.String))
    chart = (
        alt.Chart(grouped).mark_bar().encode(
            x="time_only",
            y="len",
            color="date"
        )
    )
    chart.show()

def corner_cutting_by_time_of_day(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN,
                                       pl.col("path") == PathFilter.INSIDE)
    
    trajectories = trajectories.group_by("time_of_day", "cuts_corner", "direction").agg(
        pl.col("ID").n_unique().alias("n")
    ).with_columns(percentage=(pl.col("n") /
                               pl.col("n").sum().over("time_of_day", "direction")).round(2))
    
    return trajectories.sort("time_of_day", "direction", "cuts_corner")

def overtakes_by_time_of_day_and_direction(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    trajectories = trajectories.filter(pl.col("overtakes_id").list.len() > 0)
    
    trajectories = trajectories.select("location", "ID", "time_of_day", "direction", "overtakes_id").unique().group_by("location", "time_of_day", "direction").agg(
        pl.col("overtakes_id").list.n_unique().sum().alias("n")
    )
    
    return trajectories

def rbn_speed_drop_uphill(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    
    :param trajectories: Deduplicated and interpolated trajectories
    :return: 
    """
    lp_pre = 8.0
    lp_post = 15.5
    
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N,
                                       pl.col("long_pos").is_in((lp_pre, lp_post)),
                                       pl.col("direction") == "Northbound",
                                       pl.col("primary_type").is_not_null())
    
    trajectories = filter_trajectories_linear(trajectories, "speed")
    
    results = {
        "bike_type": [],
        "constrained": [],
        "pre_mean": [],
        "pre_std": [],
        "post_mean": [],
        "post_std": [],
        "percent_change": []
    }
    
    trajectories = trajectories.collect()
    
    for bike_type in ("Regular", "Electric", "Race"):
        type_trajectories = trajectories.filter(pl.col("primary_type") == bike_type)
        
        unconstrained = type_trajectories.filter(pl.col("constrained") == "Unconstrained",
                                                 pl.col("time_of_day") == "am")
        unconstrained = unconstrained.select("ID", "long_pos", "speed")
        unconstrained = unconstrained.pivot("long_pos", index=["ID"], values=["speed"]).drop_nans().drop_nulls()
        pre_values = unconstrained.get_column(str(lp_pre)).to_numpy()
        post_values = unconstrained.get_column(str(lp_post)).to_numpy()
        results["bike_type"].append(bike_type)
        results["constrained"].append("unconstrained")
        results["pre_mean"].append(np.mean(pre_values))
        results["pre_std"].append(np.std(pre_values))
        results["post_mean"].append(np.mean(post_values))
        results["post_std"].append(np.std(post_values))
        results["percent_change"].append((np.mean(post_values) - np.mean(pre_values))/np.mean(pre_values))

        type_trajectories = type_trajectories.select("ID", "long_pos", "speed")
        type_trajectories = type_trajectories.pivot("long_pos", index=["ID"], values=["speed"]).drop_nans().drop_nulls()
        pre_values = type_trajectories.get_column(str(lp_pre)).to_numpy()
        post_values = type_trajectories.get_column(str(lp_post)).to_numpy()
        results["bike_type"].append(bike_type)
        results["constrained"].append("all")
        results["pre_mean"].append(np.mean(pre_values))
        results["pre_std"].append(np.std(pre_values))
        results["post_mean"].append(np.mean(post_values))
        results["post_std"].append(np.std(post_values))
        results["percent_change"].append((np.mean(post_values) - np.mean(pre_values))/np.mean(pre_values))
    
    from polars.selectors import numeric
    df = pl.DataFrame(results).with_columns(numeric().round(2))
    return df