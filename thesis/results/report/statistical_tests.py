from typing import Literal

import numpy as np
import polars as pl

from thesis.model.enums import Location
from thesis.results.graph.helper.filter import filter_trajectories_linear
from thesis.results.graph.linear import prepare
from thesis.results.statistical_tests import _test_long_pos

from scipy.stats import ks_2samp, ttest_rel


def compare_unconstrained(trajectories: pl.LazyFrame, time_periods: Literal["am-pm", "am-off-peak", "pm-off-peak"]):
    """
    Performs a Kolmogorov-Smirnov test between unconstrained cyclists in two time periods.
    :param time_periods: Which periods will be compared.
    """
    match time_periods:
        case "am-pm":
            period_1 = ["am"]
            period_2 = ["pm"]
        case "am-off-peak":
            period_1 = ["am"]
            period_2 = ["between_peaks", "other"]
        case "pm-off-peak":
            period_1 = ["pm"]
            period_2 = ["between_peaks", "other"]
        case _:
            raise ValueError(f"Invalid time_periods: {time_periods}")

    trajectories = prepare(trajectories, "speed")
    trajectories = filter_trajectories_linear(trajectories, "speed")
    trajectories = trajectories.filter(pl.col("constrained") == "Unconstrained")
    results = {
        "location": [],
        "direction": [],
        "p_value": [],
        "KS_stat": [],
        "n_am": [],
        "n_pm": [],
        "mean_am": [],
        "mean_pm": []
    }
    trajectories = trajectories.collect()
    for location in Location:
        for direction in ("Northbound", "Southbound"):
            relevant_trajectories = trajectories.filter(pl.col("location") == location,
                                                        pl.col("direction") == direction)
            # Measure before the infrastructure effect
            if direction == "Northbound":
                long_pos = _test_long_pos["speed"][direction][location][0]
            else:
                long_pos = _test_long_pos["speed"][direction][location][-1]
            # Split am and pm 
            am = (relevant_trajectories.filter(pl.col("long_pos") == long_pos,
                                               pl.col("time_of_day").is_in(period_1)).select("speed")
                  .get_column("speed").to_numpy())
            pm = (relevant_trajectories.filter(pl.col("long_pos") == long_pos,
                                               pl.col("time_of_day").is_in(period_2)).select("speed")
                  .get_column("speed").to_numpy())
            test = ks_2samp(am, pm, nan_policy="raise")
            results["location"].append(location)
            results["direction"].append(direction)
            results["p_value"].append(test.pvalue)
            results["KS_stat"].append(test.statistic)
            results["n_am"].append(len(am))
            results["n_pm"].append(len(pm))
            results["mean_am"].append(np.mean(am))
            results["mean_pm"].append(np.mean(pm))
    return pl.DataFrame(results).sort("location", "direction")


def rbn_t_test_south_of_path(trajectories: pl.LazyFrame) -> pl.DataFrame:
    """
    Test whether the speed difference between the start of the path and the base of the curve is significant
    in Riddarholmsbron N.
    :param trajectories: Deduplicated and interpolated trajectories.
    :return: DataFrame with test results
    """
    long_pos_south = 2.0
    long_pos_north = 7.0

    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N,
                                       pl.col("long_pos").is_in((long_pos_south, long_pos_north)))

    trajectories = filter_trajectories_linear(trajectories, "speed")

    # Keep only dominant flow trajectories
    is_dominant_am = (pl.col("time_of_day") == "am") & (pl.col("direction") == "Northbound")
    is_dominant_pm = (pl.col("time_of_day") == "pm") & (pl.col("direction") == "Southbound")
    is_dominant = is_dominant_am | is_dominant_pm
    trajectories = trajectories.filter(is_dominant,
                                       pl.col("constrained") == "Unconstrained")

    trajectories = trajectories.select("ID", "long_pos", "speed", "direction")
    trajectories = trajectories.collect().pivot("long_pos", index=["ID", "direction"], values=["speed"])
    trajectories = trajectories.drop_nans()

    results = {
        "direction": [],
        "mean_pre": [],
        "mean_post": [],
        "p-value": [],
        "t-stat": [],
        "df": []
    }

    for direction in ["Northbound", "Southbound"]:
        if direction == "Northbound":
            lp_pre = long_pos_south
            lp_post = long_pos_north
        else:
            lp_pre = long_pos_north
            lp_post = long_pos_south

        dir_trajectories = trajectories.filter(pl.col("direction") == direction)

        pre_values = dir_trajectories.get_column(str(lp_pre)).to_numpy()
        post_values = dir_trajectories.get_column(str(lp_post)).to_numpy()

        if direction == "Northbound":
            alternative = "less"
        else:
            alternative = "greater"

        t_results = ttest_rel(pre_values, post_values, nan_policy="raise", alternative=alternative)

        results["direction"].append(direction)
        results["mean_pre"].append(np.mean(pre_values))
        results["mean_post"].append(np.mean(post_values))
        results["p-value"].append(t_results.pvalue)
        results["t-stat"].append(t_results.statistic)
        results["df"].append(t_results.df)

    df = pl.DataFrame(results)
    return df


def rbn_t_test_unconstrained_electric(trajectories: pl.LazyFrame) -> pl.DataFrame:
    """
    Test whether the speed of unconstrained electric cyclists is reduced by the slope, by performing a one-sided
    paired sample t-test.
    :param trajectories: Deduplicated and interpolated trajectories.
    :return: 
    """
    lp_pre = 8.0
    lp_post = 16.0

    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N,
                                       pl.col("long_pos").is_in((lp_pre, lp_post)),
                                       pl.col("direction") == "Northbound",
                                       pl.col("constrained") == "Unconstrained",
                                       pl.col("time_of_day") == "am",
                                       pl.col("primary_type") == "Electric")

    trajectories = filter_trajectories_linear(trajectories, "speed")

    trajectories = trajectories.select("ID", "long_pos", "speed")
    trajectories = trajectories.collect().pivot("long_pos", index=["ID"], values=["speed"])
    trajectories = trajectories.drop_nans()

    results = {
        "mean_pre": [],
        "mean_post": [],
        "p-value": [],
        "t-stat": [],
        "df": []
    }

    pre_values = trajectories.get_column(str(lp_pre)).to_numpy()
    post_values = trajectories.get_column(str(lp_post)).to_numpy()

    t_results = ttest_rel(pre_values, post_values, nan_policy="raise", alternative="greater")

    results["mean_pre"].append(np.mean(pre_values))
    results["mean_post"].append(np.mean(post_values))
    results["p-value"].append(t_results.pvalue)
    results["t-stat"].append(t_results.statistic)
    results["df"].append(t_results.df)

    df = pl.DataFrame(results)
    return df
