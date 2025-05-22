import polars as pl
from scipy.stats import ks_2samp

from thesis.model.enums import Location
from thesis.results.graph.helper.columns import ensure_single_value
from thesis.results.graph.helper.filter import filter_trajectories_linear

_test_long_pos = {
    "speed": {
        "Northbound": {
            Location.RIDDARHUSKAJEN: [11.0, 21.5, 33.0],
            Location.RIDDARHOLMSBRON_N: [7.0, 11.0, 14.0],
            Location.RIDDARHOLMSBRON_S: [8.0, 12.0, 16.0]
        },
        "Southbound": {
            Location.RIDDARHUSKAJEN: [11.0, 21.5, 33.0],
            Location.RIDDARHOLMSBRON_N: [5.0, 11.0, 14.0],
            Location.RIDDARHOLMSBRON_S: [8.0, 12.0, 16.0]
        }
    },
    "lat_pos": {
        "Northbound": {
            Location.RIDDARHUSKAJEN: [11.0, 21.5, 33.0],
            Location.RIDDARHOLMSBRON_N: [7.0, 10.0, 14.0],
            Location.RIDDARHOLMSBRON_S: [4.0, 8.0, 13.0]
        },
        "Southbound": {
            Location.RIDDARHUSKAJEN: [11.0, 21.5, 33.0],
            Location.RIDDARHOLMSBRON_N: [5.0, 10.0, 14.0],
            Location.RIDDARHOLMSBRON_S: [4.0, 8.0, 13.0]
        }
    }
}


def ks_free_constrained(trajectories: pl.LazyFrame, y_col: str) -> pl.DataFrame:
    """
    Compares the distributions of the given column between constrained and unconstrained cyclists with a
    Kolmogorov-Smirnov test.
    :param trajectories: Trajectories DataFrame for a single location.
    Should be interpolated and deduplicated.
    :param y_col: Name of the variable to compare.
    :return: DataFrame with test results.
    """
    ensure_single_value(trajectories, "location")

    location = trajectories.select(pl.col("location").first()).collect().item()
    trajectories = filter_trajectories_linear(trajectories, y_col)
    results = {
        "direction": [],
        "long_pos": [],
        "p_value": [],
        "KS_stat": []
    }
    for direction in ("Northbound", "Southbound"):
        dir_trajectories = trajectories.filter(pl.col("direction") == direction)
        long_positions = _test_long_pos[y_col][direction][location]
        for long_position in long_positions:
            dir_long_pos_trajectories = dir_trajectories.filter(pl.col("long_pos") == long_position)
            p_value, stat = _ks_free_constrained_all(dir_long_pos_trajectories, y_col)
            results["direction"].append(direction)
            results["long_pos"].append(long_position)
            results["p_value"].append(p_value)
            results["KS_stat"].append(stat)
    return pl.DataFrame(results).sort("direction", "long_pos")


def _ks_free_constrained_all(trajectories: pl.LazyFrame, y_col: str) -> tuple[float, float]:
    """
    Performs a Kolmogorov-Smirnov test between constrained and unconstrained cyclists in the given
    dataframe, for the given variable.
    :param trajectories: Trajectories at a single longitudinal position.
    Should be interpolated and deduplicated.
    :param y_col: Name of the variable to compare.
    :return: p-value and KS-test statistic.
    """

    trajectories = trajectories.collect()
    ensure_single_value(trajectories, "long_pos")

    unconstrained = trajectories.filter(pl.col("constrained") == "Unconstrained")
    constrained = trajectories.filter(pl.col("constrained") != "Unconstrained")

    unconstrained = unconstrained.select(y_col).get_column(y_col).to_numpy()
    constrained = constrained.select(y_col).get_column(y_col).to_numpy()

    results = ks_2samp(unconstrained, constrained, nan_policy="raise")
    return results.pvalue, results.statistic
