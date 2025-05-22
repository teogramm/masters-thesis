import polars as pl
import altair as alt

from thesis.files import BASE_PATH
from thesis.model.enums import Location
from thesis.results.graph.infrastructure import infrastructure_feature, meeting_distance_by_width
from thesis.results.graph.interactions import following_long_dist_lat_dev, overtake_histogram
from thesis.results.graph.linear import line_graph_by_direction

_save_dir = BASE_PATH / "results"
_save_dir.mkdir(exist_ok=True)


def riddarhuskajen(trajectories: pl.LazyFrame):
    """
    Save graphs for Riddarhuskajen.
    """
    location_dir = _save_dir / "riddarhuskajen"
    location_dir.mkdir(exist_ok=True)
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN)
    # Split trajectories into peak and off-peak
    peak_type = pl.Enum(["Peak", "Off-peak"])
    trajectories = trajectories.with_columns(
        pl.when(pl.col("time_of_day").is_in(["am", "pm"])).then(pl.lit("Peak", dtype=peak_type))
        .otherwise(pl.lit("Off-peak", dtype=peak_type)).alias("time_of_day"))
    following_long_dist_lat_dev(trajectories).save(location_dir / "following.png", scale_factor=4.0)
    overtake_histogram(trajectories).save(location_dir / "overtakes.svg")

    # Invert lateral position of southbound trajectories, so positive numbers correspond to the correct travel lane
    trajectories = trajectories.with_columns(pl.when(pl.col("direction") == "Southbound")
                                             .then(-pl.col("lat_pos"))
                                             .otherwise(pl.col("lat_pos")))
    
    for y_col in ("speed", "lat_pos"):
        var_dir = location_dir / y_col
        var_dir.mkdir(exist_ok=True)
        unconstrained_with_infra = line_graph_by_direction(trajectories, "long_pos", y_col,
                                                           background="infra", background_infrastructure_type="curvature",
                                                           color="time_of_day", unconstrained_only=True, peak_only=False)
        unconstrained_with_infra.save(var_dir / "1_unconstrained.svg")

        unconstrained_vs_all = line_graph_by_direction(trajectories, "long_pos", y_col,
                                                       background="mean_all",
                                                       unconstrained_only=True)
        unconstrained_vs_all.save(var_dir / "2_all.svg")

        unconstrained_by_type = line_graph_by_direction(trajectories, "long_pos", y_col, color="type",
                                                        background="mean_all",
                                                        unconstrained_only=True)
        unconstrained_by_type.save(var_dir / "3_unconstrained_by_type.svg")

        all_by_type = line_graph_by_direction(trajectories, "long_pos", y_col, color="type",
                                              background="mean_all",
                                              unconstrained_only=False)
        all_by_type.save(var_dir / "4_all_by_type.svg")


def riddarholmsbron_n(trajectories: pl.LazyFrame):
    """
    Save graphs for Riddarholmsbron N.
    """
    location_dir = _save_dir / "riddarholmsbron_n"
    location_dir.mkdir(exist_ok=True)
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N)

    following_long_dist_lat_dev(trajectories).save(location_dir / "following.png", scale_factor=4.0)
    overtake_histogram(trajectories).save(location_dir / "overtakes.svg")
    meeting_distance_by_width(trajectories).save(location_dir / "meetings.svg")

    # Include only dominant peak flow as unconstrained
    is_dominant_am = (pl.col("time_of_day") == "am") & (pl.col("direction") == "Northbound")
    is_dominant_pm = (pl.col("time_of_day") == "pm") & (pl.col("direction") == "Southbound")
    peak_type = pl.Enum(["AM Peak", "PM Peak", "Off-peak"])
    trajectories = trajectories.with_columns(pl.when(is_dominant_am).then(pl.lit("AM Peak", dtype=peak_type))
                                             .when(is_dominant_pm).then(pl.lit("PM Peak", dtype=peak_type))
                                             .otherwise(pl.lit("Off-peak", dtype=peak_type)).alias("time_of_day"))
    infrastructure = (infrastructure_feature(Location.RIDDARHOLMSBRON_N, "width").encode(
        alt.Color(field="color", type="nominal").title("Infrastructure")
    )
                      + infrastructure_feature(Location.RIDDARHOLMSBRON_N, "elevation").encode(
                alt.Color(field="color", type="nominal").title("Infrastructure")
            )).resolve_axis(y="independent")
    infrastructure.save(location_dir / "infrastructure.svg")

    # Invert lateral position of southbound trajectories, so positive numbers correspond to the correct travel lane
    trajectories = trajectories.with_columns(pl.when(pl.col("direction") == "Southbound")
                                             .then(-pl.col("lat_pos"))
                                             .otherwise(pl.col("lat_pos")))
    
    for y_col in ("speed", "lat_pos"):
        var_dir = location_dir / y_col
        var_dir.mkdir(exist_ok=True)
        unconstrained_vs_all = line_graph_by_direction(trajectories.sort("time_of_day"), "long_pos", y_col,
                                                       background="mean_all",
                                                       unconstrained_only=True, color="time_of_day", peak_only=False)
        unconstrained_vs_all.save(var_dir / "2_all.svg")

        unconstrained_by_type = line_graph_by_direction(trajectories, "long_pos", y_col, color="type",
                                                        background="mean_all",
                                                        unconstrained_only=True)
        unconstrained_by_type.save(var_dir / "3_unconstrained_by_type.svg")

        all_by_type = line_graph_by_direction(trajectories, "long_pos", y_col, color="type",
                                              background="mean_all",
                                              unconstrained_only=False)
        all_by_type.save(var_dir / "4_all_by_type.svg")


def riddarholmsbron_s(trajectories: pl.LazyFrame):
    """
    Save graphs for Riddarholmsbron S.
    """
    location_dir = _save_dir / "riddarholmsbron_s"
    location_dir.mkdir(exist_ok=True)
    meeting_distance_by_width(trajectories.filter(pl.col("location") !=
                                                  Location.RIDDARHUSKAJEN)).save(location_dir / "meetings.svg")
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHOLMSBRON_S)

    following_long_dist_lat_dev(trajectories).save(location_dir / "following.png", scale_factor=4.0)
    overtake_histogram(trajectories).save(location_dir / "overtakes.svg")
    
    peak_type = pl.Enum(["AM Peak", "PM Peak", "Off-peak"])
    is_dominant_am = (pl.col("time_of_day") == "am") & (pl.col("direction") == "Northbound")
    is_dominant_pm = (pl.col("time_of_day") == "pm") & (pl.col("direction") == "Southbound")

    # Invert lateral position of southbound trajectories, so positive numbers correspond to the correct travel lane
    trajectories = trajectories.with_columns(pl.when(pl.col("direction") == "Southbound")
                                             .then(-pl.col("lat_pos"))
                                             .otherwise(pl.col("lat_pos")))
    
    # Keeping only the dominant flow is done inside the loop 
    for y_col in ("speed", "lat_pos"):
        var_dir = location_dir / y_col
        var_dir.mkdir(exist_ok=True)
        unconstrained_with_infra = line_graph_by_direction(
            trajectories.filter(pl.col("time_of_day").is_in(["am", "pm"])).with_columns(
                pl.col("time_of_day").replace_strict(
                    {"am": "AM Peak", "pm": "PM Peak"}, return_dtype=peak_type
                )
            ), "long_pos", y_col,
            background="infra", background_infrastructure_type="width",
            color="time_of_day", unconstrained_only=True, peak_only=False)
        unconstrained_with_infra.save(var_dir / "1_unconstrained.html")

        transformed = trajectories.with_columns(pl.when(is_dominant_am).then(pl.lit("AM Peak", dtype=peak_type))
                                                 .when(is_dominant_pm).then(pl.lit("PM Peak", dtype=peak_type))
                                                 .otherwise(pl.lit("Off-peak", dtype=peak_type)).alias("time_of_day"))
        unconstrained_vs_all = line_graph_by_direction(transformed, "long_pos", y_col,
                                                       background="mean_all",
                                                       unconstrained_only=True, peak_only=True)
        unconstrained_vs_all.save(var_dir / "2_all.html")
