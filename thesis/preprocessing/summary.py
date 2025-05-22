import polars as pl

from thesis.files import BASE_PATH
from thesis.files.filtering import SUMMARY_FILE


def unite_summary() -> pl.DataFrame:
    """
    Load all the summary files and merge them into a single dataframe with a ``location`` column.
    """
    from thesis.model.enums import Location

    TRAJECTORY_PARENT_DIR = BASE_PATH.joinpath("data/trajectories/source/")

    # Add locations and unite everything into a single dataframe
    location_type = pl.Enum(Location)
    rk = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("1_Riddarhuskajen_summary.csv"))
    rk = rk.with_columns(location=pl.lit(Location.RIDDARHUSKAJEN, dtype=location_type))

    rb_n = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("2_RiddarholmsbronN_summary.csv"))
    rb_n = rb_n.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N, dtype=location_type))

    rb_s = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("3_RiddarholmsbronS_summary.csv"))
    rb_s = rb_s.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_S, dtype=location_type))

    united = pl.concat([rk, rb_n, rb_s]).collect()
    return united


def preprocess_summary():
    from thesis.model.enums import Direction

    summary = unite_summary()

    # Convert Type from int to string enum
    type_label_enum = pl.Enum(["Bicycle", "Pedestrian"])
    time_of_day_enum = pl.Enum(["am", "pm", "between_peaks", "other"])
    summary = summary.with_columns(
        pl.col("Type").replace_strict({
            0: "Pedestrian",
            1: "Bicycle"
        }, return_dtype=type_label_enum),
        pl.col("Direction").cast(Direction),
        pl.col("Time_Ini", "Time_End").str.strptime(dtype=pl.Datetime).dt.replace_time_zone("Europe/Stockholm"),
        pl.col("TimeOfDay").replace_strict({
            0: "am",
            1: "pm",
            2: "between_peaks",
            3: "other"
        }, return_dtype=time_of_day_enum),
        pl.col("Swap_Minor", "Swap_Major", "Full_Estimated").cast(pl.Boolean)
    ).drop(["Oversize", "F_Oversize"])
    
    # Recalculate the IssueTJ column
    summary = summary.with_columns(IssueTJ=(pl.col("Path") != 0) | (pl.col("Trip") != 0) | 
        pl.col("Swap_Minor") | pl.col("Swap_Major") | pl.col("Full_Estimated")
    )
    
    # Convert f columns to boolean
    f_columns = [c for c in summary.columns if c.startswith("F_")]
    summary = summary.with_columns(pl.col(*f_columns).cast(pl.Boolean))
    
    # Recalculate ExcludedTJ
    excludedTJ = pl.col("IssueTJ")
    for col in f_columns:
        excludedTJ = excludedTJ | pl.col(col)
    summary = summary.with_columns(ExcludedTJ=excludedTJ)

    # Rename columns
    summary = summary.rename({
        "Time_Ini": "time_start",
        "Time_End": "time_end",
        "TT": "travel_time",
        "TimeOfDay": "time_of_day",
        "Direction": "direction",
        "Type": "type",
        "ObjectLength": "object_length",
        "ObjectWidth": "object_width",
        "ObjectHeight": "object_height",
        "ObjectArea": "object_area",
        "Theta": "theta",
        "Length": "length",
        "Path": "path",
        "Trip": "trip",
        "Swap_Minor": "swap_minor",
        "Swap_Major": "swap_major",
        "Full_Estimated": "full_estimated",
        "IssueTJ": "issue",
        "F_Ped": "f_ped",
        "F_EE": "f_ee",
        "F_IT": "f_it",
        "F_Swap": "f_swap",
        "F_Estimated": "f_estimated",
        "ExcludedTJ": "excluded"
    })
    
    summary.write_parquet(SUMMARY_FILE, partition_by="location")
