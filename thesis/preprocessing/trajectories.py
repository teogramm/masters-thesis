import polars as pl

from thesis.files import BASE_PATH
from thesis.files.trajectories import TRAJECTORIES_LOCATION


def unite_trajectories() -> pl.DataFrame:
    """
    Loads the separate csv files and combines them into a single dataframe with a location column.
    :return: Combined dataframe with location column.
    """
    from pathlib import Path

    from thesis.model.enums import Location

    TRAJECTORY_PARENT_DIR = BASE_PATH.joinpath("data/trajectories/source/")

    # Add locations and unite everything into a single dataframe
    location_type = pl.Enum(Location)
    rk = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("1_Riddarhuskajen.csv"))
    rk = rk.with_columns(location=pl.lit(Location.RIDDARHUSKAJEN, dtype=location_type))

    rb_n = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("2_RiddarholmsbronN.csv"))
    rb_n = rb_n.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N, dtype=location_type))

    rb_s = pl.scan_csv(TRAJECTORY_PARENT_DIR.joinpath("3_RiddarholmsbronS.csv"))
    rb_s = rb_s.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_S, dtype=location_type))

    united = pl.concat([rk, rb_n, rb_s]).collect()
    return united


def preprocess_trajectories():
    """
    * Unites everything into a single dataset
    * Parses datetime values
    * Converts some column names to lowercase
    * Reorders columns
    * Converts to parquet format
    :return: 
    """
    # Convert direction into an enum
    from thesis.model.enums import Direction

    trajectories = unite_trajectories()

    trajectories = trajectories.with_columns(
        # Convert into datetime and add timezone information
        pl.col("Time").str.strptime(dtype=pl.Datetime).dt.replace_time_zone("Europe/Stockholm"),
        # Convert estimated into a boolean
        pl.col("Estimated").cast(pl.Boolean),
        # Round LongPos to 1 decimal
        pl.col("LongPos").round(decimals=1),
        # Convert path to boolean, 0 means inside path and 1 outside path, so add not to interpret it as in_path
        ~pl.col("Path").cast(pl.Boolean),
        # Convert direction into an enum
        pl.col("Direction").cast(Direction)
    )

    # Remove the type code, keep only type name and cast as enum type
    type_label_enum = pl.Enum(["Bicycle", "Pedestrian"])
    trajectories = trajectories.drop("Type").rename({"Type_Label": "type"}).with_columns(
        pl.col("type").cast(type_label_enum)
    )

    # Rename columns with multiple words  
    trajectories = trajectories.rename({
        "LongPos": "long_pos",
        "LatPos": "lat_pos"
    })
    # Rename path column to make it more clearer
    trajectories = trajectories.rename({"Path": "in_path"})
    # Convert some columns to lowercase
    trajectories = trajectories.rename({
        n: n.lower() for n in trajectories.columns if n not in ("X", "Y", "Xp", "Yp", "ID")
    })

    # Reorder columns based on their importance
    important_columns = ["location", "time", "ID", "direction", "long_pos", "lat_pos", "type", "X", "Y", "in_path",
                         "speed", "acc", "theta", "dist", "dist_right", "dist_left"]
    unimportant_columns = set(trajectories.columns).difference(important_columns)
    # Add the unimportant columns afterwards
    important_columns.extend(unimportant_columns)
    trajectories = trajectories.select(important_columns)

    # Write as a parquet file
    trajectories.write_parquet(TRAJECTORIES_LOCATION,
                               partition_by="location")

    return trajectories
