import polars as pl
from thesis.files.trajectories import open_trajectories
from thesis.files import BASE_PATH

results_per_datapoint_file = BASE_PATH.joinpath("data/trajectories/results_datapoint.parquet")
results_per_id_file = BASE_PATH.joinpath("data/trajectories/results_id.parquet")


def create_results_per_datapoint() -> None:
    trajectories = open_trajectories().select(["location", "time", "ID"])
    trajectories.write_parquet(results_per_datapoint_file)


def create_results_per_id_file() -> None:
    trajectories = open_trajectories().select(["location", "ID"]).unique()
    trajectories.write_parquet(results_per_id_file)


def save_results_per_id(new_results: pl.DataFrame, replace=False) -> None:
    """
    Add a new column to the results per ID file.
    :param new_results: DataFrame containing the ``ID`` and ``location`` columns (to uniquely identify each
    trajectory) and one more column that will be added.
    :param replace: Whether to replace the new column if it already exists. 
    """
    columns = set(new_results.columns)
    idx_cols = {'ID', 'location'}
    new_col = columns - idx_cols
    if len(columns) != 3 or len(new_col) != 1:
        raise ValueError("The results dataframe must have the ID and location columns and one more column.")
    new_col = new_col.pop()

    results = open_results_per_id().collect()
    if new_col in results.columns and not replace:
        print(f"Column {new_col} already exists. If you want to overwrite it, set replace to True")
        return
    elif new_col in results.columns:
        results = results.drop(new_col)
    results = results.join(new_results, how="left", on=idx_cols, validate="1:1")
    results.write_parquet(results_per_id_file)


def save_results_per_datapoint(new_results: pl.DataFrame, replace=False) -> None:
    """
    Add a new column to the results per datapoint file.
    :param new_results: DataFrame containing the ``ID``, ``time``, and ``location`` columns (to uniquely identify each
    datapoint) and one more column that will be added.
    :param replace: Whether to replace the new column if it already exists. 
    """
    columns = set(new_results.columns)
    idx_cols = {'ID', 'time', 'location'}
    new_col = columns - idx_cols
    if len(columns) != 4 or len(new_col) != 1:
        raise ValueError("The results dataframe must have the ID, time and location columns and one more column.")
    new_col = new_col.pop()

    results = open_results_per_datapoint().collect()
    if new_col in results.columns and not replace:
        print(f"Column {new_col} already exists. If you want to overwrite it, set replace to True")
        return
    elif new_col in results.columns:
        results = results.drop(new_col)
    results = results.join(new_results, how="left", on=idx_cols, validate="1:1")
    results.write_parquet(results_per_datapoint_file)


def add_results_per_datapoint(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    results = open_results_per_datapoint()
    return trajectories.join(results, on=["location", "time", "ID"], how="left")


def add_results_per_id(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    results = open_results_per_id()
    trajectories = trajectories.join(results, on=["location", "ID"], how="left")
    return trajectories


def open_results_per_datapoint() -> pl.LazyFrame:
    """
    Open the file containing additional columns for each datapoint.
    Each datapoint is uniquely defined by the ``location``, ``time`` and ``ID`` combination.
    :return: Lazy
    """
    return pl.scan_parquet(results_per_datapoint_file)


def open_results_per_id() -> pl.LazyFrame:
    """
    Open the file containing additional columns for each trajectory ``location``, ``ID``
    :return: 
    """
    return pl.scan_parquet(results_per_id_file)
