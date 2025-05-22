import polars as pl

from thesis.files import BASE_PATH

SUMMARY_FILE = BASE_PATH.joinpath("data/trajectories/summary.parquet")

def add_summary_information(trajectories: pl.LazyFrame, with_overall_columns = False) -> pl.LazyFrame:
    """
    Add pre-calculated information about each trajectory.
    :param trajectories: 
    :param with_overall_columns: Some columns, such as speed, acceleration and theta, appear on both the file with
    the trajectories and in the summary file.
    If set to False, those columns are removed from the returned DataFrame.
    """
    summary = pl.scan_parquet(SUMMARY_FILE)
    if not with_overall_columns:
        duplicate_cols = ["theta", "type"]
        summary = summary.drop(duplicate_cols)
    return trajectories.join(summary, on=["location", "ID", "direction"], how="left", validate="m:1")