import sys

import polars as pl


def add_sample_size_to_colour(trajectories: pl.DataFrame, color_col: str) -> pl.DataFrame:
    """
    Sample size is per colour for every facet
    :param trajectories: 
    :param color_col: Column used to put the sample size on
    :return: 
    """
    # Calculate the average N over each group and round it
    trajectories = trajectories.with_columns(pl.col("N").mean().round().cast(pl.Int32).over(color_col).alias("N_group"))

    # If there are big variations in sample sizes between longitudinal positions, put out a warning
    if trajectories.select((pl.col("N").std().over(color_col) >
                            (pl.col("N").mean().over(color_col) * 0.15)).any()).item():
        print("Warning: large sample size deviations in graph", file=sys.stderr)

    trajectories = trajectories.with_columns(
        pl.format("{} ({})", pl.col(color_col), pl.col("N_group")).cast(str).alias(color_col)
    )

    return trajectories

def calculate_group_statistics(grouped_trajectories, y_col: str) -> pl.DataFrame:
    """
    Calculate Mean and Standard Deviation for each group of trajectories.
    :param grouped_trajectories: Already grouped trajectories.
    :param y_col: Name of the column for which statistics are calculated.
    :return: DataFrame with the grouping columns and ``Measure`` and ``<y_col>`` columns.
    """
    grouped_trajectories: pl.DataFrame | pl.LazyFrame = grouped_trajectories.agg(
        pl.col(y_col).mean().alias("Mean"),
        pl.col(y_col).std().alias("Standard Deviation"),
        pl.len().alias("N")
    )

    idx = list(set(grouped_trajectories.collect_schema().keys()) - {"Standard Deviation", "Mean"})

    if type(grouped_trajectories) is pl.LazyFrame:
        grouped_trajectories = grouped_trajectories.collect()

    grouped_trajectories = grouped_trajectories.unpivot(
        on=["Standard Deviation", "Mean"],
        index=idx,
        value_name=y_col,
        variable_name="Measure"
    )

    return grouped_trajectories