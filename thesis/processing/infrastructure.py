import polars as pl


def calculate_width(trajectories: pl.LazyFrame, width: pl.DataFrame) -> pl.LazyFrame:
    """
    Calculates the width of the path for each datapoint.
    :return: DataFrame with ``location``, ``ID``, ``time`` and ``width`` columns.
    """
    return _calculate_feature(trajectories, width, "width")


def calculate_elevation(trajectories: pl.LazyFrame, elevation: pl.DataFrame) -> pl.LazyFrame:
    """
    Calculates the gradient of the path for each datapoint.
    :return: DataFrame with ``location``, ``ID``, ``time`` and ``gradient`` columns.
    """
    return _calculate_feature(trajectories, elevation, "elevation")


def _calculate_feature(trajectories: pl.LazyFrame, feature: pl.DataFrame, feature_name: str) -> pl.LazyFrame:
    """
    Calculates the values of the given feature for each datapoint, interpolating for sections where the feature value
    changes from start to end.
    :param trajectories: DataFrame with ``location``, ``ID``, ``time`` and ``long_pos`` columns.
    :param feature: DataFrame with ``long_pos_start``, ``long_pos_end``,``<feature_name>_m_start`` and 
    ``feature_name_m_end`` columns.
    :param feature_name: Name of the feature.
    :return: DataFrame with ``location``, ``ID``, ``time`` and ``<feature_name>`` columns.
    """
    feature = feature.lazy()

    trajectories = trajectories.join(feature, on=["location"], how="full").filter(
        pl.col("long_pos").is_between(pl.col("from_long_pos"), pl.col("to_long_pos"), closed="left")
    )
    
    # Interpolate linearly between values when the width is variable
    feature = trajectories.with_columns(linear_interpolation(feature_name).round(2).alias(feature_name))
    
    # Keep only relevant columns
    feature = feature.select("location", "ID", "time", feature_name)
    
    return feature

def linear_interpolation(feature_name: str) -> pl.Expr:
    """
    Interpolate linearly between longitudinal position and infrastructure feature values according to the formula 
    
    :math:`y = (y_0 (x_1-x) + y_1*(x-x_0)/(x_1 - x_0)`.
    :param feature_name: Feature name as it appears in the ``m_start`` and ``m_end`` columns
    """
    x0 = pl.col("from_long_pos")
    x1 = pl.col("to_long_pos")
    y0 = pl.col(f"{feature_name}_m_start")
    y1 = pl.col(f"{feature_name}_m_end")
    x = pl.col("long_pos")
    return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)
