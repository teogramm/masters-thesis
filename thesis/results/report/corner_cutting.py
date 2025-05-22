import polars as pl
from sklearn.model_selection import cross_val_score, train_test_split

from thesis.model.enums import Location
from sklearn.linear_model import LinearRegression
import altair as alt

from thesis.results.graph.riddarhuskajen import _filter_crossing_location, crossing_length_scatter


def graph(trajectories: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Create a scatter plot of distance travelled on the opposite lane depending on the crossing's longitudinal position.
    """
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN)
    return crossing_length_scatter(trajectories)


def crossing_linear_regression(trajectories: pl.LazyFrame) -> pl.DataFrame:
    """
    Create a linear regression model predicting distance travelled on the opposite lane for Southbound cyclists in
    Riddarhuskajen, depending on the crossing location.
    Prints information about the model and its performance.
    :param trajectories: Unfiltered trajectories.
    :return: DataFrame with the predicted and actual y values.
    """
    trajectories = trajectories.filter(pl.col("location") == Location.RIDDARHUSKAJEN,
                                       pl.col("direction") == "Southbound")
    trajectories = _filter_crossing_location(trajectories)
    trajectories = (trajectories.select("ID", "direction", "line_crossing_info", "location", "time_of_day",
                                        "constrained")
                    .explode("line_crossing_info").unnest("line_crossing_info")).unique()
    trajectories = trajectories.filter(pl.col("crossing_dist").is_not_null(),
                                       pl.col("crossing_dist").is_not_nan(),
                                       pl.col("crossing_start").is_between(20, 30))

    # Transform constrained into a boolean column
    trajectories = trajectories.with_columns(pl.col("constrained").replace_strict({
        "Unconstrained": 0
    }, default=pl.lit(1), return_dtype=pl.Int8))

    trajectories = trajectories.collect()

    y = trajectories.select("crossing_dist").to_numpy()
    x = trajectories.select("crossing_start", "constrained").to_numpy()

    regression = LinearRegression()

    print(f"Correlation {trajectories.select(pl.corr("crossing_start", "constrained", method="spearman"))}")

    score = cross_val_score(regression, x, y)
    print(f"5-fold CV mean r2: {score.mean()}, std: {score.std()}")
    
    regression = regression.fit(x, y)
    print(f"Sample size: {x.shape[0]}")
    print(f"Model: Intercept: {regression.intercept_} Crossing_long_pos: {regression.coef_[0,0]} Constrained: {regression.coef_[0,1]}")
    
    regression = LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=76)
    regression= regression.fit(x,y)

    pred = regression.predict(x_test)

    residuals = alt.Chart(pl.DataFrame({"y_test": pred, "y_actual": y_test})).mark_circle().encode(
        alt.X("y_test:Q"),
        alt.Y("y_actual:Q")
    )
    
    return residuals