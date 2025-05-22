import polars as pl
from thesis.model.observations_periods import off_peak, observation_periods_all, am_peak, pm_peak


def is_off_peak(time_col_name: str) -> pl.Expr:
    expr = (
              ((pl.col(time_col_name).dt.day() == 1) & pl.col(time_col_name).is_between(*off_peak(1)))
            | ((pl.col(time_col_name).dt.day() == 2) & pl.col(time_col_name).is_between(*off_peak(2)))
            | ((pl.col(time_col_name).dt.day() == 3) & pl.col(time_col_name).is_between(*off_peak(3)))
    )
    return expr

def is_observed(time_col_name: str) -> pl.Expr:
    observed_periods = observation_periods_all()
    expr = None
    for day, periods in observed_periods.items():
        for period in periods:
            if expr is None:
                expr = ((pl.col(time_col_name).dt.day() == day) & (pl.col(time_col_name).is_between(*period)))
            else:
                expr = expr | ((pl.col(time_col_name).dt.day() == day) & (pl.col(time_col_name).is_between(*period)))
    return expr


def time_of_day_column(time_col_name: str) -> pl.Expr:
    am_periods = [am_peak(x) for x in range(1,4)]
    is_am = pl.any_horizontal(*[pl.col(time_col_name).is_between(start,end) for (start,end) in am_periods])
    pm_periods = [pm_peak(x) for x in range(1,4)]
    is_pm = pl.any_horizontal(*[pl.col(time_col_name).is_between(start,end) for (start,end) in pm_periods])
    op_periods = [off_peak(x) for x in range(1,4)]
    is_op = pl.any_horizontal(*[pl.col(time_col_name).is_between(start,end) for (start,end) in op_periods])
    time_of_day_enum = pl.Enum(["am", "pm", "between_peaks", "other"])
    return pl.when(is_am).then(pl.lit("am", dtype=time_of_day_enum)).when(
        is_pm
    ).then(
        pl.lit("pm", dtype=time_of_day_enum)
    ).when(is_op).then(pl.lit("between_peaks", dtype=time_of_day_enum)).otherwise(
        pl.lit("other", dtype=time_of_day_enum)
    ).alias("time_of_day")
