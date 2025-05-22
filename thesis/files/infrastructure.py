import polars as pl

from thesis.files import BASE_PATH
from thesis.model.enums import Location

_infrastructure_dir = BASE_PATH.joinpath("data/infrastructure")

def open_width() -> pl.DataFrame:
    width_rb_n = pl.read_csv(_infrastructure_dir.joinpath("width_riddarholmsbron_n.csv"))
    width_rb_s = pl.read_csv(_infrastructure_dir.joinpath("width_riddarholmsbron_s.csv"))
    # Join them into a single df with location column
    width_rb_n = width_rb_n.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N, dtype=Location))
    width_rb_s = width_rb_s.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_S, dtype=Location))
    return pl.concat([width_rb_n, width_rb_s])

def open_elevation() -> pl.DataFrame:
    elevation_rb_n = pl.read_csv(_infrastructure_dir.joinpath("elevation_riddarholmsbron_n.csv"))
    elevation_rb_n = elevation_rb_n.with_columns(location=pl.lit(Location.RIDDARHOLMSBRON_N, dtype=Location))
    return elevation_rb_n

def open_curvature() -> pl.DataFrame:
    curvature = pl.read_csv(_infrastructure_dir.joinpath("curvature_riddarhuskajen.csv"))
    curvature = curvature.with_columns(pl.col("location").cast(Location))
    return curvature