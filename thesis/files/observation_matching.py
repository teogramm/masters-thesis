import polars as pl

from thesis.files import BASE_PATH

_MATCHING_BASE_DIR = BASE_PATH.joinpath("data/trajectories/matching/")
_MATCHES_FILE = _MATCHING_BASE_DIR.joinpath("matches.parquet")


def open_all_matches() -> pl.LazyFrame:
    matches = pl.scan_parquet(_MATCHES_FILE)
    return matches


def add_matches(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    matches = open_all_matches()
    trajectories = trajectories.join(matches.filter(pl.col("ID").is_not_null()),
                                     on=["location", "ID", "direction"], how="left", validate="m:1")
    return trajectories
