from thesis.files import BASE_PATH

import polars as pl

from thesis.model.enums import Location

_CROSSING_TIMES_BASE = BASE_PATH.joinpath("data/trajectories/matching/")
_CROSSING_TIMES_FILE = _CROSSING_TIMES_BASE.joinpath("crossing_times.parquet")

def _open_crossings() -> pl.DataFrame:
    crossings = pl.read_parquet(_CROSSING_TIMES_FILE)
    return crossings

def open_crossing_times_riddarhuskajen() -> pl.DataFrame:
    crossings = _open_crossings()
    return crossings.filter(pl.col("location") == Location.RIDDARHUSKAJEN)
    
def open_crossing_times_riddarholmsbron_n() -> pl.DataFrame:
    crossings = _open_crossings()
    return crossings.filter(pl.col("location") == Location.RIDDARHOLMSBRON_N)