import polars as pl

from thesis.files import BASE_PATH


OBSERVATIONS_PROCESSED_FILE_PATH = BASE_PATH.joinpath("data/trajectories/matching/observations-proc.parquet")

def open_processed_observations(with_pedestrians=False) -> pl.DataFrame:
    observations = pl.read_parquet(OBSERVATIONS_PROCESSED_FILE_PATH)
    if not with_pedestrians:
        observations = observations.filter(~(pl.col("primary_type") == "Pedestrian"))
    return observations
