import polars as pl
import numpy as np
from scipy.spatial import ConvexHull



def calculate_convex_hull(trajectories: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the convex hull for each longitudinal position and direction compared to the centreline.
    :param trajectories: 
    :return: Dataframe containing a list of points comprising the hull for each longitudinal position and direction
    relative to the centreline.
    """
    def calculate_hull(x: list, y: list) -> np.ndarray:
        points = np.array(list(zip(x, y)))
        convex_hull = ConvexHull(points)
        return points[convex_hull.vertices]
    
    is_east = pl.when(pl.col("lat_pos") > 0).then(True).otherwise(False).alias("is_east")
    
    # Group points in path, by longitudinal and lateral position
    trajectories = trajectories.filter(pl.col("in_path")).with_columns(is_east)
    trajectories = (trajectories.select(["is_east", "long_pos", "lat_pos", "X", "Y"]).group_by(["long_pos", "is_east"]).agg(
        pl.struct(["X","Y"]).map_batches(
            lambda g: calculate_hull(g.struct.field("X"), g.struct.field("Y"))
        ).alias("hull")
    ))
    return trajectories