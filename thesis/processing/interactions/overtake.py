import polars as pl


def calculate_overtake_info(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    next_to_or_ahead_nb = (pl.col("direction") == "Northbound") & (pl.col("long_pos") >= pl.col("long_pos_other"))
    next_to_or_ahead_sb = (pl.col("direction") == "Southbound") & (pl.col("long_pos") <= pl.col("long_pos_other"))
    next_to_or_ahead_of_other = next_to_or_ahead_sb | next_to_or_ahead_nb

    # For each trajectory get information about the trajectories it overtakes
    overtakes = trajectories.explode("overtakes_id").select("location", "ID", "time", "overtakes_id", "long_pos",
                                                            "primary_type", "speed", "lat_pos")
    overtakes = overtakes.join(trajectories, left_on=["location", "overtakes_id", "time"],
                               right_on=["location", "ID", "time"], how="inner", suffix="_other")
    # Keep the first observation where the bike goes next to or ahead of the other
    overtakes = (overtakes.filter(next_to_or_ahead_of_other).sort("location", "ID", "time")
    .group_by(["location", "ID", "overtakes_id"]).agg(
            (pl.col("speed") - pl.col("speed_other")).first().alias("overtake_speed_diff"),
            ((pl.col("lat_pos") - pl.col("lat_pos_other")).first().abs().alias("overtake_lat_pos_diff")),
            pl.col("primary_type_other").first().alias("overtake_type_other"),
            pl.col("long_pos").first().alias("overtake_long_pos")
    ))

    overtakes = overtakes.group_by("location", "ID").agg(
        pl.struct( "overtake_speed_diff", "overtake_lat_pos_diff", "overtake_type_other",
                  "overtake_long_pos", pl.col("overtakes_id").first().alias("overtake_id")).alias("overtake_info")
    )

    return overtakes

def calculate_overtakes(trajectories: pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculate which trajectories are overtaking.
    :param trajectories: 
    :return: DataFrame with ``location``, ``ID`` and ``overtakes_id`` columns,
    mapping each trajectory ID to a list of IDs that it overtook.
    """
    other_is_ahead_northbound = (pl.col("direction") == "Northbound") & (pl.col("long_pos") < pl.col("long_pos_other"))
    other_is_ahead_southbound = (pl.col("direction") == "Southbound") & (pl.col("long_pos") > pl.col("long_pos_other"))
    other_is_ahead = other_is_ahead_northbound | other_is_ahead_southbound

    overtakes = trajectories.select(["location", "ID", "time", "long_pos", "direction"])
    # Find all the trajectories that are on the path at the same time
    overtakes = overtakes.join(overtakes, how="inner", on=["location", "time", "direction"], suffix="_other")
    # Remove entries for the same ID at the same time
    overtakes = overtakes.filter(pl.col("ID") != pl.col("ID_other"))
    # We now have a dataframe containing for each datapoint all the other datapoints that appear at the same time
    # on the path.
    # For each instance where two trajectories share the path, keep any timestamp of when the other trajectory was ahead
    # and when it was behind.
    overtakes = overtakes.group_by(["location", "ID", "ID_other"]).agg(
        pl.col("time").filter(other_is_ahead).first().alias("other_time_ahead"),
        pl.col("time").filter(~other_is_ahead).first().alias("other_time_behind")
    )
    # If an overtake happened there will be a time when the other trajectory was behind and a time when the other
    # trajectory was in front.
    overtakes = overtakes.filter(pl.col("other_time_ahead").is_not_null() & pl.col("other_time_behind").is_not_null())
    # For trajectory x: if trajectory y appears first ahead and then behind, that means
    # that an overtake was done by trajectory x.
    did_overtake = pl.col("other_time_behind") > pl.col("other_time_ahead")
    got_overtaken = pl.col("other_time_ahead") > pl.col("other_time_behind")
    # If trajectory y appears first behind and then ahead, that means that an overtake was done by trajectory y.
    overtakes = overtakes.with_columns(overtake_rel=
                                       pl.when(did_overtake).then(pl.lit("did_overtake", dtype=pl.Categorical))
                                       .when(got_overtaken).then(pl.lit("got_overtaken", dtype=pl.Categorical)))
    # Keep only information about the trajectories doing the overtake
    overtakes = (overtakes.filter(pl.col("overtake_rel") == "did_overtake")
                 .select("location", "ID", pl.col("ID_other").alias("overtakes_id")))
    # For each ID doing an overtake return 
    overtakes = overtakes.group_by("location", "ID").agg(pl.col("overtakes_id"))
    return overtakes
