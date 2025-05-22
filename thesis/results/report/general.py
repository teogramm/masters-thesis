from pathlib import Path
from typing import Literal

import polars as pl
import altair as alt

from thesis.filtering.filters import PathFilter, TripFilter
from thesis.model.enums import Primary_Type
from thesis.model.exprs import is_off_peak


def export_flow(trajectories: pl.LazyFrame, directory: Path):
    """
    Save the flow graphs from the general section.
    """
    flow_over_day(trajectories).save(directory / "flow_over_day.svg")
    flow_over_day_by_direction(trajectories.filter(pl.col("time").dt.day() == 1)).save(directory / "flow_over_day_by_direction.svg")
    am = flow_rush_hour(trajectories.filter(pl.col("time").dt.day() == 1), rush_hour="am")
    pm = flow_rush_hour(trajectories.filter(pl.col("time").dt.day() == 1), rush_hour="pm")
    ((am | pm).resolve_scale(y="shared")
     .configure_axis(labelFontSize=20,titleFontSize=20)
     .configure_legend(labelFontSize=20, titleFontSize=20).save(directory / "flow_rush_hours.svg"))
    

def flow_over_day(trajectories: pl.LazyFrame, hour_from=5, hour_to=24) -> alt.Chart:
    """
    Displays the number of hourly crossings between the given hours disaggregated by day.
    """
    # For each trajectory keep the hour of its first point
    grouped = trajectories.group_by("ID").agg(
        pl.col("time").first().dt.day().alias("Day"),
        pl.col("time").first().dt.hour().alias("Hour")
    )
    grouped = grouped.filter(pl.col("Hour").is_between(hour_from, hour_to))
    grouped = grouped.group_by("Day", "Hour").agg(
        pl.len().alias("N")
    )
    grouped = grouped.collect()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(field="Hour", type="ordinal"),
        alt.XOffset(field="Day", type="ordinal"),
        alt.Y(field="N", type="quantitative").title("Number of trajectories"),
        alt.Color(field="Day", type="ordinal", legend=alt.Legend(orient="top-right", fillColor="white",
                                                                 strokeColor="black", padding=2))
    ).configure_axis(labelFontSize=22,titleFontSize=22).configure_legend(labelFontSize=22, titleFontSize=22)
    return chart

def flow_over_day_by_direction(trajectories: pl.LazyFrame, hour_from=5, hour_to=24) -> alt.Chart:
    """
    Display the 
    :param trajectories: 
    :param hour_from: 
    :param hour_to: 
    :return: 
    """
    # For each trajectory keep the hour of its first point
    grouped = trajectories.group_by("ID").agg(
        pl.col("direction").first().alias("Direction"),
        pl.col("time").first().dt.hour().alias("Hour")
    )
    grouped = grouped.filter(pl.col("Hour").is_between(hour_from, hour_to))
    grouped = grouped.group_by("Hour", "Direction").agg(
        pl.len().alias("N")
    )
    grouped = grouped.collect()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(field="Hour", type="ordinal"),
        alt.XOffset(field="Direction", type="nominal"),
        alt.Y(field="N", type="quantitative").title("Number of trajectories"),
        alt.Color(field="Direction", type="nominal", legend=alt.Legend(orient="top-right", fillColor="white",
                                                                       strokeColor="black", padding=2))
    ).configure_axis(labelFontSize=22,titleFontSize=22).configure_legend(labelFontSize=22, titleFontSize=22)
    return chart

def flow_rush_hour(trajectories: pl.LazyFrame, rush_hour: Literal["am", "pm"]) -> alt.Chart:
    trajectories = trajectories.filter(pl.col("time_of_day") == rush_hour)
    grouped = trajectories.group_by("ID").agg(
        pl.all().first()
    )
    grouped = grouped.sort("time")
    grouped = grouped.group_by_dynamic("time", every="15m", group_by=[pl.col("time").dt.day().alias("Day"), "direction"]).agg(
        pl.col("ID").n_unique().alias("N")
    )
    # Transform time and date to string
    grouped = grouped.with_columns(pl.col("time").dt.strftime("%H:%M"))
    grouped = grouped.collect()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(field="time", type="ordinal"),
        alt.XOffset(field="direction", type="nominal"),
        alt.Y(field="N", type="quantitative"),
        alt.Color(field="direction", type="nominal")
    )
    return chart

def traffic_composition(observations: pl.LazyFrame) -> alt.Chart:
    """
    Graph the traffic composition for each of the three days.
    """

    filtered_types = ["Other"]
    observations = observations.filter(~pl.col("primary_type").is_in(filtered_types))


    grouped = observations.group_by(pl.col("observation_time").dt.day().alias("Day"), pl.col("primary_type")).agg(
        pl.len().alias("N")
    )

    grouped = grouped.with_columns(percentage=(pl.col("N"))/(pl.col("N").sum().over("Day")))

    grouped = grouped.collect()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(field="N", type="quantitative").stack("normalize").title("Percentage of traffic"),
        alt.Y(field="Day", type="ordinal"),
        alt.Color(field="primary_type", type="nominal").title("Type"),
    )
    
    text = chart.mark_text().encode(
        alt.ColorValue('white'),
        alt.Text("percentage:Q", format='.2')
    )
    
    return chart

def traffic_composition_peaks(observations: pl.LazyFrame) -> alt.Chart:
    """
    Graph the traffic composition for peak hours disaggregated by peak and off-peak traffic.
    """
    # trajectories = trajectories.filter(pl.col("primary_type").is_not_null())
    
    filtered_types = ["Other"]
    observations = observations.filter(~pl.col("primary_type").is_in(filtered_types))
    
    observations = observations.with_columns(~is_off_peak("observation_time").alias("is_on_peak"))
    
    grouped = observations.group_by(pl.col("observation_time").dt.day().alias("Day"), pl.col("primary_type"),
                                    pl.col("is_on_peak")).agg(
        pl.len().alias("N")
    )
    
    grouped = grouped.with_columns(pl.col("is_on_peak").replace_strict({
        True: "Peak",
        False: "Off-Peak"
    }))
    
    grouped = grouped.with_columns(percentage=(pl.col("N"))/(pl.col("N").sum().over("Day", "is_on_peak")))
    
    grouped = grouped.collect()
    chart = alt.Chart(grouped).mark_bar().encode(
        alt.X(field="N", type="quantitative").stack("normalize").title("Percentage of traffic"),
        alt.Y(field="is_on_peak", type="nominal").title(""),
        alt.Color(field="primary_type", type="nominal").title("Type"),
        alt.Row(field="Day", type="ordinal")
    )
    return chart
    
def trajectories(trajectories: pl.LazyFrame) -> pl.DataFrame:
    """
    Calculate statistics for the trajectories.
    """
    # Since errors are calculated per-trajectory, for each trajectory keep only one row
    stats = trajectories.group_by("location", "ID").agg(
        pl.all().first()
    )
    stats = stats.group_by("location").agg(
        pl.col("ID").n_unique().alias("Total trajectories"),
        (pl.col("swap_minor") | pl.col("swap_major")).sum().alias("Swapping"),
        (pl.col("path") == PathFilter.OUTSIDE.value).sum().alias("Completely outside path"),
        ((pl.col("trip") == TripFilter.INCOMPLETE.value) | (pl.col("trip") == TripFilter.COMPLETE_ABNORMAL_EXIT.value)).sum().alias("Abnormal entry or exit"),
        ((pl.col("trip") == TripFilter.COMPLETE.value).sum()).alias("Completely inside path"),
        (pl.col("f_ped") | pl.col("f_ee") |
         pl.col("f_it") | pl.col("f_swap") | pl.col("f_estimated")).sum().alias("Trajectories near other problematic trajectories"),
        pl.col("excluded").sum().alias("Total excluded")
    )
    return stats.collect()

def unmatched_by_type(matches: pl.LazyFrame) -> alt.TopLevelMixin:
    """
    Graph the composition of annotations which were not assigned to a trajectory.
    """
    unmatched = matches.filter(pl.col("ID").is_null(),
                               pl.col("primary_type") != "Other")
    unmatched = unmatched.group_by("primary_type").agg(
        pl.len().alias("N")
    )
    
    unmatched = unmatched.with_columns(pl.col("primary_type").cast(Primary_Type))
    unmatched = unmatched.sort("primary_type").collect()
    
    chart = alt.Chart(unmatched).mark_bar().encode(
        alt.X("N:Q").stack("normalize").title("Percentage"),
        alt.Color("primary_type", type="nominal").title("Type").legend(alt.Legend(direction="horizontal",
                                                                                  orient="top"))
    )
    
    return chart