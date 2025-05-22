from datetime import datetime
from zoneinfo import ZoneInfo


def get_period_name(start_hour: int) -> str:
    if start_hour == 6:
        return "AM"
    elif start_hour == 12 or start_hour == 11:
        return "OP"
    elif start_hour == 16:
        return "PM"
    raise ValueError(f"Invalid start hour {start_hour}")

def off_peak(day: int) -> tuple[datetime, datetime]:
    tz = ZoneInfo("Europe/Stockholm")
    if day == 1:
        return (
            datetime(2024,10, day, 11,0, tzinfo=tz),
            datetime(2024,10, day, 12,0, tzinfo=tz),
        )
    elif day in (2,3):
        return (
            datetime(2024,10, day, 12,0, tzinfo=tz),
            datetime(2024,10, day, 13,0, tzinfo=tz),
        )
    raise ValueError(f"day {day} not supported")

def am_peak(day: int) -> tuple[datetime, datetime]:
    tz = ZoneInfo("Europe/Stockholm")
    if day == 1:
        return (
            datetime(2024,10, day, 6,45, tzinfo=tz),
            datetime(2024,10, day, 9,7, second=30, tzinfo=tz),
        )
    elif day in (2,3):
        return (
            datetime(2024,10, day, 6,46, tzinfo=tz),
            datetime(2024,10, day, 9,5, tzinfo=tz),
        )
    raise ValueError(f"day {day} not supported")

def pm_peak(day: int) -> tuple[datetime, datetime]:
    tz = ZoneInfo("Europe/Stockholm")
    if 1 <= day <= 3:
        return (
            datetime(2024,10, day, 16,0, tzinfo=tz),
            datetime(2024,10, day, 18,20, tzinfo=tz),
        )
    raise ValueError(f"day {day} not supported")

def observation_periods_day(day: int) -> list[tuple[datetime, datetime]]:
    return [am_peak(day), off_peak(day), pm_peak(day)]

def observation_periods_all() -> dict[int,list[tuple[datetime, datetime]]]:
    return {day: observation_periods_day(day) for day in (1,2,3)}