from typing import Literal

import polars as pl
from enum import Enum

Direction = pl.Enum(["Northbound", "Southbound"])

# Ensure Back > Front, since dist_right 
Relative_Position = pl.Enum(["Front", "Back"])

Primary_Type = pl.Enum(["Regular", "Electric", "Cargo", "Folding", "Moped", "Scooter", "Other", "Pedestrian", "Race"])
Primary_Type_Literal = Literal["Regular", "Electric", "Cargo", "Folding", "Moped", "Scooter", "Other", "Pedestrian", "Race"]

Secondary_Type = pl.Enum(["Pedal", "Electric", "Women's bicycle"])

class Location(str, Enum):
    RIDDARHUSKAJEN = "Riddarhuskajen"
    RIDDARHOLMSBRON_N = "Riddarholmsbron_n"
    RIDDARHOLMSBRON_S = "Riddarholmsbron_s"