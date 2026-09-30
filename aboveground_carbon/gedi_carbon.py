"""Aboveground carbon from GEDI L4A: the reference-data rules.

Reference values are GEDI L4A version 2.1 footprint aboveground biomass density
(Mg/ha) from usable shots (l4_quality_flag = 1, degrade_flag = 0, non-negative
biomass), times the carbon fraction of dry biomass (default 0.47, the IPCC 2006
default; the InVEST guide cites 0.43-0.51). GEDI observes only between about
51.6 S and 51.6 N, from April 2019 to July 2025.

The InVEST Carbon model takes carbon per land-cover class, so the predicted
raster can also be averaged per class into a table with lucode and c_above
columns, to merge into the InVEST carbon pools table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from alphaearth_invest.invest_raster import ParameterSpec

SPEC = ParameterSpec(
    name="aboveground_carbon",
    units="Mg C/ha",
    invest_input="Carbon: c_above column of the carbon pools table (averaged by LULC class)",
)
DEFAULT_CARBON_FRACTION = 0.47
BIOMASS = "agbd"  # Mg/ha
QUALITY_FLAG = "l4_quality_flag"  # keep 1
DEGRADE_FLAG = "degrade_flag"  # keep 0
TABLE_COLUMNS = ("lucode", "c_above", "pixel_count")


def default_window(year: int) -> tuple[str, str]:
    """GEDI dates matching the embedding year: [Jan 1, next Jan 1)."""
    return f"{year}-01-01", f"{year + 1}-01-01"


def carbon_fraction(text: str) -> float:
    """argparse type: a carbon fraction strictly between 0 and 1."""
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("carbon fraction must be between 0 and 1")
    return value


def quality_carbon(shots: dict[str, np.ndarray], carbon_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask of usable shots and their carbon density (Mg C/ha).

    Usable: l4_quality_flag = 1, degrade_flag = 0, and non-negative biomass
    (which also excludes any negative fill values).
    """
    biomass = shots[BIOMASS]
    usable = (shots[QUALITY_FLAG] == 1) & (shots[DEGRADE_FLAG] == 0) & (biomass >= 0)
    return usable, biomass[usable] * carbon_fraction


def reference_source(product: str, start: str, end: str, fraction: float) -> str:
    return (
        f"GEDI L4A v2.1 AGBD ({product}), {start} to {end}, {QUALITY_FLAG}=1, {DEGRADE_FLAG}=0, agbd>=0, "
        f"x carbon fraction {fraction}"
    )


def write_table(rows: list[dict], path: Path) -> None:
    """Write per-class c_above rows as CSV, sorted by lucode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["lucode"]))
