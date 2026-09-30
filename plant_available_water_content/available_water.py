"""Plant available water content for InVEST Annual Water Yield.

PAWC = field capacity (-33 kPa) minus permanent wilting point (-1500 kPa), from
the Saxton & Rawls (2006) equations 1-2, reduced by the coarse-fragment volume
(bulk-soil basis). Output is a volumetric fraction (0-1), as InVEST expects.

Assumptions: organic matter = 1.724 x organic carbon (van Bemmelen factor),
capped at 8 %, the upper limit of the Saxton & Rawls calibration data (the SPAW
tool applies the same cap); no density or salinity adjustment.
"""

import numpy as np

from alphaearth_invest.invest_raster import ParameterSpec

SPEC = ParameterSpec(
    name="plant_available_water_content",
    units="volumetric fraction (0-1)",
    invest_input="Annual Water Yield: plant available water content raster",
)
DEFAULT_DEPTH_CM = 100  # typical root zone
ORGANIC_MATTER_PER_ORGANIC_CARBON = 1.724
MAX_ORGANIC_MATTER_PERCENT = 8.0

# Argument of saxton_rawls_pawc -> SoilGrids property that feeds it.
SOILGRIDS_PROPERTIES = {"sand": "sand", "clay": "clay", "organic_carbon": "soc", "coarse_fraction": "cfvo"}


def saxton_rawls_pawc(sand, clay, organic_carbon, coarse_fraction):
    """Bulk-soil PAWC from sand, clay, and organic carbon (percent by weight) and coarse-fragment volume fraction."""
    s, c = sand / 100, clay / 100
    om = np.minimum(organic_carbon * ORGANIC_MATTER_PER_ORGANIC_CARBON, MAX_ORGANIC_MATTER_PERCENT)
    t1500 = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * (s * om) - 0.013 * (c * om) + 0.068 * (s * c) + 0.031
    t33 = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * (s * om) - 0.027 * (c * om) + 0.452 * (s * c) + 0.299
    wilting_point = t1500 + (0.14 * t1500 - 0.02)
    field_capacity = t33 + (1.283 * t33**2 - 0.374 * t33 - 0.015)
    return (field_capacity - wilting_point) * (1 - coarse_fraction)


def reference_source(depth_cm: int, soilgrids_source: str) -> str:
    return f"Saxton & Rawls (2006) PAWC, coarse-fragment corrected, from {soilgrids_source}, 0-{depth_cm} cm"
