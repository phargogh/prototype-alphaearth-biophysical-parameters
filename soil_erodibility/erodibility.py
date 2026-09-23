"""Soil erodibility K for InVEST SDR.

K is computed from SoilGrids texture and organic carbon with the EPIC equation
(Sharpley & Williams 1990) exactly as written in the InVEST SDR user guide, and
converted from US customary units to t*ha*h/(ha*MJ*mm) by multiplying by 0.1317.
Some SWAT documentation prints the sand coefficient as -0.256; this follows the
InVEST guide's -0.0256.
"""

import numpy as np

from alphaearth_invest.invest_raster import ParameterSpec

SPEC = ParameterSpec(
    name="soil_erodibility",
    units="t*ha*h/(ha*MJ*mm)",
    invest_input="SDR: soil erodibility (K) raster",
)
DEFAULT_DEPTH_CM = 30  # topsoil
US_CUSTOMARY_TO_SI = 0.1317

# Argument of epic_k -> SoilGrids property that feeds it.
SOILGRIDS_PROPERTIES = {"sand": "sand", "silt": "silt", "clay": "clay", "organic_carbon": "soc"}


def epic_k(sand, silt, clay, organic_carbon):
    """EPIC K in t*ha*h/(ha*MJ*mm) from sand, silt, clay, and organic carbon in percent by weight."""
    sn1 = 1 - sand / 100
    a = 0.2 + 0.3 * np.exp(-0.0256 * sand * (1 - silt / 100))
    b = (silt / (clay + silt)) ** 0.3
    c = 1 - 0.25 * organic_carbon / (organic_carbon + np.exp(3.72 - 2.95 * organic_carbon))
    d = 1 - 0.7 * sn1 / (sn1 + np.exp(-5.51 + 22.9 * sn1))
    return a * b * c * d * US_CUSTOMARY_TO_SI


def reference_source(depth_cm: int, soilgrids_source: str) -> str:
    return f"EPIC K (InVEST SDR guide form, x0.1317) from {soilgrids_source}, 0-{depth_cm} cm"
