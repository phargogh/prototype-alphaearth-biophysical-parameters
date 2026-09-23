"""Contract for the rasters the prediction scripts produce.

Every prediction is a single-band float32 GeoTIFF at the embedding's 10 m
resolution, in a projected CRS with metre units (InVEST requires one), with
NODATA marking pixels outside the AOI or without embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

NODATA = -9999.0
PIXEL_SIZE_M = 10


@dataclass(frozen=True)
class ParameterSpec:
    """What is being predicted, for file names and the run report."""

    name: str  # output band and file stem, e.g. "soil_erodibility"
    units: str
    invest_input: str  # the InVEST model input this layer serves
