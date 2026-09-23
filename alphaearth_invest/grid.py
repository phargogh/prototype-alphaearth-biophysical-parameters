"""The 10 m output grid shared by the embedding stack and the prediction raster."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry.base import BaseGeometry

from alphaearth_invest.invest_raster import PIXEL_SIZE_M


@dataclass(frozen=True)
class OutputGrid:
    """A north-up grid in ``crs`` with PIXEL_SIZE_M pixels.

    Pixel edges fall on multiples of the pixel size, which is also how the
    AlphaEarth UTM tiles are laid out, so in the tiles' own UTM zone the
    embedding is copied pixel for pixel with no resampling.
    """

    crs: str
    transform: Affine
    width: int
    height: int

    @classmethod
    def covering(cls, aoi_projected: BaseGeometry, crs: str, pixel_size: float = PIXEL_SIZE_M) -> OutputGrid:
        """Smallest aligned grid containing an AOI already projected to ``crs``."""
        minx, miny, maxx, maxy = aoi_projected.bounds
        west = math.floor(minx / pixel_size) * pixel_size
        south = math.floor(miny / pixel_size) * pixel_size
        east = math.ceil(maxx / pixel_size) * pixel_size
        north = math.ceil(maxy / pixel_size) * pixel_size
        width = round((east - west) / pixel_size)
        height = round((north - south) / pixel_size)
        if width == 0 or height == 0:
            raise ValueError("AOI is smaller than one pixel")
        return cls(crs, Affine(pixel_size, 0, west, 0, -pixel_size, north), width, height)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        west, north = self.transform.c, self.transform.f
        return west, north + self.height * self.transform.e, west + self.width * self.transform.a, north

    @property
    def key(self) -> str:
        """Stable short id of the grid, for cache file names."""
        text = f"{self.crs}|{tuple(self.transform)[:6]}|{self.width}|{self.height}"
        return hashlib.sha1(text.encode()).hexdigest()[:12]

    def windows(self, size: int = 512) -> Iterator[Window]:
        for row in range(0, self.height, size):
            for col in range(0, self.width, size):
                yield Window(col, row, min(size, self.width - col), min(size, self.height - row))

    def inside(self, aoi_projected: BaseGeometry, window: Window) -> np.ndarray:
        """Boolean array for ``window``: True where the pixel centre lies inside the AOI."""
        return geometry_mask(
            [aoi_projected],
            out_shape=(int(window.height), int(window.width)),
            transform=window_transform(window, self.transform),
            invert=True,
        )
