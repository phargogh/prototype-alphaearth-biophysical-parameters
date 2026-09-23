"""Study-area (AOI) handling and output CRS selection.

InVEST requires raster inputs in a projected coordinate system with linear units
in meters, so every prediction is exported in a metric CRS. By default that CRS
is the WGS 84 / UTM zone containing the AOI centroid.

This module makes no network calls, so it can be tested offline.
"""

from __future__ import annotations

from pathlib import Path

import geopandas
import shapely
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"


def aoi_from_bbox(west: float, south: float, east: float, north: float) -> BaseGeometry:
    """Return a WGS 84 rectangle; raise ValueError for an empty or out-of-range box."""
    if not (-180 <= west < east <= 180):
        raise ValueError(f"Invalid longitudes: west={west}, east={east}")
    if not (-90 <= south < north <= 90):
        raise ValueError(f"Invalid latitudes: south={south}, north={north}")
    return shapely.box(west, south, east, north)


def aoi_from_vector(path: Path) -> BaseGeometry:
    """Dissolve all features of a vector file into one WGS 84 (multi)polygon.

    Accepts any format GDAL/OGR reads (shapefile, GeoPackage, GeoJSON), e.g. the
    watersheds layer of an InVEST run. Raises ValueError if the file has no CRS,
    no features, or no polygonal area.
    """
    frame = geopandas.read_file(path)
    if frame.empty:
        raise ValueError(f"{path} contains no features")
    if frame.crs is None:
        raise ValueError(f"{path} has no coordinate reference system")
    geometry = frame.to_crs(WGS84).union_all()
    if geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"{path} must contain polygon features, found {geometry.geom_type}")
    return geometry


def aoi_in_crs(geometry: BaseGeometry, crs: str) -> BaseGeometry:
    """Project a WGS 84 AOI to ``crs``.

    Edges are first densified to 0.001 degrees so they stay straight in
    longitude/latitude, as a --bbox or a geographic AOI file describes them.
    """
    densified = shapely.segmentize(geometry, max_segment_length=0.001)
    return geopandas.GeoSeries([densified], crs=WGS84).to_crs(crs).iloc[0]


def utm_crs_for(geometry: BaseGeometry) -> str:
    """Return the EPSG code of the standard UTM zone containing the geometry's centroid.

    Uses the regular 6-degree zones; the Norway/Svalbard zone exceptions are ignored.
    """
    centroid = geometry.centroid
    zone = min(int((centroid.x + 180) // 6) + 1, 60)
    base = 32600 if centroid.y >= 0 else 32700
    return f"EPSG:{base + zone}"
