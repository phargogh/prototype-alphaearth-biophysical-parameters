"""SoilGrids 2.0 from ISRIC's public file server (no account needed).

Each property and depth is a VRT over Cloud-Optimized GeoTIFF tiles on a 250 m
Interrupted Goode Homolosine grid, in mapped-unit Int16 with nodata -32768.
Each VRT is a ~6 MB XML index of its tiles and is slow to fetch, so it is cached
locally with the tile paths made absolute; only the tiles covering the AOI are
then read over HTTP.

Dividing mapped units by MAPPED_PER_CONVENTIONAL gives the units pedotransfer
functions expect (ISRIC SoilGrids FAQ):
  sand, silt, clay   g/kg      / 10    -> percent by weight
  soc                dg/kg     / 100   -> percent by weight
  cfvo               cm3/dm3   / 1000  -> volumetric fraction

Pedotransfer functions are nonlinear, so they are evaluated once per SoilGrids
depth interval and combined with :func:`depth_weights` rather than averaging
texture first. Reference samples are the centres of SoilGrids pixels, one per
pixel, so a coarse reference pixel is never counted twice.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform, transform_bounds
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from alphaearth_invest.files import atomic_output, fetch_bytes
from alphaearth_invest.pipeline import ReferenceSamples

DEPTH_INTERVALS_CM: tuple[tuple[int, int], ...] = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200))
INTERVAL_BOTTOMS_CM = tuple(bottom for _, bottom in DEPTH_INTERVALS_CM)
MAPPED_PER_CONVENTIONAL = {"sand": 10, "silt": 10, "clay": 10, "soc": 100, "cfvo": 1000}
DATA_URL = "https://files.isric.org/soilgrids/latest/data/{property}/"
SOURCE = "ISRIC SoilGrids 2.0 (250 m, CC-BY 4.0) via files.isric.org"

Interval = tuple[int, int]
Pedotransfer = Callable[..., np.ndarray]  # keyword arrays in conventional units -> values


def layer_name(soil_property: str, interval: Interval) -> str:
    """SoilGrids layer id, e.g. ``sand_0-5cm_mean``."""
    if soil_property not in MAPPED_PER_CONVENTIONAL:
        raise ValueError(f"unsupported SoilGrids property {soil_property!r}")
    top, bottom = interval
    return f"{soil_property}_{top}-{bottom}cm_mean"


def depth_weights(depth_cm: int) -> list[tuple[Interval, float]]:
    """Intervals covering 0..depth_cm with their thickness weights (summing to 1).

    ``depth_cm`` must be an interval boundary (5, 15, 30, 60, 100, 200) so no
    interval is split.
    """
    if depth_cm not in INTERVAL_BOTTOMS_CM:
        raise ValueError(f"depth_cm must be one of {INTERVAL_BOTTOMS_CM}, got {depth_cm}")
    return [
        (interval, (interval[1] - interval[0]) / depth_cm)
        for interval in DEPTH_INTERVALS_CM
        if interval[1] <= depth_cm
    ]


def cached_vrt(soil_property: str, interval: Interval, cache_dir: Path) -> Path:
    """Local copy of one layer's VRT whose tile paths point at files.isric.org."""
    layer = layer_name(soil_property, interval)
    path = cache_dir / "soilgrids" / f"{layer}.vrt"
    if path.exists():
        return path
    base = DATA_URL.format(property=soil_property)
    root = ElementTree.fromstring(fetch_bytes(f"{base}{layer}.vrt"))
    for source in root.iter("SourceFilename"):
        if source.get("relativeToVRT") == "1":
            source.text = f"/vsicurl/{base}{source.text.removeprefix('./')}"
            source.set("relativeToVRT", "0")
    with atomic_output(path) as temporary:
        ElementTree.ElementTree(root).write(temporary, encoding="utf-8")
    return path


def _covering_window(bounds: tuple[float, float, float, float], grid_transform) -> Window:
    """Smallest whole-pixel window containing ``bounds`` on a north-up grid."""
    west, south, east, north = bounds
    col_start, row_start = ~grid_transform * (west, north)
    col_stop, row_stop = ~grid_transform * (east, south)
    col_start, row_start = math.floor(col_start), math.floor(row_start)
    return Window(col_start, row_start, math.ceil(col_stop) - col_start, math.ceil(row_stop) - row_start)


def _read_window(vrt: Path, soil_property: str, aoi: BaseGeometry):
    """One layer in conventional units over the AOI's bounding window, as a masked array."""
    with rasterio.open(vrt) as layer:
        bounds = transform_bounds("EPSG:4326", layer.crs, *aoi.bounds, densify_pts=21)
        window = _covering_window(bounds, layer.transform)
        data = layer.read(1, window=window, masked=True)
        return data / MAPPED_PER_CONVENTIONAL[soil_property], layer.window_transform(window), layer.crs


def reference_samples(
    pedotransfer: Pedotransfer,
    properties: Mapping[str, str],
    depth_cm: int,
    aoi: BaseGeometry,
    crs: str,
    cache_dir: Path,
    source: str,
) -> ReferenceSamples:
    """Thickness-weighted ``pedotransfer`` over 0..depth_cm at SoilGrids pixel centres.

    ``properties`` maps each keyword argument of ``pedotransfer`` to the SoilGrids
    property that feeds it. ``aoi`` is in WGS 84; sample coordinates are returned
    in ``crs``. Samples cover the AOI's bounding window in the SoilGrids
    projection, so some lie outside the AOI; the pipeline keeps only those
    inside. Pixels where any input layer has no data are dropped. Must run
    inside ``rasterio.Env(**alphaearth_invest.embeddings.GDAL_REMOTE_OPTIONS)``.
    """
    weights = depth_weights(depth_cm)
    layers = [(soil_property, interval) for interval, _ in weights for soil_property in properties.values()]
    with ThreadPoolExecutor(max_workers=8) as executor:
        vrts = dict(zip(layers, executor.map(lambda layer: cached_vrt(*layer, cache_dir), layers)))

    total = None
    grid = None  # (transform, shape, crs) of the first layer read; all layers must share it
    for interval, weight in weights:
        inputs = {}
        for name, soil_property in properties.items():
            values, window_transform, layer_crs = _read_window(vrts[soil_property, interval], soil_property, aoi)
            if grid is None:
                grid = (window_transform, values.shape, layer_crs)
            elif (window_transform, values.shape) != grid[:2]:
                raise RuntimeError("SoilGrids layers are not on one grid; cannot combine them")
            inputs[name] = values
        mask = np.logical_or.reduce([np.ma.getmaskarray(values) for values in inputs.values()])
        with np.errstate(invalid="ignore", divide="ignore"):  # nodata pixels are NaN and masked below
            evaluated = np.asarray(pedotransfer(**{name: values.filled(np.nan) for name, values in inputs.items()}))
        weighted = np.ma.masked_array(evaluated * weight, mask=mask | ~np.isfinite(evaluated))
        total = weighted if total is None else total + weighted

    grid_transform, _, soil_crs = grid
    rows, cols = np.nonzero(~np.ma.getmaskarray(total))
    xs, ys = rasterio.transform.xy(grid_transform, rows, cols, offset="center")
    x, y = transform(soil_crs, crs, np.asarray(xs), np.asarray(ys))
    return ReferenceSamples(x=np.asarray(x), y=np.asarray(y), value=total.data[rows, cols], source=source)
