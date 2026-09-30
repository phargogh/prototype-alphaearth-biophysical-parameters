"""AlphaEarth Foundations annual satellite embedding (v1), read from public Cloud-Optimized GeoTIFFs.

Source Cooperative hosts the embedding as 8192 x 8192 pixel GeoTIFFs in UTM
zones, free to read without an account
(s3://us-west-2.opendata.source.coop/tge-labs/aef). The files store signed 8-bit
integers; :func:`dequantize` converts them to embedding values ("divide by
127.5; square; multiply by the sign of the original value", AlphaEarth GCS
readme). -128 marks a masked pixel and always appears in all 64 bands together.
Each GeoTIFF is stored bottom-up, so it is read through its companion warped
VRT, which corrects the orientation. A GeoParquet index lists every file with
its footprint.

:func:`fetch_stack` copies the embedding over an OutputGrid into one local
64-band int8 GeoTIFF. The copy is cached by year and grid, so predicting several
parameters for the same AOI downloads the embedding once.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import geopandas
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from alphaearth_invest.files import atomic_output, download_once
from alphaearth_invest.grid import OutputGrid

BANDS = tuple(f"A{index:02d}" for index in range(64))
DEFAULT_YEAR = 2024
RAW_NODATA = -128
LICENSE = "AlphaEarth Foundations Satellite Embedding, Google and Google DeepMind, CC-BY 4.0"
INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
SOURCE = "Source Cooperative mirror (s3://us-west-2.opendata.source.coop/tge-labs/aef)"

# GDAL settings for anonymous reads of the public bucket (its name contains dots,
# so path-style addressing is required) and of other public HTTP rasters.
GDAL_REMOTE_OPTIONS = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_REGION": "us-west-2",
    "AWS_VIRTUAL_HOSTING": "FALSE",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}


def dequantize(raw: np.ndarray) -> np.ndarray:
    """Map stored int8 values to embedding values in [-1, 1] as float32.

    The caller must drop RAW_NODATA pixels first; they have no embedding value.
    """
    values = raw.astype(np.float32)
    return np.sign(values) * (values / 127.5) ** 2


def tile_vrts(index_path: Path, aoi: BaseGeometry, year: int) -> list[str]:
    """GDAL paths of the warped VRTs of every tile for ``year`` that intersects the WGS 84 AOI."""
    index = geopandas.read_parquet(index_path, columns=["path", "year", "geom"], filters=[("year", "==", year)])
    tiles = index[index.intersects(aoi)]
    return [path.replace("s3://", "/vsis3/", 1).removesuffix(".tiff") + ".vrt" for path in tiles["path"]]


def fetch_stack(aoi: BaseGeometry, year: int, grid: OutputGrid, cache_dir: Path) -> Path:
    """Return a local int8 GeoTIFF of the raw embedding on ``grid``, downloading it if needed.

    Pixels with no embedding hold RAW_NODATA in all 64 bands. Must run inside
    ``rasterio.Env(**GDAL_REMOTE_OPTIONS)``.
    """
    path = cache_dir / "embeddings" / f"aef_{year}_{grid.key}.tif"
    if path.exists():
        return path
    index_path = download_once(INDEX_URL, cache_dir / "aef_index.parquet")
    vrt_paths = tile_vrts(index_path, aoi, year)
    if not vrt_paths:
        raise ValueError(f"No AlphaEarth embedding tiles for {year} intersect the AOI")

    profile = {
        "driver": "GTiff",
        "count": len(BANDS),
        "dtype": "int8",
        "nodata": RAW_NODATA,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "interleave": "pixel",
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    with atomic_output(path) as temporary, contextlib.ExitStack() as stack:
        tiles = [stack.enter_context(_open_on_grid(vrt_path, grid)) for vrt_path in vrt_paths]
        destination = stack.enter_context(rasterio.open(temporary, "w", **profile))
        destination.descriptions = BANDS
        destination.update_tags(year=str(year), source=SOURCE, license=LICENSE)
        for window in grid.windows():
            destination.write(_mosaic_window(tiles, grid, window), window=window)
    return path


@contextlib.contextmanager
def _open_on_grid(vrt_path: str, grid: OutputGrid) -> Iterator[tuple[WarpedVRT, tuple[float, ...]]]:
    """A tile warped onto the grid (nearest neighbour), with its bounds in the grid CRS."""
    with rasterio.open(vrt_path) as tile:
        bounds = transform_bounds(tile.crs, grid.crs, *tile.bounds)
        with WarpedVRT(
            tile,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=Resampling.nearest,
            nodata=RAW_NODATA,
        ) as warped:
            yield warped, bounds


def _mosaic_window(tiles: list[tuple[WarpedVRT, tuple[float, ...]]], grid: OutputGrid, window: Window) -> np.ndarray:
    """Raw embedding for one window; where tiles overlap, the first tile with data wins."""
    block = np.full((len(BANDS), int(window.height), int(window.width)), RAW_NODATA, dtype=np.int8)
    window_bounds = rasterio.windows.bounds(window, grid.transform)
    for warped, bounds in tiles:
        if not _overlaps(window_bounds, bounds):
            continue
        incoming = warped.read(window=window)
        fill = (block[0] == RAW_NODATA) & (incoming[0] != RAW_NODATA)
        block[:, fill] = incoming[:, fill]
    return block


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def read_at(stack_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Embedding values (n, 64) at points in the stack's CRS (NaN where none), and a mask of points with one."""
    with rasterio.open(stack_path) as stack:
        raw = np.array(list(stack.sample(zip(x, y), masked=False)), dtype=np.int8).reshape(len(x), len(BANDS))
    valid = raw[:, 0] != RAW_NODATA
    values = np.full(raw.shape, np.nan, dtype=np.float32)
    values[valid] = dequantize(raw[valid])
    return values, valid
