#!/usr/bin/env python
"""Predict aboveground biomass carbon density (InVEST Carbon model c_above) from AlphaEarth embeddings.

Reads the embedding from Source Cooperative and GEDI L4A v2.1 footprints from
NASA's ORNL DAAC. Downloading GEDI needs a free NASA Earthdata login (see
alphaearth_invest/gedi.py); granules are cached, so later runs over the same
area and dates do not. Reference data: individual usable GEDI footprints times a
carbon fraction (see gedi_carbon.py). A random forest learns carbon density from
the 64 embedding bands, accuracy is estimated by spatial-block cross-validation,
and carbon is predicted at 10 m in Mg C/ha (= t/ha, the InVEST unit).

With --c-above-table and --lulc the script also averages the prediction within
each class of your land-cover raster (for example the LULC raster of your
InVEST run) and writes lucode and c_above columns to merge into the carbon
pools table.

Example:
    python aboveground_carbon/predict_aboveground_carbon.py \
        --bbox -122.30 37.37 -122.20 37.43 --output-tif outputs/aboveground_carbon.tif \
        --c-above-table outputs/c_above_by_lulc.csv --lulc lulc.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform
from shapely.geometry.base import BaseGeometry

import gedi_carbon
from alphaearth_invest import gedi, pipeline
from alphaearth_invest.invest_raster import NODATA


def build_reference(aoi: BaseGeometry, crs: str, args: argparse.Namespace) -> pipeline.ReferenceSamples:
    default_start, default_end = gedi_carbon.default_window(args.year)
    start, end = args.gedi_start or default_start, args.gedi_end or default_end
    granules = gedi.search(aoi.bounds, start, end)
    if not granules:
        raise ValueError(
            f"No GEDI L4A granules intersect the AOI between {start} and {end}. "
            "GEDI covers about 51.6 S to 51.6 N from April 2019 to July 2025."
        )
    datasets = (gedi_carbon.BIOMASS, gedi_carbon.QUALITY_FLAG, gedi_carbon.DEGRADE_FLAG)
    longitudes, latitudes, carbon = [], [], []
    for path in gedi.granule_files(granules, args.cache_dir / "gedi"):
        shots = gedi.read_shots(path, aoi.bounds, datasets)
        usable, values = gedi_carbon.quality_carbon(shots, args.carbon_fraction)
        longitudes.append(shots[gedi.LONGITUDE][usable])
        latitudes.append(shots[gedi.LATITUDE][usable])
        carbon.append(values)
    x, y = transform("EPSG:4326", crs, np.concatenate(longitudes), np.concatenate(latitudes))
    return pipeline.ReferenceSamples(
        x=np.asarray(x),
        y=np.asarray(y),
        value=np.concatenate(carbon),
        source=gedi_carbon.reference_source(f"{gedi.PRODUCT}, individual footprints", start, end, args.carbon_fraction),
    )


def c_above_by_lulc(prediction_path: Path, lulc_path: Path) -> list[dict]:
    """Mean predicted carbon density and pixel count for each land-cover code.

    The land-cover raster is resampled onto the prediction grid by nearest
    neighbour. Pixels that are nodata in either raster, or outside the land-cover
    extent, are ignored.
    """
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with rasterio.open(prediction_path) as prediction, rasterio.open(lulc_path) as lulc:
        with WarpedVRT(
            lulc,
            crs=prediction.crs,
            transform=prediction.transform,
            width=prediction.width,
            height=prediction.height,
            resampling=Resampling.nearest,
            add_alpha=True,
        ) as lulc_on_grid:
            for _, window in prediction.block_windows(1):
                carbon = prediction.read(1, window=window)
                codes = lulc_on_grid.read(1, window=window)
                valid = (carbon != NODATA) & (lulc_on_grid.read_masks(1, window=window) > 0)
                if not valid.any():
                    continue
                unique_codes, inverse = np.unique(codes[valid], return_inverse=True)
                block_sums = np.bincount(inverse, weights=carbon[valid].astype(np.float64))
                block_counts = np.bincount(inverse)
                for code, total, count in zip(unique_codes.tolist(), block_sums, block_counts):
                    sums[int(code)] = sums.get(int(code), 0.0) + float(total)
                    counts[int(code)] = counts.get(int(code), 0) + int(count)
    return [{"lucode": code, "c_above": sums[code] / counts[code], "pixel_count": counts[code]} for code in sums]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pipeline.add_common_arguments(parser)
    parser.add_argument(
        "--carbon-fraction",
        type=gedi_carbon.carbon_fraction,
        default=gedi_carbon.DEFAULT_CARBON_FRACTION,
        help="Carbon fraction of dry biomass (default %(default)s).",
    )
    parser.add_argument("--gedi-start", help="First GEDI date, YYYY-MM-DD (default: Jan 1 of --year).")
    parser.add_argument("--gedi-end", help="GEDI end date, exclusive, YYYY-MM-DD (default: Jan 1 after --year).")
    parser.add_argument("--c-above-table", type=Path, help="Also write mean c_above per LULC class to this CSV.")
    parser.add_argument("--lulc", type=Path, help="Land-cover raster for --c-above-table, e.g. your InVEST LULC.")
    args = parser.parse_args(argv)
    if args.c_above_table and not args.lulc:
        parser.error("--c-above-table requires --lulc")

    result = pipeline.run(args, gedi_carbon.SPEC, build_reference)
    if args.c_above_table:
        gedi_carbon.write_table(c_above_by_lulc(result.prediction_path, args.lulc), args.c_above_table)
        result.report["c_above_table"] = {"path": str(args.c_above_table), "lulc": str(args.lulc)}
    pipeline.emit_report(result)


if __name__ == "__main__":
    main()
