#!/usr/bin/env python
"""Predict the InVEST Annual Water Yield plant available water content (PAWC) raster from AlphaEarth embeddings.

Reads the embedding from Source Cooperative and SoilGrids 2.0 from ISRIC's file
server; neither needs an account. Reference data: Saxton & Rawls (2006) PAWC,
corrected for coarse fragments (see available_water.py), at SoilGrids pixel
centres, thickness-averaged over 0-100 cm by default. A random forest learns
PAWC from the 64 embedding bands, accuracy is estimated by spatial-block
cross-validation, and PAWC is predicted at 10 m. The result refines the
SoilGrids-based estimate spatially; it cannot be more accurate.

Example:
    python plant_available_water_content/predict_plant_available_water_content.py \
        --bbox -122.30 37.37 -122.20 37.43 --output-tif outputs/pawc.tif
"""

from __future__ import annotations

import argparse

from shapely.geometry.base import BaseGeometry

import available_water
from alphaearth_invest import pipeline, soilgrids


def build_reference(aoi: BaseGeometry, crs: str, args: argparse.Namespace) -> pipeline.ReferenceSamples:
    return soilgrids.reference_samples(
        available_water.saxton_rawls_pawc,
        available_water.SOILGRIDS_PROPERTIES,
        args.depth_cm,
        aoi,
        crs,
        args.cache_dir,
        source=available_water.reference_source(args.depth_cm, soilgrids.SOURCE),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pipeline.add_common_arguments(parser)
    parser.add_argument(
        "--depth-cm",
        type=int,
        choices=soilgrids.INTERVAL_BOTTOMS_CM,
        default=available_water.DEFAULT_DEPTH_CM,
        help="Average PAWC over 0..DEPTH cm (default %(default)s).",
    )
    args = parser.parse_args(argv)
    pipeline.emit_report(pipeline.run(args, available_water.SPEC, build_reference))


if __name__ == "__main__":
    main()
