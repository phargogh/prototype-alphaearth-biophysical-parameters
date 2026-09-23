#!/usr/bin/env python
"""Predict the InVEST SDR soil erodibility (K) raster from AlphaEarth embeddings.

Reads the embedding from Source Cooperative and SoilGrids 2.0 from ISRIC's file
server; neither needs an account. Reference data: EPIC K (see erodibility.py) at
SoilGrids pixel centres, thickness-averaged over the topsoil (0-30 cm by
default). A random forest learns K from the 64 embedding bands, accuracy is
estimated by spatial-block cross-validation, and K is predicted at 10 m. The
result refines the SoilGrids-based estimate spatially; it cannot be more
accurate than that reference.

Example:
    python soil_erodibility/predict_soil_erodibility.py \
        --bbox -122.30 37.37 -122.20 37.43 --output-tif outputs/soil_erodibility.tif
"""

from __future__ import annotations

import argparse

from shapely.geometry.base import BaseGeometry

import erodibility
from alphaearth_invest import pipeline, soilgrids


def build_reference(aoi: BaseGeometry, crs: str, args: argparse.Namespace) -> pipeline.ReferenceSamples:
    return soilgrids.reference_samples(
        erodibility.epic_k,
        erodibility.SOILGRIDS_PROPERTIES,
        args.depth_cm,
        aoi,
        crs,
        args.cache_dir,
        source=erodibility.reference_source(args.depth_cm, soilgrids.SOURCE),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    pipeline.add_common_arguments(parser)
    parser.add_argument(
        "--depth-cm",
        type=int,
        choices=soilgrids.INTERVAL_BOTTOMS_CM,
        default=erodibility.DEFAULT_DEPTH_CM,
        help="Average K over 0..DEPTH cm (default %(default)s).",
    )
    args = parser.parse_args(argv)
    pipeline.emit_report(pipeline.run(args, erodibility.SPEC, build_reference))


if __name__ == "__main__":
    main()
