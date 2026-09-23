"""Predict an InVEST raster input from AlphaEarth embeddings, on your own machine.

One run:
  1. Copy the annual embedding over a 10 m grid covering the AOI into a local
     cache (see alphaearth_invest.embeddings).
  2. Keep the reference samples inside the AOI, draw at most --sample-size of
     them at random, and look up the embedding at each.
  3. Estimate accuracy by k-fold cross-validation over spatial blocks. Every
     block is held out exactly once, and autocorrelated neighbours never
     straddle a training and a test fold.
  4. Fit a scikit-learn random forest on all samples.
  5. Predict window by window and write a float32 GeoTIFF with nodata, in a
     metric CRS, as InVEST requires.

Each parameter script supplies its own reference samples; everything else lives
here. Random-forest predictions are averages of training values, so they never
leave the range of the reference data.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely
from shapely.geometry.base import BaseGeometry
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from alphaearth_invest import embeddings
from alphaearth_invest.aoi import aoi_from_bbox, aoi_from_vector, aoi_in_crs, utm_crs_for
from alphaearth_invest.files import atomic_output
from alphaearth_invest.grid import OutputGrid
from alphaearth_invest.invest_raster import NODATA, PIXEL_SIZE_M, ParameterSpec

CACHE_ENVIRONMENT_VARIABLE = "ALPHAEARTH_INVEST_CACHE"


@dataclass(frozen=True)
class ReferenceSamples:
    """Reference values at points, with coordinates in the output CRS."""

    x: np.ndarray
    y: np.ndarray
    value: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if not (self.x.ndim == self.y.ndim == self.value.ndim == 1):
            raise ValueError("x, y, and value must be 1-D arrays")
        if not (len(self.x) == len(self.y) == len(self.value)):
            raise ValueError("x, y, and value must have equal lengths")
        if not np.all(np.isfinite(self.value)):
            raise ValueError("reference values must be finite")

    def subset(self, keep: np.ndarray) -> ReferenceSamples:
        return ReferenceSamples(self.x[keep], self.y[keep], self.value[keep], self.source)


@dataclass(frozen=True)
class TrainingConfig:
    sample_size: int = 5000
    folds: int = 5
    block_size_m: float = 2000.0
    number_of_trees: int = 100
    seed: int = 42
    min_samples: int = 100

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.block_size_m <= 0:
            raise ValueError("block_size_m must be positive")
        if self.number_of_trees <= 0:
            raise ValueError("number_of_trees must be positive")


class InsufficientSamplesError(RuntimeError):
    """Too few reference samples or spatial blocks in the AOI to train and evaluate a model."""


@dataclass(frozen=True)
class RunResult:
    prediction_path: Path
    grid: OutputGrid
    report: dict


def new_model(config: TrainingConfig) -> RandomForestRegressor:
    """Random forest: sqrt(64) = 8 variables per split, each tree fit on half the
    samples, leaves down to one sample."""
    return RandomForestRegressor(
        n_estimators=config.number_of_trees,
        max_features="sqrt",
        max_samples=0.5,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=config.seed,
    )


def spatial_blocks(x: np.ndarray, y: np.ndarray, block_size_m: float) -> np.ndarray:
    """Integer id of the block_size_m square containing each point."""
    cells = np.stack([np.floor(x / block_size_m), np.floor(y / block_size_m)], axis=1)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    return groups.ravel()


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - observed
    sse = float(np.sum(error**2))
    sst = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(sse / len(observed))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": 1 - sse / sst,
        "observed_mean": float(observed.mean()),
        "observed_min": float(observed.min()),
        "observed_max": float(observed.max()),
        "predicted_mean": float(predicted.mean()),
    }


def cross_validate(features: np.ndarray, target: np.ndarray, groups: np.ndarray, config: TrainingConfig) -> dict:
    """Pooled out-of-fold metrics from grouped k-fold cross-validation."""
    n_blocks = len(np.unique(groups))
    if len(target) < config.min_samples or n_blocks < config.folds:
        raise InsufficientSamplesError(
            f"Got {len(target)} samples in {n_blocks} spatial blocks; need at least {config.min_samples} "
            f"samples and {config.folds} blocks. Enlarge the AOI, widen the reference data window, "
            "or reduce --block-size-m."
        )
    predicted = np.empty(len(target), dtype=np.float64)
    splitter = GroupKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    for train, test in splitter.split(features, target, groups):
        predicted[test] = new_model(config).fit(features[train], target[train]).predict(features[test])
    return {
        "validation": f"{config.folds}-fold cross-validation over {config.block_size_m:g} m spatial blocks",
        "n_samples": len(target),
        "n_blocks": n_blocks,
        **regression_metrics(target, predicted),
    }


def training_samples(
    reference: ReferenceSamples, aoi_projected: BaseGeometry, stack_path: Path, config: TrainingConfig
) -> tuple[np.ndarray, ReferenceSamples]:
    """Embedding features and reference samples for points inside the AOI that have an embedding."""
    samples = reference.subset(shapely.contains_xy(aoi_projected, reference.x, reference.y))
    if len(samples.value) > config.sample_size:
        rng = np.random.default_rng(config.seed)
        samples = samples.subset(np.sort(rng.choice(len(samples.value), config.sample_size, replace=False)))
    features, has_embedding = embeddings.read_at(stack_path, samples.x, samples.y)
    return features[has_embedding], samples.subset(has_embedding)


def predict_to_geotiff(
    model: RandomForestRegressor,
    stack_path: Path,
    aoi_projected: BaseGeometry,
    grid: OutputGrid,
    path: Path,
    band_name: str,
) -> None:
    """Predict every pixel inside the AOI that has an embedding; write NODATA elsewhere.

    The GeoTIFF is written under a temporary name and renamed when complete, so
    a failed run never leaves a partial file at ``path``.
    """
    profile = {
        "driver": "GTiff",
        "count": 1,
        "dtype": "float32",
        "nodata": NODATA,
        "crs": grid.crs,
        "transform": grid.transform,
        "width": grid.width,
        "height": grid.height,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    with atomic_output(path) as temporary:
        with rasterio.open(stack_path) as stack, rasterio.open(temporary, "w", **profile) as output:
            if (stack.transform, stack.width, stack.height) != (grid.transform, grid.width, grid.height):
                raise RuntimeError(f"{stack_path} is not on the output grid")
            output.set_band_description(1, band_name)
            for window in grid.windows():
                raw = stack.read(window=window)
                predict = (raw[0] != embeddings.RAW_NODATA) & grid.inside(aoi_projected, window)
                block = np.full(predict.shape, NODATA, dtype=np.float32)
                if predict.any():
                    block[predict] = model.predict(embeddings.dequantize(raw[:, predict].T))
                output.write(block, 1, window=window)


def default_cache_dir() -> Path:
    return Path(os.environ.get(CACHE_ENVIRONMENT_VARIABLE, Path.home() / ".cache" / "alphaearth-invest"))


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument(
        "--aoi",
        type=Path,
        help="Polygon vector file (shapefile, GeoPackage, GeoJSON) of the study area, e.g. InVEST watersheds.",
    )
    area.add_argument(
        "--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Study area in degrees."
    )
    parser.add_argument("--output-tif", type=Path, required=True, help="Write the prediction GeoTIFF here.")
    parser.add_argument("--year", type=int, default=embeddings.DEFAULT_YEAR, help="AlphaEarth embedding year (default %(default)s).")
    parser.add_argument("--crs", help="Output CRS, e.g. EPSG:32610. Default: UTM zone of the AOI centroid.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help=f"Downloaded data cache (default ${CACHE_ENVIRONMENT_VARIABLE} or %(default)s).",
    )
    defaults = TrainingConfig()
    parser.add_argument("--sample-size", type=int, default=defaults.sample_size, help="Maximum reference samples.")
    parser.add_argument("--folds", type=int, default=defaults.folds, help="Cross-validation folds.")
    parser.add_argument(
        "--block-size-m", type=float, default=defaults.block_size_m, help="Side of the spatial blocks for validation."
    )
    parser.add_argument("--trees", type=int, default=defaults.number_of_trees, help="Random-forest trees.")
    parser.add_argument("--seed", type=int, default=defaults.seed)


def run(
    args: argparse.Namespace,
    spec: ParameterSpec,
    build_reference: Callable[[BaseGeometry, str, argparse.Namespace], ReferenceSamples],
) -> RunResult:
    """Validate, train, predict, and write the GeoTIFF requested on the command line.

    ``build_reference(aoi, crs, args)`` gets the WGS 84 AOI and the output CRS
    and returns reference samples with coordinates in that CRS.
    """
    config = TrainingConfig(
        sample_size=args.sample_size,
        folds=args.folds,
        block_size_m=args.block_size_m,
        number_of_trees=args.trees,
        seed=args.seed,
    )
    aoi = aoi_from_vector(args.aoi) if args.aoi else aoi_from_bbox(*args.bbox)
    crs = args.crs or utm_crs_for(aoi)
    aoi_projected = aoi_in_crs(aoi, crs)
    grid = OutputGrid.covering(aoi_projected, crs)

    with rasterio.Env(**embeddings.GDAL_REMOTE_OPTIONS):
        stack_path = embeddings.fetch_stack(aoi, args.year, grid, args.cache_dir)
        reference = build_reference(aoi, crs, args)
    features, samples = training_samples(reference, aoi_projected, stack_path, config)
    groups = spatial_blocks(samples.x, samples.y, config.block_size_m)
    metrics = cross_validate(features, samples.value, groups, config)
    model = new_model(config).fit(features, samples.value)
    predict_to_geotiff(model, stack_path, aoi_projected, grid, args.output_tif, spec.name)

    report = {
        "parameter": spec.name,
        "units": spec.units,
        "invest_input": spec.invest_input,
        "embedding_source": embeddings.SOURCE,
        "embedding_year": args.year,
        "embedding_cache": str(stack_path),
        "reference_source": reference.source,
        "crs": crs,
        "pixel_size_m": PIXEL_SIZE_M,
        "nodata": NODATA,
        "training": asdict(config),
        "cross_validation_metrics": metrics,
        "output_tif": str(args.output_tif),
    }
    return RunResult(prediction_path=args.output_tif, grid=grid, report=report)


def emit_report(result: RunResult) -> None:
    """Print the run report and save it beside the GeoTIFF."""
    text = json.dumps(result.report, indent=2)
    print(text)
    result.prediction_path.with_suffix(".json").write_text(text + "\n")
