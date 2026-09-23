"""Opt-in checks against the live public data sources (NETWORK_TESTS=1)."""

import numpy as np
import pytest
import rasterio

from alphaearth_invest import embeddings
from alphaearth_invest.aoi import aoi_from_bbox, aoi_in_crs
from alphaearth_invest.embeddings import RAW_NODATA, dequantize
from alphaearth_invest.grid import OutputGrid

CRS = "EPSG:32610"
SMALL_AOI = aoi_from_bbox(-122.25, 37.40, -122.247, 37.402)  # about 260 m x 220 m near Jasper Ridge


@pytest.fixture(scope="module")
def stack(network, tmp_path_factory):
    grid = OutputGrid.covering(aoi_in_crs(SMALL_AOI, CRS), CRS)
    return grid, embeddings.fetch_stack(SMALL_AOI, 2024, grid, tmp_path_factory.mktemp("cache"))


@pytest.mark.network
def test_embedding_stack_is_on_the_grid_with_unit_length_vectors(stack):
    grid, path = stack
    with rasterio.open(path) as dataset:
        assert (dataset.transform, dataset.width, dataset.height) == (grid.transform, grid.width, grid.height)
        raw = dataset.read()
    assert (raw[0] != RAW_NODATA).all()
    norms = np.linalg.norm(dequantize(raw.reshape(64, -1).T), axis=1)
    np.testing.assert_allclose(norms, 1, atol=0.02)
