"""Shared pytest fixtures.

Network tests read public data (Source Cooperative, ISRIC) and run only when
NETWORK_TESTS=1; otherwise they are skipped.
"""

import os

import pytest


@pytest.fixture(scope="session")
def network():
    if os.environ.get("NETWORK_TESTS") != "1":
        pytest.skip("set NETWORK_TESTS=1 to run tests that read public data over the internet")
    import rasterio

    from alphaearth_invest.embeddings import GDAL_REMOTE_OPTIONS

    with rasterio.Env(**GDAL_REMOTE_OPTIONS):
        yield
