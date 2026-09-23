import numpy as np
import pytest

from alphaearth_invest.embeddings import dequantize


def test_dequantize_matches_published_rule():
    raw = np.array([-127, -64, 0, 1, 60, 127], dtype=np.int8)
    expected = np.sign(raw) * (raw / 127.5) ** 2
    result = dequantize(raw)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert np.all(np.diff(result) > 0)  # monotonic in the stored value


def test_dequantize_matches_a_published_pixel_value():
    # A00 at EPSG:32610 (560005, 4140005), 2024: stored 60, float value 0.2215.
    assert float(dequantize(np.array([60], dtype=np.int8))[0]) == pytest.approx(0.2215, abs=5e-5)
