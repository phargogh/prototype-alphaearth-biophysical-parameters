import argparse

import numpy as np
import pytest

import gedi_carbon


@pytest.mark.parametrize("text", ["0", "1", "-0.2", "1.5"])
def test_carbon_fraction_must_be_a_proper_fraction(text):
    with pytest.raises(argparse.ArgumentTypeError):
        gedi_carbon.carbon_fraction(text)


def test_c_above_table_has_invest_column_names_sorted_by_code(tmp_path):
    path = tmp_path / "c_above.csv"
    gedi_carbon.write_table(
        [{"lucode": 30, "c_above": 1.5, "pixel_count": 4}, {"lucode": 10, "c_above": 55.5, "pixel_count": 12}], path
    )
    assert path.read_text().splitlines() == ["lucode,c_above,pixel_count", "10,55.5,12", "30,1.5,4"]


def test_default_gedi_window_is_the_embedding_year():
    assert gedi_carbon.default_window(2024) == ("2024-01-01", "2025-01-01")


def test_quality_filter_keeps_good_shots_and_converts_to_carbon():
    shots = {
        "agbd": np.array([100.0, 100.0, 100.0, -9999.0]),
        "l4_quality_flag": np.array([1, 0, 1, 1]),
        "degrade_flag": np.array([0, 0, 3, 0]),
    }
    usable, carbon = gedi_carbon.quality_carbon(shots, 0.47)
    np.testing.assert_array_equal(usable, [True, False, False, False])
    np.testing.assert_allclose(carbon, [47.0])
