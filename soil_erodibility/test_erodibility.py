"""EPIC K must match an independent transcription of the InVEST guide equation."""

import math

import numpy as np
import pytest

import erodibility

TEXTURES = [
    (40, 40, 20, 1.5),  # loam
    (85, 10, 5, 0.5),  # loamy sand
    (10, 70, 20, 2.0),  # silt loam
    (20, 20, 60, 3.0),  # clay
]


def epic_k_reference(san, sil, cla, oc):
    sn1 = 1 - san / 100
    a = 0.2 + 0.3 * math.exp(-0.0256 * san * (1 - sil / 100))
    b = (sil / (cla + sil)) ** 0.3
    c = 1.0 - 0.25 * oc / (oc + math.exp(3.72 - 2.95 * oc))
    d = 1.0 - 0.7 * sn1 / (sn1 + math.exp(-5.51 + 22.9 * sn1))
    return a * b * c * d * 0.1317


@pytest.mark.parametrize("texture", TEXTURES)
def test_matches_published_equation(texture):
    value = float(erodibility.epic_k(*texture))
    assert value == pytest.approx(epic_k_reference(*texture), rel=1e-12)
    assert 0.009 <= value <= 0.092  # InVEST guide's typical SI range


def test_loam_erodibility_value():
    assert float(erodibility.epic_k(40, 40, 20, 1.5)) == pytest.approx(0.0343, abs=5e-4)


def test_evaluates_arrays_elementwise():
    columns = [np.array(column, dtype=float) for column in zip(*TEXTURES)]
    np.testing.assert_allclose(erodibility.epic_k(*columns), [epic_k_reference(*t) for t in TEXTURES])


def test_soilgrids_inputs_name_the_function_arguments():
    assert list(erodibility.SOILGRIDS_PROPERTIES) == ["sand", "silt", "clay", "organic_carbon"]
    erodibility.epic_k(**{name: 30.0 for name in erodibility.SOILGRIDS_PROPERTIES})
