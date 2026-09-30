"""Saxton & Rawls PAWC must match an independent transcription of the equations."""

import pytest

import available_water as aw


def saxton_rawls_reference(sand_pct, clay_pct, organic_matter_pct, coarse_fraction=0.0):
    s, c, om = sand_pct / 100, clay_pct / 100, organic_matter_pct
    t1500 = -0.024 * s + 0.487 * c + 0.006 * om + 0.005 * s * om - 0.013 * c * om + 0.068 * s * c + 0.031
    t33 = -0.251 * s + 0.195 * c + 0.011 * om + 0.006 * s * om - 0.027 * c * om + 0.452 * s * c + 0.299
    wilting_point = t1500 + (0.14 * t1500 - 0.02)
    field_capacity = t33 + (1.283 * t33**2 - 0.374 * t33 - 0.015)
    return (field_capacity - wilting_point) * (1 - coarse_fraction)


def pawc(sand, clay, organic_matter, coarse=0.0):
    return float(aw.saxton_rawls_pawc(sand, clay, organic_matter / aw.ORGANIC_MATTER_PER_ORGANIC_CARBON, coarse))


@pytest.mark.parametrize("soil", [(40, 20, 2.5, 0.0), (80, 5, 1.0, 0.1), (10, 45, 3.0, 0.0), (30, 30, 5.0, 0.3)])
def test_matches_published_equation(soil):
    assert pawc(*soil) == pytest.approx(saxton_rawls_reference(*soil), rel=1e-9)


def test_loam_matches_saxton_rawls_table_3():
    # Saxton & Rawls (2006) Table 3, loam (40 % sand, 20 % clay, 2.5 % OM): PAW 0.14.
    assert pawc(40, 20, 2.5) == pytest.approx(0.14, abs=0.005)


def test_organic_matter_is_capped_at_calibration_limit():
    capped = float(aw.saxton_rawls_pawc(40, 20, 10.0, 0.0))  # 10 % OC = 17 % OM
    assert capped == pytest.approx(pawc(40, 20, aw.MAX_ORGANIC_MATTER_PERCENT))


def test_coarse_fragments_scale_available_water():
    assert pawc(40, 20, 2.5, coarse=0.25) == pytest.approx(0.75 * pawc(40, 20, 2.5))


def test_soilgrids_inputs_name_the_function_arguments():
    assert list(aw.SOILGRIDS_PROPERTIES) == ["sand", "clay", "organic_carbon", "coarse_fraction"]
    aw.saxton_rawls_pawc(**{name: 0.1 for name in aw.SOILGRIDS_PROPERTIES})
