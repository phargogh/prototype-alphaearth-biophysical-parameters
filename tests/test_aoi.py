import geopandas
import pytest
import shapely

from alphaearth_invest.aoi import aoi_from_bbox, aoi_from_vector, utm_crs_for


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (-122.24, 37.40, "EPSG:32610"),  # Jasper Ridge, California
        (36.82, -1.29, "EPSG:32737"),  # Nairobi, southern hemisphere
        (-180.0, 10.0, "EPSG:32601"),  # western edge of zone 1
        (180.0, 10.0, "EPSG:32660"),  # antimeridian stays in zone 60
        (3.0, 0.0, "EPSG:32631"),  # equator counts as northern hemisphere
    ],
)
def test_utm_crs_for_centroid(lon, lat, expected):
    assert utm_crs_for(shapely.Point(lon, lat)) == expected


def test_bbox_rejects_inverted_or_out_of_range_coordinates():
    with pytest.raises(ValueError):
        aoi_from_bbox(10, 0, 5, 1)
    with pytest.raises(ValueError):
        aoi_from_bbox(0, 1, 1, 0)
    with pytest.raises(ValueError):
        aoi_from_bbox(0, -91, 1, 0)


def test_vector_aoi_is_dissolved_and_reprojected_to_wgs84(tmp_path):
    # Two adjacent 1 km squares in UTM 10N near Jasper Ridge.
    squares = [shapely.box(560000, 4139000, 561000, 4140000), shapely.box(561000, 4139000, 562000, 4140000)]
    path = tmp_path / "watersheds.gpkg"
    geopandas.GeoDataFrame(geometry=squares, crs="EPSG:32610").to_file(path)

    aoi = aoi_from_vector(path)

    assert aoi.geom_type == "Polygon"
    west, south, east, north = aoi.bounds
    assert -122.4 < west < east < -122.2
    assert 37.3 < south < north < 37.5
    assert utm_crs_for(aoi) == "EPSG:32610"


def test_vector_aoi_requires_polygons(tmp_path):
    path = tmp_path / "points.gpkg"
    geopandas.GeoDataFrame(geometry=[shapely.Point(0, 0)], crs="EPSG:4326").to_file(path)
    with pytest.raises(ValueError, match="polygon"):
        aoi_from_vector(path)


def test_aoi_in_crs_keeps_lon_lat_edges_straight():
    from alphaearth_invest.aoi import aoi_in_crs

    box = aoi_from_bbox(-122.30, 37.37, -122.20, 37.43)
    projected = aoi_in_crs(box, "EPSG:32610")
    # Densified to 0.001 degree segments: the 0.1 x 0.06 degree box gets ~320 vertices.
    assert len(projected.exterior.coords) > 300
    assert projected.area == pytest.approx(59.2e6, rel=0.01)  # about 8.8 km x 6.7 km
