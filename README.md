# AlphaEarth → InVEST biophysical parameters (prototype)

Scripts that predict InVEST biophysical input layers from the Google AlphaEarth
Foundations satellite embedding (64 bands, 10 m, annual, 2017 onward). They run
on your own machine and read public files; no Google Earth Engine account is
involved.

This is a **prototype**. It shows that the embeddings can reproduce each reference
dataset. It is not a validated production mapping method (see Limitations).

## Parameters

| Folder | InVEST model | Layer | Units | Reference data the model learns from |
|---|---|---|---|---|
| [aboveground_carbon](aboveground_carbon/) | Carbon Storage and Sequestration | aboveground biomass carbon (`c_above`) | Mg C/ha | GEDI L4A footprint biomass × carbon fraction |
| [soil_erodibility](soil_erodibility/) | Sediment Delivery Ratio (SDR) | soil erodibility K | t·ha·h/(ha·MJ·mm) | EPIC K equation applied to SoilGrids 2.0 |
| [plant_available_water_content](plant_available_water_content/) | Annual Water Yield | plant available water content (PAWC) | fraction 0–1 | Saxton & Rawls (2006) equations applied to SoilGrids 2.0 |

**Why these three.** Carbon, SDR, and Annual Water Yield are among the most widely
used InVEST models. Each needs a biophysical layer that is usually expensive to map
locally and that relates to what the embeddings capture: vegetation structure,
land cover, terrain, and soil surface conditions. Two other layers were considered
and left out:

- **LULC and DEM** are not biophysical parameters. LULC is the key the parameter
  tables hang off, and a DEM is measured directly.
- **Precipitation and reference ET** vary smoothly at climate scales. Gridded
  climate products (WorldClim, CHIRPS, CGIAR ET0) map them directly, so
  predicting them from embeddings would only add error.

## Data sources

| Data | Source | Account |
|---|---|---|
| AlphaEarth embedding | [Source Cooperative](https://source.coop/tge-labs/aef) Cloud-Optimized GeoTIFFs | none |
| SoilGrids 2.0 | [ISRIC file server](https://files.isric.org/soilgrids/latest/data/) | none |
| GEDI L4A v2.1 | [NASA ORNL DAAC](https://doi.org/10.3334/ORNLDAAC/2056), via earthaccess | free NASA Earthdata login to download |
| Land cover for the carbon table | your own raster, such as the LULC of your InVEST run | none |

## How it works

Each script follows the same steps, shared in
[alphaearth_invest/pipeline.py](alphaearth_invest/pipeline.py):

1. Copy the embedding for `--year` onto a 10 m grid covering the study area (AOI),
   in the AOI's UTM zone or in `--crs`.
2. Sample the reference data inside the AOI and pair each sample with its embedding.
3. Estimate accuracy with 5-fold cross-validation over 2 km spatial blocks.
   Every block is held out once, and autocorrelated neighbouring samples never
   sit on both sides of a train/test split.
4. Fit a scikit-learn random forest on all samples (100 trees by default).
5. Predict every 10 m pixel in the AOI.

The output is a Float32 GeoTIFF with nodata −9999 in a metric CRS, as InVEST
requires. A JSON run report with the metrics, reference source, and settings is
saved next to it.

The science for each parameter lives in its folder:

- [erodibility.py](soil_erodibility/erodibility.py) holds the EPIC K formula.
- [available_water.py](plant_available_water_content/available_water.py) holds the Saxton & Rawls PAWC formula.
- [gedi_carbon.py](aboveground_carbon/gedi_carbon.py) holds the GEDI quality rule and carbon fraction.

## Setup

Install the environment into this repository with mamba:

```bash
mamba env create --prefix ./.mamba-env --file environment.yml --yes
```

The environment also installs the shared `alphaearth_invest` package in editable mode.

The soil scripts need no account. The carbon script downloads GEDI granules from
NASA, which needs a free [NASA Earthdata login](https://urs.earthdata.nasa.gov).
Provide it through the `EARTHDATA_TOKEN` environment variable, through
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or through a
`urs.earthdata.nasa.gov` entry in `~/.netrc`.

Downloads are cached in `~/.cache/alphaearth-invest`. Change the location with
`--cache-dir` or the `ALPHAEARTH_INVEST_CACHE` environment variable. The cache holds:

- the embedding for each study area, year, and grid, reused by all three scripts;
- the SoilGrids tile indexes;
- the GEDI granules. With them cached, later carbon runs need no login.

## Usage

Pass the study area as `--aoi` with any polygon vector file, such as the InVEST
watersheds shapefile, or as `--bbox`.

```bash
./.mamba-env/bin/python soil_erodibility/predict_soil_erodibility.py --aoi watersheds.shp --output-tif outputs/soil_erodibility.tif
```

Run any script with `--help` for all options. Useful ones are `--year`, `--crs`,
`--sample-size`, `--folds`, `--block-size-m`, `--trees`, and `--cache-dir`.

InVEST aligns and resamples its raster inputs to a common grid. If you need to
match a coarser DEM grid yourself, average the 10 m prediction rather than using
nearest neighbour:

```bash
gdalwarp -r average -tr 30 30 soil_erodibility.tif soil_erodibility_30m.tif
```

## Tests

```bash
./.mamba-env/bin/python -m pytest
```

The default run is offline. Tests that read Source Cooperative and ISRIC run
with `NETWORK_TESTS=1`:

```bash
NETWORK_TESTS=1 ./.mamba-env/bin/python -m pytest
```

The tests check each formula against an independent transcription and against
published values. They also cover:

- the embedding dequantization, and unit-length vectors in downloaded data;
- the GEDI quality filter, HDF5 reading, and cached downloads;
- depth weighting and spatial cross-validation;
- prediction masking and the per-class carbon table;
- atomic output files, so failed runs leave no partial files.

## Verification run

All three scripts were run end to end on a 59 km² box around Jasper Ridge,
California (`--bbox -122.30 37.37 -122.20 37.43`, year 2024, default settings).
Each wrote an 891 × 674 pixel GeoTIFF on the 10 m UTM 10N grid, with nodata set.

| Layer | Samples | Spatial blocks | Cross-validated R² | RMSE |
|---|---|---|---|---|
| Aboveground carbon | 2505 | 23 | 0.72 | 64 Mg C/ha |
| Soil erodibility K | 879 | 21 | 0.47 | 0.0024 |
| PAWC, 0–100 cm | 879 | 21 | 0.52 | 0.0093 |

Run times:

| Step | Time |
|---|---|
| First embedding download for the box (33 MB) | 60 s |
| First SoilGrids run with an empty cache | 84 s |
| Soil runs with a warm cache | 12–21 s |
| Carbon run with GEDI cached | 5 s |

**Limits of this verification.**

- **GEDI download not run.** This machine has no NASA Earthdata login, so the
  download ran only in tests with a stand-in downloader. The carbon run read
  cached copies of the nine 2024 granules over the box, cut to the box in NASA's
  HDF5 layout. They were built earlier during development from Google's Earth
  Engine copy of the same granules. That was the only way to get real GEDI
  values here, and the scripts themselves do not use Earth Engine.
- **Embedding reader check.** During development, the downloaded embedding also
  matched Earth Engine's copy of the dataset over the whole box, to float rounding.

A single small area says little about performance elsewhere. Check the metrics in
each run report for your own AOI. Differences in R² below about 0.01 between runs
are noise: floating-point-level changes to the reference values moved R² by about
0.003.

## Limitations

- **The soil layers can only be as good as SoilGrids.** K and PAWC learn from
  250 m SoilGrids predictions, not field measurements. The 10 m result sharpens
  SoilGrids spatially but cannot correct it. The metrics measure agreement with
  SoilGrids, not with real soils. Where soil survey data exist (for example
  gSSURGO in the US), they are the better reference.
- **Metrics stay optimistic on small AOIs.** SoilGrids is smooth over distances
  longer than the 2 km blocks, and a small AOI has few blocks.
- **Random-forest predictions stay inside the training range.** They never
  extrapolate, and they shrink toward the mean at the extremes.
- **Predictions cover every pixel with an embedding,** including water and
  built-up land where a parameter may be undefined. Mask or override those
  classes as your InVEST setup needs.
- **Downloads grow with area and time window.** The embedding is roughly 0.6 MB
  per km² per year. GEDI granules are 150–250 MB each and cover whole orbit
  segments, so even a small AOI needs every granule that crosses it. The
  one-year box above needs nine granules, 1.7 GB. Processing runs in 512-pixel
  windows, so memory use stays small.
