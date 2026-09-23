# Aboveground biomass carbon (InVEST Carbon: `c_above`)

[predict_aboveground_carbon.py](predict_aboveground_carbon.py) predicts
aboveground biomass carbon density at 10 m in Mg C/ha. InVEST calls this unit
t/ha. The embedding comes from Source Cooperative and GEDI from NASA. The
reference-data rules live in [gedi_carbon.py](gedi_carbon.py).

## Reference data

- **Source.** GEDI L4A version 2.1 footprint aboveground biomass density
  (`agbd`, Mg/ha, 25 m footprints), from NASA's ORNL DAAC. Each footprint is one
  training sample.
- **Quality filter.** Only shots with `l4_quality_flag = 1`, `degrade_flag = 0`,
  and non-negative biomass are kept.
- **Carbon conversion.** Biomass is multiplied by `--carbon-fraction`. The default
  is 0.47, the IPCC 2006 default; the InVEST guide cites 0.43–0.51.
- **Time window.** GEDI data come from the embedding year by default, set with
  `--gedi-start` and `--gedi-end`. A wider window gives more samples but mixes
  in land-cover change.
- **Coverage.** GEDI observes only between about 51.6° S and 51.6° N, from April
  2019 to July 2025. Outside that area or window the script stops with an error.

## NASA Earthdata login

Searching NASA's catalog needs no account, but downloading granules does.
Register free at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov).
Then provide the login in one of three ways:

- set `EARTHDATA_TOKEN`;
- set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`;
- add a `urs.earthdata.nasa.gov` entry to `~/.netrc`.

The script never prompts for a password. Granules are cached under
`--cache-dir`, so later runs over the same area and dates need no login. Each
granule is 150–250 MB and covers a whole orbit segment.

## From raster to InVEST table

The InVEST Carbon model takes one carbon value per land-cover class, not a raster.
With `--c-above-table` and `--lulc` the script averages the prediction within
each land-cover class and writes a CSV to merge into your carbon pools table:

```
lucode,c_above,pixel_count
```

Pass the LULC raster of your InVEST run as `--lulc`, so the codes match your
table. It is resampled onto the prediction grid by nearest neighbour. The
`c_below`, `c_soil`, and `c_dead` columns still come from other sources.

## Example

```bash
../.mamba-env/bin/python predict_aboveground_carbon.py --bbox -122.30 37.37 -122.20 37.43 --output-tif ../outputs/aboveground_carbon.tif --c-above-table ../outputs/c_above_by_lulc.csv --lulc lulc.tif
```

## Caveats seen in the verification run

- **Low-biomass classes come out too high.** In the Jasper Ridge run, ESA
  WorldCover grassland averaged about 17 Mg C/ha. GEDI L4A has a noise floor at
  low biomass, and the random forest cannot predict below its lowest training
  values. Check low-biomass classes against local data or IPCC defaults.
- **Water and wetland get nonzero values.** Water averaged about 46 Mg C/ha,
  mostly from mixed shoreline pixels. Set such classes explicitly in the carbon
  pools table.
