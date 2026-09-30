# Plant available water content (InVEST Annual Water Yield)

[predict_plant_available_water_content.py](predict_plant_available_water_content.py)
predicts the Annual Water Yield PAWC raster at 10 m as a volumetric fraction
from 0 to 1. It needs no account: the embedding comes from Source Cooperative
and SoilGrids from ISRIC's file server. The formula lives in
[available_water.py](available_water.py).

## Reference data

InVEST defines PAWC as field capacity minus permanent wilting point. Both are
estimated from SoilGrids 2.0 texture with the Saxton & Rawls (2006) pedotransfer
equations (eqs. 1–2):

```
θ1500t = -0.024S + 0.487C + 0.006OM + 0.005(S·OM) - 0.013(C·OM) + 0.068(S·C) + 0.031
θ1500  = θ1500t + (0.14 θ1500t - 0.02)
θ33t   = -0.251S + 0.195C + 0.011OM + 0.006(S·OM) - 0.027(C·OM) + 0.452(S·C) + 0.299
θ33    = θ33t + (1.283 θ33t² - 0.374 θ33t - 0.015)
PAWC   = (θ33 - θ1500) · (1 - coarse fragment volume fraction)
```

S and C are sand and clay as weight fractions, and OM is organic matter in percent.

**Assumptions**

- Organic matter is 1.724 × SoilGrids organic carbon, the van Bemmelen factor.
- Organic matter is capped at 8%, the limit of the Saxton & Rawls calibration
  data. The SPAW tool applies the same cap.
- The coarse-fragment term converts fine-earth water content to bulk soil, using
  SoilGrids `cfvo`.
- No density or salinity adjustment is applied.

PAWC is computed for each SoilGrids depth interval, then averaged by thickness
over 0 to `--depth-cm`. The default is 100 cm, a typical root zone. The Annual
Water Yield model multiplies PAWC by the smaller of root-restricting-layer depth
and rooting depth, so match `--depth-cm` to the soil depth you assume.

## Example

```bash
../.mamba-env/bin/python predict_plant_available_water_content.py --aoi watersheds.shp --output-tif ../outputs/pawc.tif
```

## Caveat

The reference is itself a 250 m model product. The prediction refines SoilGrids
spatially but cannot be more accurate than it. The metrics measure agreement
with SoilGrids, not with field-measured water retention.
