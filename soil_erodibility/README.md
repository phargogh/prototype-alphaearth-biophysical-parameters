# Soil erodibility K (InVEST SDR)

[predict_soil_erodibility.py](predict_soil_erodibility.py) predicts the SDR soil
erodibility raster at 10 m in t·ha·h/(ha·MJ·mm). It needs no account: the
embedding comes from Source Cooperative and SoilGrids from ISRIC's file server.
The formula lives in [erodibility.py](erodibility.py).

## Reference data

K comes from SoilGrids 2.0 sand, silt, clay, and organic carbon, using the EPIC
equation (Sharpley & Williams 1990) as written in the InVEST SDR user guide:

```
K = A · B · C · D · 0.1317
A = 0.2 + 0.3 exp[-0.0256 SAN (1 - SIL/100)]
B = (SIL / (CLA + SIL))^0.3
C = 1 - 0.25 OC / (OC + exp(3.72 - 2.95 OC))
D = 1 - 0.7 SN1 / (SN1 + exp(-5.51 + 22.9 SN1)),   SN1 = 1 - SAN/100
```

SAN, SIL, CLA, and OC are percent by weight. The factor 0.1317 converts US
customary units to SI, as the InVEST guide specifies.

Some SWAT documentation prints the sand coefficient in A as −0.256 instead of
−0.0256. This script follows the InVEST guide. With −0.0256, a loam comes out at
K ≈ 0.034, inside the guide's typical range of 0.009–0.092.

K is computed for each SoilGrids depth interval, then averaged by thickness over
0 to `--depth-cm`. The default is 30 cm, the usual topsoil depth.

## Example

```bash
../.mamba-env/bin/python predict_soil_erodibility.py --aoi watersheds.shp --output-tif ../outputs/soil_erodibility.tif
```

## Caveat

The reference is itself a 250 m model product. The prediction refines SoilGrids
spatially but cannot be more accurate than it. The metrics measure agreement
with SoilGrids. Measured K values, such as gSSURGO `kwfact` in the US, would make
a better reference where available.
