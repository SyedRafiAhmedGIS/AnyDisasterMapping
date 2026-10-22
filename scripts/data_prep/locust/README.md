# Locust Outbreak Mapping — Data Preparation

**Task:** segmentation (habitat / affected cropland) + change detection (new infestation)
**Sensor:** Sentinel-2, 4 channels (B2, B3, B4, B8)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| FAO DLIS locust surveys | Public (points) | Presence/absence records, NOT masks |
| Sentinel-2 L2A | Public (Copernicus) | Imagery source |

**Label derivation (important):** FAO data are point observations. Build a
raster label by (a) buffering presence points, (b) intersecting with cropland
/ grassland land cover (ESA WorldCover), (c) manual cleaning. State this
derivation clearly in any derived dataset you release.

## Expected folder layout

```
data/locust/FAO_S2/
├── PRE/{id}.tif      # 4-ch pre-date
├── POST/{id}.tif     # 4-ch post-date
├── GT/{id}.tif       # seg: {0=bg, 1=habitat/affected}; cd: {0,1}
├── train.txt / val.txt / test.txt
```

## Steps

1. Derive labels as above; tile to chips.
2. Dummy set: `python scripts/data_prep/locust/make_dummy.py --out data/locust/FAO_S2_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain locust --path data/locust/FAO_S2_SMOKE`

## Configs
- `configs/locust/fao_s2/seg_unet.yaml`
- `configs/locust/fao_s2/cd_bit.yaml`
