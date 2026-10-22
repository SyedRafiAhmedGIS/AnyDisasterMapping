# Wildfire Smoke Plume Mapping — Data Preparation

**Task:** segmentation (clear / smoke / cloud)
**Sensor:** optical RGB (MODIS, VIIRS, GOES)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| Smoke-masked MODIS/VIIRS collections | Research release | e.g. HM-smoke style datasets |
| GOES-R ABI smoke products | Public (NOAA) | Derived masks |

## Expected folder layout

```
data/smoke_plume/SMOKE_MODIS/
├── IMG/{id}.tif      # 3-ch RGB
├── GT/{id}.tif       # {0=clear, 1=smoke, 2=cloud}
├── train.txt / val.txt / test.txt
```

## Steps

1. Download imagery + masks; unify to the 3-class convention.
2. Dummy set: `python scripts/data_prep/smoke_plume/make_dummy.py --out data/smoke_plume/SMOKE_MODIS_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain smoke_plume --path data/smoke_plume/SMOKE_MODIS_SMOKE`

## Configs
- `configs/smoke_plume/smoke_modis/seg_unet.yaml`
- `configs/smoke_plume/smoke_modis/seg_segformer_b2.yaml`
