# Hurricane / Cyclone Storm-Surge Mapping — Data Preparation

**Task:** change detection (no-change / surge inundation)
**Sensor:** Sentinel-1 SAR, 2 channels (VV, VH) per date
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| Sen1Floods11 | Public on GEE | S1 flood labels — hurricane events included |
| xBD wind events | Public | Optical wind damage (alternative) |

## Expected folder layout

```
data/hurricane/S1_SURGE/
├── PRE/{id}.tif      # 2-ch (VV, VH)
├── POST/{id}.tif     # 2-ch
├── GT/{id}.tif       # {0=no-change, 1=surge inundation}
├── train.txt / val.txt / test.txt
```

## Steps

1. Export Sen1Floods11 hurricane chips via the GEE snippet in their repo.
2. Dummy set: `python scripts/data_prep/hurricane/make_dummy.py --out data/hurricane/S1_SURGE_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain hurricane --path data/hurricane/S1_SURGE_SMOKE`

## Configs
- `configs/hurricane/s1_surge/cd_bit.yaml`
- `configs/hurricane/s1_surge/cd_changeos.yaml`
