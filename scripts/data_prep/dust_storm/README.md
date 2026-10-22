# Dust / Sand Storm Mapping — Data Preparation

**Task:** segmentation (clear / dust plume / cloud)
**Sensor:** optical RGB (MODIS, VIIRS, or Sentinel-2)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| MODIS/VIIRS dust detection studies | Research release | Plume masks from RGB composites |
| Earthdata (GIBBS/MOD09) | Public | Raw imagery; masks must be derived |

## Expected folder layout

```
data/dust_storm/MODIS_DUST/
├── IMG/{id}.tif      # 3-ch RGB
├── GT/{id}.tif       # {0=clear, 1=dust, 2=cloud}
├── train.txt / val.txt / test.txt
```

## Steps

1. Collect event imagery; digitise or derive plume/cloud masks (threshold +
   manual cleaning is common practice).
2. Dummy set: `python scripts/data_prep/dust_storm/make_dummy.py --out data/dust_storm/MODIS_DUST_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain dust_storm --path data/dust_storm/MODIS_DUST_SMOKE`

## Configs
- `configs/dust_storm/modis_dust/seg_unet.yaml`
- `configs/dust_storm/modis_dust/seg_segformer_b2.yaml`
