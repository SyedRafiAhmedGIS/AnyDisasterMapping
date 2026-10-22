# Glacial Lake Outburst Flood (GLOF) — Data Preparation

**Task:** segmentation (lake extent) + **semantic change detection** (lake
expansion; the repo's first shipped `task: scd` config)
**Sensor:** optical (Sentinel-2 / Landsat), 4 channels (B2, B3, B4, B8)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| ICIMOD glacial-lake inventories (HKH) | Public | Lake polygons, multi-date |
| GAMDAM / HKH lake database | Public | Regional inventories |

## Expected folder layout

```
data/glof/ICIMOD/
├── T1/{id}.tif       # 4-ch pre-date
├── T2/{id}.tif       # 4-ch post-date
├── GT_CD/{id}.tif    # {0=no-change, 1=new/expanded water}
├── GT_T1/{id}.tif    # {0=land, 1=water, 2=snow/ice}
├── GT_T2/{id}.tif    # {0=land, 1=water, 2=snow/ice}
├── train.txt / val.txt / test.txt
```

## Steps

1. Rasterise inventory lake polygons for two dates; add snow/ice class from
   NDSI thresholding.
2. Seg mode stacks T1+T2 (8-ch input) and predicts the T2 extent.
3. Dummy set: `python scripts/data_prep/glof/make_dummy.py --out data/glof/ICIMOD_SMOKE`
4. Sanity: `python scripts/data_prep/sanity_check.py --domain glof --path data/glof/ICIMOD_SMOKE`

## Configs
- `configs/glof/icimod/seg_unet.yaml`
- `configs/glof/icimod/seg_segformer_b2.yaml`
- `configs/glof/icimod/scd_changemamba_scd.yaml` (first shipped scd config; needs the ChangeMamba kernel)
