# Avalanche Debris Mapping — Data Preparation

**Task:** segmentation (background / avalanche debris)
**Sensor:** optical (SPOT-6/7, Sentinel-2) or SAR-derived composites, 4 channels (R, G, B, NIR)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| Swiss (SLF/SwissTopo) avalanche outlines | Open data | Outlines + orthophotos 2000–2022 |
| 2026 multi-region SAR avalanche benchmark | Check release | SAR debris mapping study |

## Expected folder layout

```
data/avalanche/AVALANCHE/
├── IMG/{id}.tif      # 4-ch (R, G, B, NIR)
├── GT/{id}.tif       # {0=background, 1=avalanche debris}
├── train.txt / val.txt / test.txt
```

## Steps

1. Download outlines + imagery; rasterise outlines to binary masks.
2. Tile to chips (e.g. 512x512); write `train/val/test.txt`.
3. Dummy set: `python scripts/data_prep/avalanche/make_dummy.py --out data/avalanche/AVALANCHE_SMOKE`
4. Sanity: `python scripts/data_prep/sanity_check.py --domain avalanche --path data/avalanche/AVALANCHE_SMOKE`

## Configs
- `configs/avalanche/avalanche_slf/seg_unet.yaml`
- `configs/avalanche/avalanche_slf/seg_segformer_b2.yaml`
