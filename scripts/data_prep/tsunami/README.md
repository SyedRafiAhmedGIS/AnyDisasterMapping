# Tsunami Inundation Mapping — Data Preparation

**Task:** change detection (no-change / inundated) from pre/post pairs
**Sensor:** optical RGB or SAR GCPS composites, 3 channels per date
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| xBD tsunami events (xView2) | Public | Pre/post chips, building-damage masks (relabel to inundation) |
| MMFlood | Public | SAR flood mapping (transferable method reference) |

## Expected folder layout

```
data/tsunami/XBD_TSUNAMI/
├── PRE/{id}.tif      # 3-ch
├── POST/{id}.tif     # 3-ch
├── GT/{id}.tif       # {0=no-change, 1=inundated}
├── train.txt / val.txt / test.txt
```

## Steps

1. Download xBD tsunami holdout tiles; rasterise damaged/inundated areas to a
   binary inundation GT (or use community inundation mappings).
2. Dummy set: `python scripts/data_prep/tsunami/make_dummy.py --out data/tsunami/XBD_TSUNAMI_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain tsunami --path data/tsunami/XBD_TSUNAMI_SMOKE`

## Configs
- `configs/tsunami/xbd_tsunami/cd_bit.yaml`
- `configs/tsunami/xbd_tsunami/cd_changeos.yaml`
