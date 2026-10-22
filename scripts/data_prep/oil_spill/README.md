# Oil Spill Mapping (SAR) — Data Preparation

**Task:** segmentation (sea / oil slick / look-alike) + change detection (new slick)
**Sensor:** Sentinel-1 SAR, 2 channels (VV, VH) per date
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| Krestenitis et al. Sentinel-1 benchmark (OSD) | Request via paper link | 1831 tiled S1 images, 3-class masks |
| SAR Oil Spill Dataset Part I (SIPI) | Public download | 65 SAR images w/ masks |
| CSIRO S1 oil spill | Request | Large S1 collection |
| OSCD (Oil Spill Change Detection) | Public | Multi-temporal slick change |

## Expected folder layout

```
data/oil_spill/OILSPILL/
├── PRE/{id}.tif      # 2-ch float (VV, VH), any scale (normalized in config)
├── POST/{id}.tif     # 2-ch float
├── GT/{id}.tif       # seg: {0=sea, 1=slick, 2=look-alike}; cd: {0,1}
├── train.txt         # one id per line (same base name across PRE/POST/GT)
├── val.txt
└── test.txt
```

## Steps

1. Download from a source above and convert to the layout with a short script
   (reproject, stack VV+VH as 2 bands, rasterise slick polygons to GT).
2. Generate split files listing base ids (one per line).
3. Generate a synthetic smoke-test set:
   `python scripts/data_prep/oil_spill/make_dummy.py --out data/oil_spill/OILSPILL_SMOKE`
4. Sanity check:
   `python scripts/data_prep/sanity_check.py --domain oil_spill --path data/oil_spill/OILSPILL_SMOKE`

## Configs
- `configs/oil_spill/oilspill_sar/seg_unet.yaml` (lightweight)
- `configs/oil_spill/oilspill_sar/seg_segformer_b2.yaml` (modern)
- `configs/oil_spill/oilspill_sar/cd_bit.yaml` (change detection)
