# Volcanic Eruption Mapping — Data Preparation

**Task:** segmentation (background / lava flow / vent / fumarole)
**Sensor:** VHR optical (RGB; thermal bands can be appended as extra channels)
**Primary datasets:**

| Dataset | Access | Notes |
|---|---|---|
| PyRawS (ESA Phi-Lab) | Public on Zenodo | VHR tri-stereo, lava/vent/fumarole/incrustation |
| Sundhnúkur (Iceland) lava-flow mappings | Research release | 2023–24 eruption time series |
| xBD eruption events | Public (xView2 site) | Building damage focus |

## Expected folder layout

```
data/volcano/PYRAWS/
├── IMG/{id}.tif      # 3-ch RGB (0..1)
├── GT/{id}.tif       # {0=bg, 1=lava flow, 2=vent, 3=fumarole}
├── train.txt / val.txt / test.txt
```

## Steps

1. Download PyRawS from Zenodo; keep RGB chips + 4-class masks (merge
   incrustation into fumarole or re-map as needed).
2. Dummy set: `python scripts/data_prep/volcano/make_dummy.py --out data/volcano/PYRAWS_SMOKE`
3. Sanity: `python scripts/data_prep/sanity_check.py --domain volcano --path data/volcano/PYRAWS_SMOKE`

## Configs
- `configs/volcano/volcano_pyraws/seg_unet.yaml`
- `configs/volcano/volcano_pyraws/seg_segformer_b2.yaml`
