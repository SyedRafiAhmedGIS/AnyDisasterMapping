# 🏚️ Infrastructure damage (earthquake) — domain guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/quickstart_colab.ipynb)
[![Back to README](https://img.shields.io/badge/←_back-README-181717?logo=github)](https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/README.md)

## What this domain covers

Collapsed blocks lose the regular urban texture: rubble reads rougher and darker, shadows change. Pre/post pairs around seismic events are the standard protocol.

| | |
|---|---|
| **Task** | cd |
| **Best sensors** | Optical / SAR |
| **Reference datasets** | xBD building-damage annotations |
| **Ready notebook** | [`quickstart_colab.ipynb`](https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/quickstart_colab.ipynb) |
| **Data-prep guide** | [`scripts/data_prep/infra_damage/README.md`](https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/scripts/data_prep/infra_damage/README.md) |

## Run it in 3 steps

1. Open the notebook with the badge above (or from the README index).
2. **Runtime ▸ Run all** — setup, a labeled synthetic sample, inference, dashboard and export chain automatically.
3. Click 📤 to upload your own scene(s) — the inline preview shows exactly which files fit, and the pipeline re-runs on them automatically.

## Upload checklist

- Accepted: `.png .jpg .jpeg .tif .tiff .h5` — GeoTIFF georeference is kept when present.
- Sweet spot: 512–2048 px per side, ≤ 20 MB per file, pre/post pairs from the *same* area.
- Set the **GSD (m/px)** box so km² statistics match your sensor (Sentinel-2 ≈ 10 m/px).
- No imagery at hand? The 🛟 button generates an AI-style synthetic infra_damage scene — clearly labeled, never presented as real data.

## Honest-results policy

Every output is stamped with the mode that produced it — **CHECKPOINT** (your `.pth`), **QUICK-TRAIN** (genuinely briefly-trained) or **SYNTHETIC-DEMO** (clearly-labeled heuristic). Train your own checkpoint in [`satellite_finetune_colab.ipynb`](https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping/blob/main/satellite_finetune_colab.ipynb) and upgrade any domain to CHECKPOINT mode.
