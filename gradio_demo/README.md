---
title: AnyDisasterMapping Demo
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# 🌍 AnyDisasterMapping — Gradio web demo

No-code web UI over the same engine as the Colab notebooks:

| tab | what it does |
|---|---|
| 🛟 Synthetic demo | generates a pre/post disaster pair (13 domains) + change map — **zero uploads needed** |
| 🆚 Change detection | upload your own PRE/POST pair → heatmap + overlay + km² stats |
| 🖼️ Segmentation | single-scene class mapping (water, burn scar, lava, …) |
| 🎨 AI illustrations | AI-style poster renders, watermarked **NOT A REAL PHOTO** |

**Honest-demo policy:** every output is mode-labeled (`SYNTHETIC-DEMO` heuristic here;
`CHECKPOINT` / `QUICK-TRAIN` in the notebooks when real weights are used). Nothing is faked.

## Run locally

```bash
pip install -r requirements.txt
python app.py          # → http://127.0.0.1:7860
```

## Deploy on Hugging Face Spaces (free)

1. Create a Space → SDK **Gradio** → upload this folder (`app.py`, `requirements.txt`).
2. Done — the Space builds itself. No secrets, no weights, no real photos.
