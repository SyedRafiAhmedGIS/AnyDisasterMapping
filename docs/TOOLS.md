# Tools & Deployment

Everything in `tools/` plus the container and demo app — the bridge from a
trained experiment to a deployed product.

## 1. ONNX / TorchScript export — `tools/export_onnx.py`

Builds the model through the **same registry path the trainer uses**, loads
your checkpoint, and traces the forward pass exactly as the task handlers
invoke it (seg: single image; cd/scd: dual inputs — `DUAL_INPUT_MODELS`
kwargs call or channel-concat, per `src/tasks/*.py`).

```bash
# from a trained experiment dir (config.yaml + best.pth)
python tools/export_onnx.py --exp_path results/<run> --torchscript --check

# architecture check with random weights (no checkpoint needed)
python tools/export_onnx.py --config configs/oil_spill/oilspill_sar/cd_bit.yaml
```

| flag | meaning |
|---|---|
| `--exp_path` | export from a trained run (loads `config.yaml` + checkpoint) |
| `--config` | export from a YAML only (random weights) |
| `--size` | square input size, default 256 |
| `--opset` | ONNX opset, **default 18** (17 down-conversion is lossy in onnxscript) |
| `--dynamic` | try dynamic batch/H/W axes; falls back to static when the architecture can't support it (honest, never hard-fails) |
| `--torchscript` | also trace `model.torchscript.pt` |
| `--check` | verify with onnxruntime: shape match + max abs diff vs PyTorch |

Outputs land in `exports/<name>/` with `export_manifest.json` (shapes,
byte sizes, SHA-256). Extras: `pip install -e ".[export]"`.

## 2. Pretrained weights — `tools/download_weights.py`

```bash
python tools/download_weights.py --list                  # what is published
python tools/download_weights.py --model unet            # one model
python tools/download_weights.py --model bit --task cd   # task-prefixed layout
```

Pulls from `SyedRafiAhmedGIS/AnyDisasterMapping-weights` on the Hugging Face
Hub (flat `<model>.pth` or `seg/ cd/ scd/`-prefixed — same layout
`checkpoint_zoo_colab.ipynb` consumes). Until the release is published the
tool explains the fallbacks (train a config, or the zoo notebook's
in-session tiny training) instead of failing.

Behaviour worth knowing:

- `--model unet` prefers the flat root file; if only task-prefixed variants
  exist it asks you to add `--task` rather than guessing.
- `--all` keeps the repo layout under `--dest` (`seg/unet.pth`,
  `cd/unet.pth`, …) so same-named files never overwrite each other;
  `--all --task cd` downloads only that folder.
- Every download prints its SHA-256. A `SHA256SUMS` file in the release is
  verified automatically; `--expect-sha256 <hex>` pins one file; `--revision
  <commit>` pins the whole repo to an exact commit.
- Exit code is non-zero on any failed/unverifiable download or when nothing
  matches; "release not published yet" is the only graceful 0.

### Checkpoint safety

All checkpoint reads go through `src/utils/safe_load.py`
(`torch.load(weights_only=True)`). A file that needs full unpickling (e.g.
older MAE/SatMAE releases embedding `argparse.Namespace`) is **refused** by
default. If you trust the source, opt in explicitly:

```bash
ANYDISASTER_TRUST_CHECKPOINTS=1 python train.py --config <yaml>
```

## 3. Gradio demo — `gradio_demo/app.py`

Browser inference UI: upload imagery, pick a registered model, overlay the
prediction. Runs standalone (`pip install gradio>=4.0 pillow numpy imageio
scikit-image`) or via Docker:

```bash
docker build -t anydisaster-demo --target demo .
docker run --rm -p 7860:7860 anydisaster-demo    # → http://localhost:7860
# (the demo image sets GRADIO_SERVER_NAME=0.0.0.0; outside Docker it binds to 127.0.0.1)
```

## 4. Docker

Single `Dockerfile`, two targets, dependency layer cached against
`constraints.txt`:

```bash
docker build -t anydisaster .                    # CPU/GPU training image
docker run --rm --gpus all anydisaster \
    python scripts/smoke_test_all_domains.py     # verify GPU stack
docker run --rm --gpus all \
    -v "$PWD/data":/workspace/data -v "$PWD/results":/workspace/results \
    anydisaster python train.py --config configs/flood/urbanflood/seg_unet.yaml
```

## 5. Dataset versioning — DVC + Zenodo

`dvc.yaml` pins the make-data → train → test pipeline;
`docs/DATA_VERSIONING.md` walks through pushing a DVC remote and minting a
Zenodo DOI for the dataset snapshot.
