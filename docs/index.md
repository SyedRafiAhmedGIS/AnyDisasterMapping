# AnyDisasterMapping

A unified satellite **disaster-mapping framework** covering the full lifecycle of
crisis response with Earth observation: benchmark datasets, model zoo, training,
inference, and deployment — companion to the survey
*Earth Observation for Disaster Mapping: Benchmarks, Methods, Challenges and
Future Perspectives* (SSRN [6725082](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6725082)).

## What's inside

| Asset | Count | Where |
|---|---|---|
| Task contracts | 3 — `seg`, `cd`, `scd` | `src/tasks/` |
| Registered models | 34 | `src/core/registry.py` |
| Registered datasets | 24 | `src/core/registry.py` |
| Ready-to-run configs | 628 YAML | `configs/` |
| Colab notebooks | 17 (all auto-run, Drive-aware) | repo root |
| Disaster domains | 13 | `docs/domains/` |

One config drives everything — model, dataset, augmentation, training and test
loops are shared across all three tasks:

```bash
pip install -e . -c constraints.txt          # verified-good dependency set
python train.py --config configs/flood/urbanflood/seg_unet.yaml
python test.py  --exp_path results/<run> --checkpoint best.pth
```

## One-click Colab

Every notebook auto-runs top-to-bottom, mounts Google Drive, degrades
gracefully to synthetic data when a dataset or checkpoint is unavailable, and
never blocks on user input.

| Notebook | Purpose |
|---|---|
| `quickstart_colab.ipynb` | 5-minute end-to-end tour |
| `demo_colab.ipynb` / `demo_gif_colab.ipynb` | inference + animated output |
| `flood_ / wildfire_ / tsunami_ / hurricane_ / landslide_ / volcano_colab.ipynb` | per-domain pipelines |
| `change_detection_colab.ipynb` | bi-temporal `cd` task |
| `scd_colab.ipynb` | semantic change detection + transition matrix |
| `satellite_finetune_colab.ipynb` | fine-tune on your own imagery |
| `batch_inference_colab.ipynb` | large-tile / multi-scene sweeps |
| `ai_images_colab.ipynb` | generated-imagery stress tests |
| `dataset_download_colab.ipynb` | access hub for all 24 benchmarks |
| `checkpoint_zoo_colab.ipynb` | real weights → real inference (34-model zoo) |
| `evaluation_metrics_colab.ipynb` | OA / mIoU / F1 / confusion matrices |

## Deployment

Trained models leave the framework through [`tools/export_onnx.py`](TOOLS.md)
(ONNX opset 18 + TorchScript, onnxruntime-parity checked), the Dockerfile
(CPU + `--target demo` Gradio image), or the Gradio app in `gradio_demo/`.
See **Tools & Deployment** for the full guide.

## Provenance & QA

This distribution includes the independent engineering review's remediation
(`FIXES_APPLIED.md`): all blockers fixed, 58/58 registry imports tested in CI,
full CPU smoke suite green (`scripts/smoke_test_all_domains.py`).
