# Data versioning — DVC pipeline + Zenodo DOI

The 24 registered benchmarks are gated by their providers' licenses, so the
framework ships **adapters + access instructions** rather than the bytes.
This page describes how to version whichever subset you actually use, with
bit-exact reproducibility.

## 1. DVC pipeline (`dvc.yaml`)

Three stages model the canonical flow — data preparation, training, testing —
per domain config:

```bash
pip install dvc                      # one-time, or add to dev extras
dvc repro                            # runs only what changed
dvc status                           # drift check
```

Stages (see `dvc.yaml` at the repo root; defaults run the GLOF domain on synthetic
dummy data via `scripts/data_prep/glof/make_dummy.py` — change the `vars:` block
(`config`, `data_dir`, `run_dir`, `epochs`) for your own domain):

1. `make_data` — runs the domain's `scripts/data_prep/<domain>/make_dummy.py`
   **or** your licensed adapter download; tracks the prepared split files
   (`train.txt / val.txt / test.txt`) and `dvc.out` the data dir.
2. `train` — `python train.py --config <yaml>`; outs: `results/<run>/`.
3. `test` — `python test.py --exp_path results/<run> --force`;
   metrics: `results/<run>/test_metrics.json`.

To add your own domain: override the `vars:` block (and the `make_data` cmd
if you use a licensed adapter), then `dvc repro`. CI stays DVC-free (it uses the
synthetic dummies), so adding stages never slows the pipeline down.

## 2. Storing the bytes — DVC remote

```bash
dvc remote add -d storage s3://your-bucket/anydisaster   # or ssh/, gdrive/, azure/
dvc push                                                 # upload data outs
dvc pull                                                 # restore anywhere
```

DVC writes `.dvc` pointer files (tiny, commit-ready) while the payloads live
in the remote. Team members reproduce your exact data snapshot with
`git pull && dvc pull` — no "which flood version was this?" again.

## 3. Public archival — Zenodo DOI

For citable, immutable releases (paper reviewer requirements etc.):

1. Push the prepared dataset subset to a Zenodo deposit (or an open HF
   dataset repo, then mirror to Zenodo).
2. Record the returned **DOI** in the domain doc (`docs/domains/<domain>.md`)
   and in the dataset adapter's access card (`dataset_download_colab.ipynb`).
3. Tag the framework commit that consumed the dataset:
   `git tag data/<dataset>-v1.0 <commit>` so code + data versions stay paired.

Zenodo accepts files up to 50 GB each and mints a versioned DOI per release;
combined with the sha256 manifests the framework already emits for
checkpoints (`export_manifest.json`), any result in any paper can be traced
back to exact input bytes.
