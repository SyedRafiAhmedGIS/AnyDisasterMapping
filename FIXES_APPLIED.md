# Round 4 — licensing resolution + remaining release blockers (2026-10-22)

Addresses the outstanding "must do before release" items. Every upstream
license below was **verified against the official release on 2026-10-22**
(repository page / GitHub API / upstream LICENSE file), not inferred.

## 1. Licensing decision: RESEARCH-ONLY distribution

- README badge no longer advertises a plain Apache-2.0 (it now reads
  "Apache-2.0 core, mixed research-only bundle", orange); the `## License`
  section states the decision and the three-tier component inventory.
- `NOTICE` states the research-only decision up front.
- `pyproject.toml` `license` field no longer implies the whole distribution
  is Apache-2.0.

## 2. Upstream licenses verified (previously unidentified)

| Component | Upstream | Verified terms | Action taken |
|---|---|---|---|
| `src/models/BIT/` | justchenhao/BIT_CD | No LICENSE file; README: **non-commercial, research only** | `third_party_licenses/BIT.LICENSE` + headers + dir notice |
| `src/models/DSIFN/` | GeoZcx/DSIFN repo | No LICENSE file; README: **non-commercial, research only** | `third_party_licenses/DSIFN.LICENSE` + headers + dir notice |
| `src/models/RS3Mamba/` | walking-shadow/Official_Remote_Sensing_Mamba | **Apache-2.0** | `RS3Mamba.LICENSE` + `Apache-2.0.LICENSE` + headers + dir notice |
| `src/models/FarSeg/` | Z-Zheng/FarSeg | **Apache-2.0** | `FarSeg.LICENSE` + headers + dir notice |
| `src/models/UNetFormer.py` | WangLibo1995/GeoSeg | **GPL-3.0** (copy-left!) | `UNetFormer-GeoSeg.LICENSE` + full `GPL-3.0.LICENSE` + header |
| `src/models/SpectralGPT/` | danfenghong/IEEE_TPAMI_SpectralGPT | **GPL-3.0** for SpectralGPT-authored code (the earlier CC BY-NC inference was wrong for the wrapper — but the embedded MAE-derived code IS CC BY-NC, so the folder stays research-only) | `SpectralGPT.LICENSE` + corrected notices |
| `src/models/HyperSigma/` | whu-sigma/hypersigma | **Apache-2.0** (embedded BEiT parts MIT) | `HyperSigma.LICENSE` + dir notice |
| `src/models/ChangeOS.py` | Z-Zheng/ChangeOS | **No upstream license** (GitHub license field: null) — all rights reserved by default | `ChangeOS.LICENSE` + header |

27 model files now carry an explicit `LICENSE-HEADER` comment; the "no
license header found" section in `THIRD_PARTY_NOTICES.md` is replaced by the
resolved inventory plus a new "Repository-local re-implementations" section
(UNet, UNet++, DeepLabV3+, FCN8s, SwinUperNet, convnext, DinoV2DPT variants,
DamageFormer).

## 3. CI: torch matrix

Both jobs now run on **torch 2.5.1 AND 2.14.1** (CPU wheels) so the range
`constraints.txt` allows (`torch>=2.1,<3`) is actually exercised — the
2.14.1 path is the one this distribution was fully verified on.

## 4. Reproducibility: opt-in `training.deterministic`

New trainer flag (default **off**, zero behaviour change): sets
`cudnn.deterministic=True`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True, warn_only=True)` and seeds
`CUBLAS_WORKSPACE_CONFIG`. Covered by new `tests/test_determinism.py`
(3 tests: default no-op, explicit off no-op, on-state verified).

## 5. Not resolvable from the packaging side (authors' action)

- Publishing `SyedRafiAhmedGIS/AnyDisasterMapping-weights` (needs the HF
  account; `download_weights.py` + SHA256SUMS verification are ready and
  tested).
- ~~The 17 notebooks' duplicated setup block~~ **resolved in Round 5**
  (shared `tools/colab_kit.py` + thin cells).
- ~~Notebooks clone `main` unpinned~~ **resolved in Round 5** (standalone
  fetch pinned to tag `v5.0.0`; tag must be pushed once per
  `GITHUB_UPLOAD_INSTRUCTIONS.md` §5).

---

# Round 5 — notebook dedup + pinned kit, deprecations, CI/health (2026-10-22)

## 1. The 17-notebook duplication, closed

`tools/colab_kit.py` (torch-free, 32 KB) is now the single source of truth
for the code every notebook used to inline: the synthetic-scene engine,
heuristics, visualisation, readers, Drive mount, RUN dashboard and both
EXPORT variants, plus the 5 upload helpers. Cells became thin delegates.

- **Extraction, not transcription:** the engine was lifted byte-exact from
  the notebooks via AST (`scripts/build_colab_kit.py`), so no copy drift is
  possible; interactive helpers were ported to explicit `globals()`-namespace
  functions with identical prints/file outputs.
- **Equivalence proof:** `scripts/verify_colab_equivalence.py` executes
  original (v4) vs refactored cells side by side — 59/59 checks identical
  (engine arrays on all 13 domains, RUN saved files pixel-for-pixel, EXPORT
  manifests/zip contents, MOUNT branches, upload STATE effects).
- **Two latent bugs fixed in the ports:** missing `clear_output` import
  (NameError on first upload click) and a hardcoded `/content` export path
  (FileNotFoundError outside Colab). `google.colab` is optional now, so the
  demos run end-to-end locally.
- **Clone pinning (was: needs repo history):** standalone Colab runs fetch
  the kit pinned to tag `v5.0.0`, with a *visible warning* fallback to
  `main` until the tag is pushed (`GITHUB_UPLOAD_INSTRUCTIONS.md` §5).
  Consistency enforced by `scripts/validate_notebooks.py` and a dedicated
  test.
- **Permanent guards:** `tests/test_colab_kit.py` (31 torch-free tests) and
  the notebook validator are wired into CI; the validator fails if any
  shared body is ever re-inlined into a notebook.

## 2. Deprecated APIs migrated (14 files)

- `timm.models.layers` → `timm.layers` with old-timm fallback (11 files:
  UNetFormer, SegFormer, vmamba, ChangeMamba ×2, HyperSigma ×2, SkySense,
  RS3Mamba ×1, SpectralGPT util) — the FutureWarning spam on every model
  import is gone.
- `torch.cuda.amp.custom_fwd/custom_bwd` → `torch.amp` with
  `device_type="cuda"` in `vmamba.py` + the 3 vendored selective-scan test
  files (Mask2Former already used the modern pattern in-repo);
  `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")` in
  `vmamba_checks.py`.
- Verified: zero repo-originated FutureWarnings when importing 9 model
  modules (warnings-as-errors harness). Remaining pytest warnings trace to
  `fvcore`'s `torch.jit.script` inside site-packages — upstream, not ours.

## 3. CI / release / community health

- `lint` job: ruff over first-party code (vendored dirs excluded in
  `pyproject [tool.ruff]`; notebooks have their own validator, not ruff).
- `docs.yml`: mkdocs gh-pages deploy (strict build first) on docs changes.
- `release.yml`: on `v*` tags → sdist + wheel + SHA256SUMS → GitHub Release.
- Colab-kit tests added to the CI import job.
- New: `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`.
- `pyproject`: explicit `torch>=2.1,<3`, `torchvision>=0.16,<1` bounds
  mirrored from `constraints.txt`.

---

# Round 3 — full execution verification (torch-dependent suite actually run)

The Round 2 note above asked for `pytest tests` and
`python scripts/smoke_test_all_domains.py` to be run before releasing — this
round executes exactly that, on **Python 3.12.15 / torch 2.14.1+cpu /
transformers 4.57.6**:

| Check | Result |
|---|---|
| `python -W error::SyntaxWarning -m compileall` (src, scripts, tests, tools, gradio_demo, train.py, test.py) | **PASS** (exit 0) |
| `pytest tests` | **74/74 PASS** (incl. registry import-all 34 models / 24 datasets / 3 tasks, safe-load, trainer regressions, repo integrity, download-weights) |
| `python scripts/smoke_test_all_domains.py` | **12/12 PASS** (11 domains trained 1 epoch + test.py on CPU; `glof_scd` honest CUDA skip — selective-scan kernel needs a CUDA runtime) |
| Notebooks (17) | **ALL** valid JSON, every code cell AST-parses, badges + Drive cell present |
| `tools/export_onnx.py --check` | seg UNet parity **5.96e-08**, cd BIT parity **5.66e-07** (onnxruntime) |
| `tools/download_weights.py --list` | graceful no-release path, exit 0 |
| `gradio_demo/app.py` | module executes, demo object builds |
| `mkdocs build --strict` | clean (0 warnings) |
| `test.py` freshness | `.meta` sidecar + `test_metrics_<split>.json` + `[SKIP] fresh` behaviour verified end-to-end |
| CI placeholder sweep | **fixed this round** — this changelog itself contained the §1.3 placeholder token (uppercased GitHub-username literal) in its heading, which made the CI `! grep` step fail on every push; rephrased so the sweep passes while keeping the changelog meaningful. |

Known, intentionally unchanged: vendored third-party modules still emit
`FutureWarning`s under torch ≥2.6 (`torch.cuda.amp.custom_fwd/bwd` in
ChangeMamba's `vmamba.py`, `timm.models.layers` imports in SegFormer) —
deprecations only, no functional impact; touching vendored code is deferred
to upstream syncs.

---

# Round 2 — review follow-up (applied on top of the fixes below)

Static review findings addressed in this revision. **Not executed in the authoring
sandbox** (no PyTorch available there): the changes are compile-checked, the
torch-free tests pass (`tests/test_repo_integrity.py`, `tests/test_download_weights.py`),
and the torch-dependent tests (`tests/test_safe_load.py`,
`tests/test_trainer_regressions.py`) are added for CI to exercise. Please run
`pytest tests` and `python scripts/smoke_test_all_domains.py` before releasing.

| # | Area | Change |
|---|---|---|
| 1 | Licensing | Added `NOTICE`, `THIRD_PARTY_NOTICES.md`, `third_party_licenses/` (upstream texts: DINOv3, SegFormer/NVIDIA, MAE/SatMAE CC BY-NC, SAM/SAM2, HRNet, Swin, CLIP, Mamba, VMamba, ChangeMamba, Lovasz, SiamCRNN, BEiT), per-directory `LICENSE_NOTICE.txt`, README License section. Non-commercial / DINOv3-licensed components are flagged. |
| 2 | Checkpoint safety | New `src/utils/safe_load.py` (`weights_only=True`, explicit `ANYDISASTER_TRUST_CHECKPOINTS=1` opt-in). All 15+ `torch.load` sites routed through it; SatMAE/SpectralGPT unsafe fallbacks now opt-in; trainer checkpoints store plain Python types only. |
| 3 | `dvc.yaml` | Rewritten against files that exist (GLOF generator + config), full dataset overrides, docs updated; `tests/test_repo_integrity.py` guards it. |
| 4 | Docker demo | `GRADIO_SERVER_NAME=0.0.0.0` in the `demo` stage; `app.py` honours `GRADIO_SERVER_NAME/PORT`. |
| 5 | `download_weights.py` | No more same-name overwrites (`--all` keeps layout), flat file preferred / ambiguity error, real 404 detection, SHA-256 printing + `SHA256SUMS` / `--expect-sha256` verification, `--revision` pinning, non-zero exit on failure. |
| 6 | `test.py` | Freshness now also checks checkpoint name + split + config hash (`*.meta` sidecar); explicit split writes `test_metrics_<split>.json`; `dataset.test` is created if absent so overrides are never dropped. |
| 7 | Trainer | Dataset class resolved via `lazy_import(...).module_path/.class_name` (no closure-cell introspection, no silent `except`). |
| 8 | Trainer | `max_steps` shorter than an epoch now still evaluates, saves `best.pth` and advances the epoch. |
| 9 | Trainer | `_seedN` output-dir suffix is idempotent (no `_seedN_seedN` on re-load). |
| 10 | Trainer | Early stop mid-epoch keeps `latest.pth` epoch consistent with the saved weights. |
| - | Misc | Raw docstrings remove the remaining `SyntaxWarning`s; README now lists all 17 notebooks; CREDITS count fixed; the SSRN number is the *paper*, not the review report; CI compiles `tools/`, `gradio_demo/`, `train.py`, `test.py` with warnings as errors and runs the new tests. |

Known, intentionally not changed: the 17 Colab notebooks still duplicate a large
shared setup/engine block (extracting it into an importable module is a larger
refactor); CI still pins torch 2.5.1 only.

---

# FIXES_APPLIED — Remediation Changelog (2026-10-21)

This release applies every fix from the independent review of
`AnyDisasterMapping-extended` (companion report: *AnyDisasterMapping_Review_Report.pdf*; the SSRN
entry 6725082 cited in the README is the *paper*, not this report). All changes were verified on Python 3.12 / torch 2.14.1 CPU:
**compile 100% · import-all 58/58 · smoke suite 12/12 · notebooks 17/17 JSON+AST**.

---

## 1. Blockers fixed (P0)

### 1.1 `src/models/ChangeMamba/vmamba.py` — IndentationError (CRITICAL)
The `except:` fallback for the triton import had no indented body, raising
`IndentationError` at import time and killing **both ChangeMamba factories on
every host** (11 YAML configs affected).

Fix went further than re-indenting — the original fallback imported
`csm_triton` anyway, and `csm_triton` hard-imports `triton`, so the module was
dead on **CPU-only hosts even after re-indenting**. Final design:
- `from .csm_triton import ...` is now guarded and sets `TRITON_CSM_AVAILABLE`.
- Without triton, `CrossScanTriton / CrossMergeTriton / CrossScanTriton1b1`
  are aliased to the pure-pytorch `CrossScan / CrossMerge` classes defined in
  the same file (the configured `"v2"` forward type never uses triton kernels).
- Result: `changemamba` (32.0M params), `changemamba_scd` (34.1M params) and
  `rs3mamba` (64.1M params) now **build on CPU**. The SSM forward still
  honestly requires the CUDA selective-scan kernel — unchanged, documented.

Side-effect discovery: `RS3Mamba` and `SAM2FPN/SAM2Mamba` also failed to import
through the same vmamba dependency — fixed by the same change.

### 1.2 `transformers==4.57.0` — yanked on PyPI
Fresh `pip install -e .` failed without workarounds. Now
`transformers>=4.57.1,<5` (verified resolving to **4.57.6**). The `<5` cap is
kept because vendored Mask2Former / CLIPDPT / SAMDPT use the 4.x API.

### 1.3 GitHub-username placeholder tokens (58 occurrences / 28 files)
All `YOUR_` + `GITHUB_` + `USERNAME`-style placeholder tokens replaced — first with the upstream repository owner the original framework,
then rebranded to this distribution's owner **`SyedRafiAhmedGIS`**:
- 26 links in 13 Colab notebooks ("Open In Colab" + repo badges — previously
  guaranteed 404s)
- 13 in README.md badges/links
- 6 in GITHUB_UPLOAD_INSTRUCTIONS.md
- 13 in docs/domains/*.md

### 1.4 `src/common/lovasz_loss.py` — `is` comparison with string literal
`classes is 'present'` worked only by interning accident (SyntaxWarning on
3.12+). Fixed to `==`.

---

## 2. Colab collection — every file now asks for Drive & runs clean

### 2.1 Dedicated "Mount Google Drive" cell in ALL 17 notebooks
Previously 12/13 mounted Drive only optionally at export time and
`demo_colab.ipynb` had no mount at all. Every notebook now has a prominent
`#@title ☁️ MOUNT GOOGLE DRIVE` form cell right after SETUP:
`MOUNT_DRIVE` toggle, authorization prompt, persistent
`/content/drive/MyDrive/AnyDisasterMapping` root, graceful degradation outside
Colab.

### 2.2 Four NEW notebooks (previously missing functionality)

| notebook | what it adds |
|---|---|
| `dataset_download_colab.ipynb` | Access hub for **all 24 registered benchmarks**: per-dataset access cards (where to request it, which adapter loads it), 3 fetch strategies (auto pull · your own Drive file id · zip upload), structure verifier with 3×3 tile preview, synthetic fallback so it never blocks |
| `checkpoint_zoo_colab.ipynb` | **Real weights → real inference** for the 34-model zoo: weights cards, HF release mirror → own `.pth` upload → **train a REAL tiny U-Net in-session** (~60s) fallback; checkpoint fingerprinting (sha256, param count); genuine inference dashboard. No more heuristic-only demo mode |
| `scd_colab.ipynb` | First notebook covering the **`scd` task**: uploads/synthesizes land-cover map pairs, trains a real tiny Siamese dual-decoder SCD model in-session, renders the signature from→to transition matrix (forest→built-up etc.) |
| `evaluation_metrics_colab.ipynb` | **Metric suite matching test.py conventions**: OA · mIoU · per-class IoU/precision/recall/F1-Dice/support · confusion matrix heatmap, for both `seg` and `cd` conventions; metrics.json export; controllable synthetic error-model |

All four follow the house style: Colab form cells, auto pip, upload buttons +
inline preview, never-block synthetic fallback, results zip + manifest +
Drive copy + browser download, patched badges.

### 2.3 Verification of all 17 notebooks
Every notebook passes: valid nbformat JSON, **every code cell AST-parses**,
badges point to `SyedRafiAhmedGIS/AnyDisasterMapping`, Drive cell present.

---

## 3. Test & CI infrastructure (P1)

- **`tests/test_registry_import_all.py`** (new): imports *every* model
  (34) / dataset (24) / task (3) registry target on every host. This is the
  test that would have caught the vmamba blocker on day one — it found the
  additional RS3Mamba/SAM2 breakage immediately.
- **`.github/workflows/ci.yml`** (new): two jobs — (1) py3.10 + py3.12 matrix:
  compile-all, import-all, placeholder sweep; (2) full CPU smoke suite.
- **`constraints.txt`** (new): verified-good pins + rationale
  (`pip install -e . -c constraints.txt`).
- **`pyproject.toml`**: `requires-python = ">=3.10,<3.13"` (3.12 verified).
- **smoke test**: `glof_scd` skip now gated on `torch.cuda.is_available()`
  (not merely `import triton`) and its message explains that module imports
  are covered by the CI import job.

---

## 4. Code-quality fixes

- **`src/core/trainer.py`**: `pin_memory=(num_workers > 0)` in both loader
  paths — removes the PyTorch UserWarning and the pointless pinned-copy with
  `num_workers=0`.
- **`src/common/lovasz_loss.py`**: `is` → `==` (see 1.4).

## 5. Verified not broken / intentionally unchanged

- `numpy==1.26.4` pin kept (vendored Mamba code relies on NumPy-1 promotion
  rules); the nlopt-driven numpy-2 resolver conflict is documented in
  `constraints.txt` instead of risking an untested upgrade.
- Smoke suite: **12/12 PASS** (11 executed + `glof_scd` honest CUDA skip).
- Runtime smoke timings unchanged (6.1–6.5 s/domain, single-domain train).

## 6. Deployment & ops addons (now shipped)

1. ~~Publish model weights~~ → **consumer tooling shipped**:
   `tools/download_weights.py` lists/pulls from
   `SyedRafiAhmedGIS/AnyDisasterMapping-weights` (flat or `seg/ cd/ scd/`
   layout), degrades gracefully until the release goes live, and prints the
   exact publish command. The actual upload remains an authors' action.
2. ~~Dataset mirrors~~ → **versioning pipeline shipped**: `dvc.yaml`
   (data → train → test, DVC-tracked) + `docs/DATA_VERSIONING.md`
   (DVC remotes + Zenodo DOI workflow). Mirror uploads remain an authors'
   action.
3. **ONNX/TorchScript export helper**: `tools/export_onnx.py` — registry-path
   model build, task-aware inputs (seg single-image; cd/scd dual-input per
   `DUAL_INPUT_MODELS`), opset 18 default (17 down-conversion is lossy in
   onnxscript), static-by-default with honest dynamic-axes fallback,
   external-weight folding to single-file, `--check` onnxruntime parity.
   Extras group: `pip install -e ".[export]"`. Verified: UNet seg
   (parity 8.9e-08) and BIT cd (5.7e-07) round-trips.
4. **Gradio demo**: `gradio_demo/app.py` (+ Docker `--target demo`).
5. **MkDocs Material site**: `mkdocs.yml` + `docs/index.md` +
   `docs/TOOLS.md`; domain pages' `../../` links rewritten to absolute
   GitHub URLs (they could never resolve on a deployed site). Verified with
   `mkdocs build --strict` — zero warnings.
6. **Dockerfile**: `python:3.12-slim` + `constraints.txt` cached dependency
   layer, CPU/GPU via pip CUDA wheels, `demo` target for the Gradio app;
   `.dockerignore` keeps data/weights/artifacts out of images.

## 7. Post-addon verification (this release)

compileall exit 0 · import-all 58/58 · smoke suite 12/12 · notebooks 17/17
JSON+AST+badges+Drive-cell · `tools/export_onnx.py` seg+cd round-trips with
onnxruntime parity · `tools/download_weights.py` graceful no-release path ·
`mkdocs build --strict` clean · repackaged zip.
