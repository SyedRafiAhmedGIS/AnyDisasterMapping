# CHANGELOG — AnyDisasterMapping Extended

All changes in this extension, file by file. Original repository untouched
except where explicitly listed. Base: `AnyDisasterMapping-master` (paper: SSRN 6725082).

## Fixed — Round 5: notebook dedup + pinned kit, deprecations, CI/health (2026-10-22)

- **17-notebook deduplication — shared core `tools/colab_kit.py`.** The four
  large blocks that every notebook duplicated (SETUP engine ≈400 lines,
  Drive-mount, RUN dashboard, EXPORT zip) and the 5 identical upload helpers
  now live in one torch-free, unit-tested module. Notebooks import it from
  the repo, or fetch it standalone in Colab **pinned to tag `v5.0.0`**
  (warned fallback to `main` before first release). Two latent bugs fixed in
  the ports: missing `clear_output` import (NameError on first upload) and a
  hardcoded `/content` export path (crashed outside Colab); `google.colab`
  is now optional so demos run end-to-end locally.
- **Behavioral equivalence proven:** an exec-based harness
  (`scripts/verify_colab_equivalence.py`, dev-only) compared original v4
  cells with the refactored ones — 59/59 checks identical (13-domain
  synthetic engine arrays, RUN files/pixels/console, EXPORT manifests/zip
  contents, MOUNT branches, upload STATE effects). Permanent guards:
  `tests/test_colab_kit.py` (31 tests) + `scripts/validate_notebooks.py`
  (JSON/AST/dedup/KIT_REV/README) wired into CI.
- **Deprecated APIs migrated (14 files).** `timm.models.layers` →
  `timm.layers` (old-timm fallback kept) in 11 files;
  `torch.cuda.amp.custom_fwd/custom_bwd/autocast` → `torch.amp`
  (`device_type="cuda"`) in `vmamba.py`, `vmamba_checks.py` and the three
  selective-scan kernel test files. Repo-originated FutureWarnings on
  import: 0 (verified with a warnings-as-errors harness over 9 model
  modules; the two remaining pytest warnings originate in `fvcore`'s
  `torch.jit.script`, i.e. site-packages, not this repo).
- **CI expanded.** New `lint` job (ruff over first-party code; vendored
  model dirs excluded via `pyproject [tool.ruff]`) + notebook validator.
  New workflows: `docs.yml` (mkdocs gh-pages deploy on docs changes) and
  `release.yml` (sdist+wheel+SHA256SUMS attached to GitHub Releases on
  `v*` tags). Colab-kit unit tests added to the import job.
- **Community health files.** `SECURITY.md` (research-code security policy,
  safe-loading scope), `CONTRIBUTING.md` (dev setup, house rules),
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `CITATION.cff`
  (GitHub-rendered citation; preferred-citation = SSRN paper).
- **Dependency hygiene.** `pyproject` now carries explicit
  `torch>=2.1,<3` / `torchvision>=0.16,<1` bounds mirroring
  `constraints.txt` (which remains the documented source of truth).
- **Docs/upload instructions refreshed.** README Colab section documents
  the shared kit + pin; `GITHUB_UPLOAD_INSTRUCTIONS.md` §4 no longer tells
  users to create a workflow the repo now ships, and adds the required
  `git tag v5.0.0` step that activates the kit pin + release workflow.

## Fixed — Round 4: licensing resolution + release blockers (2026-10-22)

- **Licensing decision: RESEARCH-ONLY distribution.** README badge and
  License section rewritten (plain Apache-2.0 badge was misleading), NOTICE
  and pyproject `license` field updated to state the decision.
- **Upstream licenses verified against official releases** (not inferred):
  BIT → non-commercial/research-only (README statement); DSIFN →
  non-commercial/research-only; RS3Mamba → Apache-2.0; FarSeg → Apache-2.0;
  UNetFormer (GeoSeg) → **GPL-3.0**; SpectralGPT → **GPL-3.0** with embedded
  MAE code CC BY-NC (earlier inference corrected, folder stays
  research-only); HyperSigma → Apache-2.0; ChangeOS → **no upstream
  license**. Added 9 files to `third_party_licenses/` (incl. full GPL-3.0
  text), `LICENSE-HEADER` comments to 28 model files, per-directory
  `LICENSE_NOTICE.txt` for BIT/DSIFN/RS3Mamba/FarSeg/ChangeMamba/HyperSigma
  and corrected SpectralGPT's; `THIRD_PARTY_NOTICES.md` inventory rewritten.
- **CI torch matrix:** both jobs now test torch 2.5.1 AND 2.14.1 (CPU) so
  the `constraints.txt` range is exercised.
- **Reproducibility:** new opt-in `training.deterministic` flag
  (cudnn.deterministic / benchmark off / deterministic algorithms
  warn-only + CUBLAS_WORKSPACE_CONFIG); default off, covered by
  `tests/test_determinism.py`.

## Fixed — Round 3 execution verification (2026-10-22)

- `FIXES_APPLIED.md` — the §1.3 heading contained the literal GitHub-username
  placeholder token, tripping the CI "Placeholder sweep" step (`! grep` fails
  when the token appears in any tracked `*.md`); rephrased so CI passes.
  Full torch-dependent verification (pytest 74/74, smoke 12/12, notebooks
  17/17, ONNX parity, mkdocs strict) executed — see `FIXES_APPLIED.md` Round 3.

## Added — Colab collection, Gradio demo & domain guides (2026-10)

### One-click Colab notebooks (12 new, beside `demo_colab.ipynb`)
| File | Covers |
|------|--------|
| `quickstart_colab.ipynb` | All 13 domains + seg/cd task picker, full auto pipeline |
| `flood_colab.ipynb` / `wildfire_colab.ipynb` | Flood & burn-scar mapping (seg + cd) |
| `tsunami_colab.ipynb` / `hurricane_colab.ipynb` | Inundation & damage change detection |
| `landslide_colab.ipynb` / `volcano_colab.ipynb` | Scar / lava-ash segmentation |
| `change_detection_colab.ipynb` | BIT/ChangeOS-style pairs + CHECKPOINT mode (TinyUNetSiam `.pth`) |
| `satellite_finetune_colab.ipynb` | QUICK-TRAIN: 3-minute real training, exports `tiny_unet_cd.pth` |
| `ai_images_colab.ipynb` | AI-illustration images, watermarked, no real photos (+ optional diffusion cell) |
| `demo_gif_colab.ipynb` | Regenerates `assets/demo.gif` for this README |
| `batch_inference_colab.ipynb` | Multi-image batch, `_pre/_post` pairing, contact sheet + CSV |

Every notebook: auto-run chain (setup → data → inference → dashboard → export), upload
widget buttons with a **preview of what to upload** shown before uploading, inline
result previews, 🛟 synthetic fallback (never blocks), mode-stamped results
(CHECKPOINT / QUICK-TRAIN / SYNTHETIC-DEMO). Validated by JSON+AST checks on all cells
and local end-to-end execution of the pipeline cells.

### Web, docs & assets
- `gradio_demo/` — NEW: Gradio web demo (`app.py`, `requirements.txt`, HF Spaces card);
  4 tabs (synthetic demo / change detection / segmentation / AI illustrations).
- `docs/domains/*.md` — NEW: 13 per-domain guides (run steps, upload checklist, sensors, datasets).
- `assets/demo.gif` — NEW: README demo GIF (5 procedural frames, 0.6 MB, photo-free).
- `README.md` — demo GIF embed, 12-notebook index with Colab badges, Gradio section, domain-guide links.
- `assets/CREDITS.md` — provenance for the GIF and web-demo gallery.

## Added — 9 extended disaster domains (Objective 2)

### Dataset adapters (src/datasets/)
| File | Purpose |
|------|---------|
| `src/datasets/_disaster_common.py` | NEW — shared loading helpers for the extended adapters (multi-format read, channel fixes, split lists, `DisasterDatasetBase`). Original adapters untouched. |
| `src/datasets/oilspill_sar.py` | NEW — Sentinel-1 oil-spill dataset (seg: sea/slick/look-alike; cd: new slick). 2-ch VV/VH per date. |
| `src/datasets/avalanche_slf.py` | NEW — avalanche debris segmentation (4-ch R/G/B/NIR, 2-class). |
| `src/datasets/volcano_pyraws.py` | NEW — volcanic features segmentation (3-ch RGB, 4-class: lava/vent/fumarole). |
| `src/datasets/tsunami_inundation.py` | NEW — tsunami inundation change detection (3-ch pairs, binary). |
| `src/datasets/hurricane_surge.py` | NEW — storm-surge change detection (2-ch SAR pairs, binary). |
| `src/datasets/glof_icimod.py` | NEW — glacial-lake dataset, **dual-mode seg + scd** (first scd-capable dataset+config in repo). 4-ch per date. |
| `src/datasets/duststorm_modis.py` | NEW — dust plume segmentation (3-ch RGB, 3-class: clear/dust/cloud). |
| `src/datasets/smoke_plume.py` | NEW — smoke plume segmentation (3-ch RGB, 3-class: clear/smoke/cloud). |
| `src/datasets/locust_habitat.py` | NEW — locust habitat/affected cropland (seg + cd dual-mode, 4-ch S2 pairs). |

### Registry
| File | Change |
|------|--------|
| `src/core/registry.py` | MODIFIED — added 9 `dataset_libs` entries (`oilspill_sar`, `avalanche_slf`, `volcano_pyraws`, `tsunami_inundation`, `hurricane_surge`, `glof_icimod`, `duststorm_modis`, `smoke_plume`, `locust_habitat`), lazy-import, same pattern as existing. No existing entry touched. |

### Configs (20 new YAMLs)
| Path | Configs |
|------|---------|
| `configs/oil_spill/oilspill_sar/` | `seg_unet.yaml`, `seg_segformer_b2.yaml`, `cd_bit.yaml` |
| `configs/avalanche/avalanche_slf/` | `seg_unet.yaml`, `seg_segformer_b2.yaml` |
| `configs/volcano/volcano_pyraws/` | `seg_unet.yaml`, `seg_segformer_b2.yaml` |
| `configs/tsunami/xbd_tsunami/` | `cd_bit.yaml`, `cd_changeos.yaml` |
| `configs/hurricane/s1_surge/` | `cd_bit.yaml`, `cd_changeos.yaml` |
| `configs/glof/icimod/` | `seg_unet.yaml`, `seg_segformer_b2.yaml`, `scd_changemamba_scd.yaml` (repo-first scd config) |
| `configs/dust_storm/modis_dust/` | `seg_unet.yaml`, `seg_segformer_b2.yaml` |
| `configs/smoke_plume/smoke_modis/` | `seg_unet.yaml`, `seg_segformer_b2.yaml` |
| `configs/locust/fao_s2/` | `seg_unet.yaml`, `cd_bit.yaml` |

### Data preparation (scripts/data_prep/)
| File | Purpose |
|------|---------|
| `_dummy_common.py` | NEW — shared synthetic dummy-data generator library (64×64 tiles, correct bands, deterministic). |
| `sanity_check.py` | NEW — generic per-domain dataset validator (layout, splits, label ranges, adapter round-trip). |
| `<9 domains>/make_dummy.py` | NEW — 9 per-domain dummy generators (oil_spill, avalanche, volcano, tsunami, hurricane, glof, dust_storm, smoke_plume, locust). |
| `<9 domains>/README.md` | NEW — 9 per-domain download + folder-layout + filelist guides with honest dataset-access notes. |
| `smoke_test_all_domains.py` (repo root `scripts/`) | NEW — end-to-end smoke test: dummy generation → 1-epoch train → test.py per domain (12 runs incl. cd/scd variants), JSON summary. |

### Colab demo (Objective 3)
| File | Purpose |
|------|---------|
| `demo_colab.ipynb` | NEW — zero-edit auto-running notebook: idempotent setup (GPU check + CPU fallback, pinned installs, repo acquisition via cwd/GitHub/zip), upload widgets (seg/cd modes, png/jpg/tif/h5) with **inline preview + suitability verdict** and bundled synthetic fallback so Run-all never blocks, 3 clearly-labeled model modes (QUICK-TRAIN 1–10 epochs on dummy data / user checkpoint / synthetic demo), results dashboard (overlays + legend, pre/post change grids, stats table with % and km², class/pie/loss/confidence plots, plain-language report), auto-export `results_analysis_<timestamp>.zip` with manifest and `files.download()`. |

### Illustrations (Objective 4 — copyright-safe)
| File | Method |
|------|--------|
| `assets/hero_banner.png` | AI-generated stylized illustration (fictional scene, no real imagery) |
| `assets/architecture_diagram.png` | Programmatically drawn (matplotlib) |
| `assets/colab_workflow.png` | Programmatically drawn (matplotlib) |
| `assets/domain_matrix.png` | Programmatically drawn (matplotlib) |
| `assets/synthetic_samples.png` | Programmatically drawn synthetic tiles |
| `assets/CREDITS.md` | NEW — credits + copyright statement |

### Docs (Objective 5)
| File | Change |
|------|--------|
| `README.md` | MODIFIED — badges (Python/PyTorch/Colab/License), AI hero banner, "Launch Colab" button, 9-domain table, beginner quickstart, **"Next steps you might want"** section, updated layout/dataset-prep sections, architecture diagram. |
| `docs/ADDITIONAL_DISASTERS.md` | NEW — 21 further disaster-domain candidates in 3 feasibility tiers with verified datasets and add-one instructions. |
| `CHANGELOG.md` | NEW — this file. |
| `CHANGES.md` | NEW — executive summary of the extension. |
| `GITHUB_UPLOAD_INSTRUCTIONS.md` | NEW — exact git + web-UI upload guide, Colab button setup, checklist. |
| `LICENSE` | REPLACED — placeholder text replaced with the full Apache-2.0 license text (matching `pyproject.toml`'s declared license). |

## Modified — original files
Only two original files were touched (`src/core/registry.py`, `README.md`);
both changes are purely additive. `pip install -e .`, `train.py`, and `test.py`
behavior for existing domains is unchanged (verified by smoke tests).

## Removed
Nothing.

## Fixed — Colab demo verification pass (end-to-end runtime audit)

`demo_colab.ipynb` was executed headlessly cell-by-cell (simulated Colab runtime)
in both segmentation and change-detection modes. Three real bugs were found and fixed:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Dummy-tile vs crop size | `CropSizeError: crop (256,256) exceeds image (64,64)` — quick-train crashed in both train and val loops | `quick_train()` now shrinks **every** crop transform (`RandomCrop`, `CenterCrop`, …) in **every** augmentation split to the 64 px dummy tile size |
| Channel-budget mismatch | `RuntimeError: expected 5 channels, got 3` — flood/cau UNet expects `in_channels=5` while uploads kept their native 3 bands | `preprocess_uploaded()` now repeats/trims the upload to the model's exact channel budget (auto from `in_channels`, or per-date `input_nc`); also scales 0-1 float inputs when Normalize stats are 0-255 based |
| cd models need two inputs | Siamese cd models (`BIT.forward(x1, x2)`) crashed on single-tensor inference | New `_forward()` helper calls `model(pre, post)` for two-tensor cd models with a concat fallback for single-tensor models (ChangeOS); RUN cell builds the pre tensor automatically |

Robustness hardening (independent re-run safety):
- Every interactive cell now creates its own target dirs (`/content/uploads`, `/content/checkpoints`, `/content/results/*`) so it works even when re-run standalone without the SETUP cell.
- The raw predicted mask is now saved (`predicted_masks/pred_mask.npy` + `.png`) so the EXPORT zip manifest is never missing it.

Verification evidence (CPU, simulated runtime):
- seg Run-all (flood, QUICK-TRAIN, 3 epochs): trained 6 steps, loss 0.49 → 0.38; genuine model prediction (128×128, classes [0,1], **no heuristic fallback**); overlay + legend + pie/loss + confidence histogram produced; export zip 18 entries; auto-download invoked.
- cd Run-all (tsunami, BIT siamese): trained on dummy pre/post pairs; two-tensor forward produced a real change mask; `pre_post_change.png` grid generated; export + download OK.
- Upload-cell interaction tests: auto-fallback on Run-all (never blocks), mocked button-click upload (saves, previews, extracts metadata), and cd verdict warning on wrong file count — all pass.
