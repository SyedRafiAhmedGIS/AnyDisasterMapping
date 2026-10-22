# Third-party notices

The original code in this repository is Apache-2.0 (see `LICENSE`). The
directories below contain vendored / adapted third-party code that keeps its
**own** license. Full license texts are in [`third_party_licenses/`](third_party_licenses/).

> **Important for commercial use.** Some components are *non-commercial /
> research-only* or carry custom terms. The repository as a whole is therefore
> **not** freely usable for commercial purposes as-is. To use it commercially,
> remove the restricted components (rows marked ⚠) and the registry entries /
> configs that depend on them, or obtain separate permission from the owners.
> This file is a good-faith inventory taken from file headers and upstream
> repositories; it is not legal advice — verify before redistributing.

| | Path(s) | Upstream | License | Text |
|---|---|---|---|---|
| ⚠ | `src/models/dinov3/`, `src/models/mask2former/` (files carrying the DINOv3 header) | facebookresearch/dinov3 | **DINOv3 License Agreement** (custom; redistribution only under that agreement, trade-controls/ITAR restrictions, attribution) | `DINOv3.LICENSE.md` |
| ⚠ | `src/models/SegFormer/mix_transformer.py`, `segformer_head.py` | NVlabs/SegFormer | **NVIDIA Source Code License** — use limited to *non-commercial* (research/evaluation) | `SegFormer.LICENSE` |
| ⚠ | `src/models/SatMAE/` | sustainlab-group/SatMAE (built on facebookresearch/mae) | **CC BY-NC 4.0** upstream | `SatMAE.LICENSE`, `MAE.LICENSE` |
| ⚠ | `src/models/SpectralGPT/` | danfenghong/IEEE_TPAMI_SpectralGPT (**verified 2026-10-22**) | SpectralGPT-authored code: **GPL-3.0**; MAE-derived parts: **CC BY-NC 4.0** (non-commercial) — treat the folder as research-only | `SpectralGPT.LICENSE`, `GPL-3.0.LICENSE`, `MAE.LICENSE` |
| ⚠ | `src/models/BIT/` | justchenhao/BIT_CD (**verified 2026-10-22**) | No LICENSE file upstream; README: **"non-commercial and research purposes only"** | `BIT.LICENSE` |
| ⚠ | `src/models/DSIFN/` | GeoZcx/…DSIFN (**verified 2026-10-22**) | No LICENSE file upstream; README: **"non-commercial and research purposes only"** | `DSIFN.LICENSE` |
| ⚠ | `src/models/ChangeOS.py` | Z-Zheng/ChangeOS (**verified 2026-10-22**) | **No upstream license** (GitHub license field: null) — all rights reserved by default | `ChangeOS.LICENSE` |
| ⚠ | `src/models/UNetFormer.py` | WangLibo1995/GeoSeg (**verified 2026-10-22**) | **GPL-3.0** (copy-left: distribution of combined works must stay GPL-compatible) | `UNetFormer-GeoSeg.LICENSE`, `GPL-3.0.LICENSE` |
|  | `src/models/sam2/` | facebookresearch/sam2 | Apache-2.0 | `SAM2.LICENSE` |
|  | `src/models/SAMDPT/` | facebookresearch/segment-anything | Apache-2.0 | `SAM.LICENSE` |
|  | `src/models/HRNet/` | HRNet-Semantic-Segmentation (Microsoft) | MIT | `HRNet.LICENSE` |
|  | `src/models/SkySense/swin_transformer_v2.py` | microsoft/Swin-Transformer | MIT | `SwinTransformer.LICENSE` |
|  | `src/models/HyperSigma/` | whu-sigma/hypersigma (**verified 2026-10-22**: Apache-2.0); embedded BEiT/unilm parts MIT | Apache-2.0 + MIT | `HyperSigma.LICENSE`, `BEiT-unilm.LICENSE` |
|  | `src/models/CLIP/` | openai/CLIP | MIT | `CLIP.LICENSE` |
|  | `src/models/ChangeMamba/` (Mamba parts) | state-spaces/mamba | Apache-2.0 | `Mamba.LICENSE` |
|  | `src/models/ChangeMamba/` (VMamba parts) | MzeroMiko/VMamba | MIT | `VMamba.LICENSE` |
|  | `src/models/ChangeMamba/` (ChangeMamba parts) | ChenHongruixuan/ChangeMamba | Apache-2.0 | `ChangeMamba.LICENSE` |
|  | `src/models/RS3Mamba/` | walking-shadow/Official_Remote_Sensing_Mamba (**verified 2026-10-22**) | Apache-2.0 | `RS3Mamba.LICENSE`, `Apache-2.0.LICENSE` |
|  | `src/models/FarSeg/`, `FarSeg.py`, `FarSegPP.py` | Z-Zheng/FarSeg (**verified 2026-10-22**); `FarSeg.py` is a repository-local re-implementation carrying its own MIT header | Apache-2.0 (folder) / MIT (single-file re-impl) | `FarSeg.LICENSE`, `Apache-2.0.LICENSE` |
|  | `src/models/SiamCRNN.py` | ChenHongruixuan/SiamCRNN | MIT | `SiamCRNN.LICENSE` |
|  | `src/common/lovasz_loss.py` | bermanmaxim/LovaszSoftmax | MIT | `Lovasz.LICENSE` |

## Repository-local re-implementations

`UNet.py`, `UNetPlusPlus.py`, `DeepLabV3Plus.py`, `FCN8s.py`,
`SwinUperNet.py`, `convnext.py`, `DinoV2DPT.py`, `DINOV2DPT_LoRA.py`,
`DamageFormer.py` and similar files carry a `LICENSE-HEADER` comment marking
them as **repository-local implementations** (original code, Apache-2.0) of
architectures described in the papers cited in their docstrings — no
upstream repository code was vendored for these files.

## Pretrained weights

Weights are **not** part of this repository and have their own terms
(e.g. DINOv3 weights: DINOv3 License; SAM/SAM2: Apache-2.0; others: see each
release page). Datasets referenced by the `scripts/data_prep/` adapters are
likewise subject to their own licenses.
