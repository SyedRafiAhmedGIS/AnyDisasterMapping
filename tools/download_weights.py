#!/usr/bin/env python3
"""Download pretrained weights from the project's Hugging Face release.

The companion weights repository is:

    https://huggingface.co/SyedRafiAhmedGIS/AnyDisasterMapping-weights

Layout convention (mirrored by checkpoint_zoo_colab.ipynb):
    <model>.pth                     flat names at the repo root, e.g. unet.pth
    seg/<model>.pth                 optional task-prefixed layout (also accepted)
    cd/<model>.pth · scd/<model>.pth

Usage:
    python tools/download_weights.py --list
    python tools/download_weights.py --model unet
    python tools/download_weights.py --model bit --task cd --dest checkpoints/
    python tools/download_weights.py --all --task seg      # keeps seg/ subdir
    python tools/download_weights.py --model unet --revision <commit-sha> \\
        --expect-sha256 <hex>

Integrity / trust
-----------------
* Every download prints its SHA-256. If the repo publishes a ``SHA256SUMS``
  file it is verified automatically; ``--expect-sha256`` pins a single file.
* ``--revision`` pins the repo to an exact commit (default: ``main``).
* Checkpoints are loaded with ``weights_only=True`` by this codebase, so a
  tampered pickle is rejected rather than executed — but still only download
  weights from a repository you trust.

Exit codes: 0 = ok, or the release is simply not published yet (a friendly
hint is printed); 1 = genuine failure (network/HF error, no matching weights,
any failed or unverifiable download).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ID = "SyedRafiAhmedGIS/AnyDisasterMapping-weights"
REPO_URL = f"https://huggingface.co/{REPO_ID}"
WEIGHT_SUFFIXES = (".pth", ".pt", ".safetensors")


def _hf():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        return HfApi(), hf_hub_download
    except ImportError:
        sys.exit("✗ huggingface_hub is required (pip install huggingface_hub) — "
                 "it is already a core dependency of this package; your "
                 "environment may be missing it.")


def _is_not_found(exc: Exception) -> bool:
    """True for a genuine 404 / repo-not-found from the Hub (no substring guessing)."""
    try:
        from huggingface_hub.utils import RepositoryNotFoundError
        if isinstance(exc, RepositoryNotFoundError):
            return True
    except ImportError:
        pass
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 404


def _not_published_hint() -> None:
    print(f"ℹ The weights release '{REPO_ID}' is not reachable yet.")
    print("  This is expected until the release is published under this account.")
    print("\n  Meanwhile you can still get real weights by:")
    print("   1. training any config for a few epochs:  python train.py --config <yaml>")
    print("      (checkpoints land in results/<run>/best.pth)")
    print("   2. using checkpoint_zoo_colab.ipynb — it trains a real tiny model")
    print("      in-session when the release is unavailable")
    print("\n  Publishing your own release (one command per file):")
    print(f"      huggingface-cli upload {REPO_ID} best.pth unet.pth "
          f"--repo-type model")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sums(text: str) -> dict[str, str]:
    """Parse `sha256sum`-style lines: '<hex>  <relative/path>'."""
    sums: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].strip().lstrip("*")] = parts[0].lower()
    return sums


def list_files(revision: str) -> int:
    api, _ = _hf()
    try:
        files = sorted(api.list_repo_files(REPO_ID, revision=revision))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            _not_published_hint()
            return 0
        print(f"✗ could not reach the Hugging Face Hub: {type(e).__name__}: {e}")
        return 1
    if not files:
        print(f"ℹ '{REPO_ID}' is reachable but empty.")
        return 0
    print(f"◆ {REPO_URL}  (revision: {revision})")
    print(f"  {len(files)} file(s):")
    for f in files:
        print(f"    {f}")
    return 0


def _select(weights: list[str], model: str, task: str | None, all_models: bool):
    if all_models:
        return [f for f in weights if f.startswith(f"{task}/")] if task else weights
    stem = Path(model).stem
    chosen = [f for f in weights if Path(f).stem == stem]
    if task:
        prefixed = [f for f in chosen if f.startswith(f"{task}/")]
        chosen = prefixed or chosen
    else:
        # No task given: prefer the flat root-level file so seg/cd/scd
        # variants of the same model name are never fetched together.
        flat = [f for f in chosen if "/" not in f]
        chosen = flat or chosen
    return chosen


def download(model: str, task: str | None, dest: Path, all_models: bool,
             revision: str = "main", expect_sha256: str | None = None) -> int:
    api, hf_hub_download = _hf()
    try:
        files = sorted(api.list_repo_files(REPO_ID, revision=revision))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            _not_published_hint()
            return 0
        print(f"✗ could not reach the Hugging Face Hub: {type(e).__name__}: {e}")
        return 1

    weights = [f for f in files if f.endswith(WEIGHT_SUFFIXES)]
    chosen = _select(weights, model, task, all_models)
    if not chosen:
        print("✗ no weights matching "
              + ("--all" if all_models else f"model='{model}'")
              + (f" task='{task}'" if task else "") + f" in {REPO_ID}")
        print("  run `python tools/download_weights.py --list` to see "
              "what is published")
        return 1
    if not all_models and len(chosen) > 1:
        print(f"✗ '{model}' matches several files ({', '.join(chosen)}); "
              "add --task seg|cd|scd to pick one.")
        return 1
    if expect_sha256 and len(chosen) != 1:
        print("✗ --expect-sha256 only applies to a single file (use --model).")
        return 1

    sums: dict[str, str] = {}
    sums_name = next((f for f in files if Path(f).name == "SHA256SUMS"), None)
    if sums_name:
        try:
            sums = _parse_sums(Path(hf_hub_download(
                repo_id=REPO_ID, filename=sums_name, revision=revision)).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not read {sums_name}: {type(e).__name__}: {str(e)[:100]}")
    else:
        print("ℹ no SHA256SUMS published in the release — files cannot be verified "
              "against a manifest (hashes are printed for manual checking).")

    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in chosen:
        # --all keeps the repo's relative layout so seg/unet.pth and
        # cd/unet.pth can never overwrite each other; single-model downloads
        # stay flat for convenience.
        target = dest / f if all_models else dest / Path(f).name
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"◆ downloading {f} → {target} …")
        try:
            path = Path(hf_hub_download(repo_id=REPO_ID, filename=f, revision=revision))
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ download failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        digest = _sha256(path)
        expected = (expect_sha256.lower() if expect_sha256 else sums.get(f))
        if expected and digest != expected:
            print(f"  ✗ SHA-256 mismatch for {f}: got {digest}, expected {expected} — "
                  "NOT saving this file.")
            continue
        if path.resolve() != target.resolve():
            shutil.copyfile(path, target)
        size_mb = target.stat().st_size / 1e6
        status = "verified" if expected else "unverified"
        print(f"  ✓ saved {target}  ({size_mb:.1f} MB, sha256 {digest[:16]}… {status})")
        ok += 1
    print(f"\n◆ {ok}/{len(chosen)} checkpoint(s) in {dest}/ — point "
          f"test.py / tools/export_onnx.py at them via --checkpoint")
    return 0 if ok == len(chosen) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="List published weight files")
    ap.add_argument("--model", help="Model name as registered in src/core/registry.py")
    ap.add_argument("--task", choices=("seg", "cd", "scd"),
                    help="Prefer task-prefixed files (seg/, cd/, scd/); with --all, "
                         "download only that task's folder")
    ap.add_argument("--all", action="store_true", help="Download every published weight")
    ap.add_argument("--dest", default="checkpoints", help="Target dir (default checkpoints/)")
    ap.add_argument("--revision", default="main",
                    help="Hub revision (branch/tag/commit SHA) to read; pin a commit for "
                         "reproducible, tamper-evident downloads")
    ap.add_argument("--expect-sha256", help="Expected SHA-256 of the single --model file")
    args = ap.parse_args()

    if args.list:
        return list_files(args.revision)
    if args.all:
        return download("all", args.task, Path(args.dest), True, args.revision)
    if args.model:
        return download(args.model, args.task, Path(args.dest), False,
                        args.revision, args.expect_sha256)
    ap.error("nothing to do — use --list, --model <name> or --all")
    return 2


if __name__ == "__main__":
    sys.exit(main())
