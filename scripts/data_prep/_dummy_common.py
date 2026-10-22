#!/usr/bin/env python3
"""Shared synthetic dummy-data generator for the 9 extended disaster domains.

Produces small (default 64x64) GeoTIFF-like samples with the correct band
counts and label structure for each domain so the full train -> test
pipeline can be smoke-tested WITHOUT downloading the real datasets.

Every domain-specific ``make_dummy.py`` in this tree imports from here.
The generated imagery is procedurally drawn (random noise + blobs) and is
NOT real satellite data; it exists purely to validate wiring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def _save_tif(path: str, array: np.ndarray) -> None:
    if tifffile is None:
        raise ImportError("tifffile is required to write dummy TIFFs")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array)


def _draw_blobs(rng: np.random.Generator, shape, n_blobs, radius_range):
    """Return a boolean (H, W) canvas with random ellipse blobs."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_blobs):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(*radius_range, size=2)
        mask = (((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2) <= 1.0
        canvas |= mask
    return canvas


def make_image(rng, size, channels):
    """Random speckle/noise image (C, H, W) float32 with mild structure."""
    img = rng.normal(0.45, 0.12, size=(channels, size, size)).clip(0, 1)
    # add a few brighter/darker patches for realism
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(3):
        cy, cx = rng.integers(0, size, 2)
        r = rng.integers(size // 6, size // 2)
        spot = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        delta = rng.normal(0, 0.15)
        img[:, spot] += delta
    return img.clip(0, 1).astype(np.float32)


def make_label(rng, size, n_classes, class_weights=None, min_frac=0.02):
    """Random integer label map (H, W) with at least some foreground."""
    label = np.zeros((size, size), dtype=np.uint8)
    weights = class_weights or [1.0] * (n_classes - 1)
    # guarantee each foreground class appears
    for ci in range(1, n_classes):
        blob = _draw_blobs(rng, (size, size), n_blobs=2, radius_range=(2, max(3, size // 5)))
        label[blob] = ci
    # sprinkle extra noise pixels
    noise = rng.random((size, size)) < 0.03
    label[noise] = rng.integers(0, n_classes, size=noise.sum())
    return label


def write_split_lists(out_dir: Path, ids):
    for split, split_ids in (("train", ids["train"]), ("val", ids["val"]),
                             ("test", ids["test"])):
        with open(out_dir / f"{split}.txt", "w") as f:
            f.write("\n".join(split_ids) + "\n")


def default_ids(n_train=4, n_val=2, n_test=2, prefix="tile"):
    ids = [f"{prefix}_{i:04d}" for i in range(n_train + n_val + n_test)]
    return {"train": ids[:n_train],
            "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


# ---------------------------------------------------------------------------
# Domain generators
# ---------------------------------------------------------------------------

def gen_pair_domain(out, size, ch, n_classes, ids, seg=True, prefix="tile",
                    ch_post=None):
    """Domains with PRE/POST/GT layout (oil spill, tsunami, hurricane, locust).

    ``ch`` applies to PRE; ``ch_post`` (default = ch) applies to POST. This
    supports configs whose stacked input has an odd channel count (e.g. the
    CAU flood config's 5-channel input = 3 pre + 2 post).
    """
    if ch_post is None:
        ch_post = ch
    rng = np.random.default_rng(42)
    for split, split_ids in ids.items():
        for sid in split_ids:
            pre = make_image(rng, size, ch)
            post = make_image(rng, size, ch_post)
            _save_tif(str(out / "PRE" / f"{sid}.tif"), pre)
            _save_tif(str(out / "POST" / f"{sid}.tif"), post)
            if seg:
                gt = make_label(rng, size, n_classes)
            else:
                gt = make_label(rng, size, 2)
            _save_tif(str(out / "GT" / f"{sid}.tif"), gt)
    write_split_lists(out, ids)
    print(f"[dummy] {out}  ({sum(len(v) for v in ids.values())} samples, "
          f"{size}x{size}, {ch}-ch/date, task={'seg' if seg else 'cd'} "
          f"{n_classes}-class GT)")


def gen_single_domain(out, size, ch, n_classes, ids, prefix="tile"):
    """Domains with IMG/GT layout (avalanche, volcano, dust, smoke)."""
    rng = np.random.default_rng(42)
    for split, split_ids in ids.items():
        for sid in split_ids:
            img = make_image(rng, size, ch)
            gt = make_label(rng, size, n_classes)
            _save_tif(str(out / "IMG" / f"{sid}.tif"), img)
            _save_tif(str(out / "GT" / f"{sid}.tif"), gt)
    write_split_lists(out, ids)
    print(f"[dummy] {out}  ({sum(len(v) for v in ids.values())} samples, "
          f"{size}x{size}, {ch}-ch, {n_classes}-class GT)")


def gen_glof(out, size, ids, prefix="tile"):
    """GLOF: T1/T2/GT_CD/GT_T1/GT_T2 layout for seg + scd modes."""
    rng = np.random.default_rng(42)
    for split, split_ids in ids.items():
        for sid in split_ids:
            t1 = make_image(rng, size, 4)
            t2 = make_image(rng, size, 4)
            cd = make_label(rng, size, 2)
            t1l = make_label(rng, size, 3)   # land / water / snow-ice
            t2l = make_label(rng, size, 3)
            _save_tif(str(out / "T1" / f"{sid}.tif"), t1)
            _save_tif(str(out / "T2" / f"{sid}.tif"), t2)
            _save_tif(str(out / "GT_CD" / f"{sid}.tif"), cd)
            _save_tif(str(out / "GT_T1" / f"{sid}.tif"), t1l)
            _save_tif(str(out / "GT_T2" / f"{sid}.tif"), t2l)
    write_split_lists(out, ids)
    print(f"[dummy] {out}  ({sum(len(v) for v in ids.values())} samples, "
          f"{size}x{size}, 4-ch/date, seg+scd labels)")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def base_parser(description, default_out, dual_mode=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=str, default=default_out,
                   help=f"Output directory (default: {default_out})")
    p.add_argument("--size", type=int, default=64, help="Tile size (default 64)")
    p.add_argument("--n-train", type=int, default=4)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--n-test", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    if dual_mode:
        p.add_argument("--task", choices=["seg", "cd"], default="seg",
                       help="seg: multi-class GT; cd: binary change GT "
                            "(dual-mode domains such as oil spill / locust)")
    return p


def build_ids(args, prefix):
    return {"train": [f"{prefix}_train_{i:03d}" for i in range(args.n_train)],
            "val": [f"{prefix}_val_{i:03d}" for i in range(args.n_val)],
            "test": [f"{prefix}_test_{i:03d}" for i in range(args.n_test)]}
