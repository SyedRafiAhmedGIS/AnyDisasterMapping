#!/usr/bin/env python3
"""Sanity-check a prepared (or dummy) dataset for any extended domain.

Validates folder layout, file presence, label value ranges, channel counts,
and — when the repo is importable — instantiates the dataset adapter and
pulls one sample through the full transform path.

Usage:
    python scripts/data_prep/sanity_check.py --domain glof --path data/glof/ICIMOD
    python scripts/data_prep/sanity_check.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LAYOUTS = {
    # domain: (dirs required, image dir for channel check, expected seg classes or None)
    "oil_spill":   {"dirs": ["PRE", "POST", "GT"], "img_dir": "PRE", "gt_dir": "GT"},
    "avalanche":   {"dirs": ["IMG", "GT"], "img_dir": "IMG", "gt_dir": "GT"},
    "volcano":     {"dirs": ["IMG", "GT"], "img_dir": "IMG", "gt_dir": "GT"},
    "tsunami":     {"dirs": ["PRE", "POST", "GT"], "img_dir": "PRE", "gt_dir": "GT"},
    "hurricane":   {"dirs": ["PRE", "POST", "GT"], "img_dir": "PRE", "gt_dir": "GT"},
    "glof":        {"dirs": ["T1", "T2", "GT_CD", "GT_T1", "GT_T2"], "img_dir": "T1", "gt_dir": "GT_T2"},
    "dust_storm":  {"dirs": ["IMG", "GT"], "img_dir": "IMG", "gt_dir": "GT"},
    "smoke_plume": {"dirs": ["IMG", "GT"], "img_dir": "IMG", "gt_dir": "GT"},
    "locust":      {"dirs": ["PRE", "POST", "GT"], "img_dir": "PRE", "gt_dir": "GT"},
}

# adapter registry names + expected tuple lengths per task
ADAPTERS = {
    "oil_spill":   ("oilspill_sar", {"seg": 3, "cd": 4}),
    "avalanche":   ("avalanche_slf", {"seg": 3}),
    "volcano":     ("volcano_pyraws", {"seg": 3}),
    "tsunami":     ("tsunami_inundation", {"cd": 4}),
    "hurricane":   ("hurricane_surge", {"cd": 4}),
    "glof":        ("glof_icimod", {"seg": 3, "scd": 6}),
    "dust_storm":  ("duststorm_modis", {"seg": 3}),
    "smoke_plume": ("smoke_plume", {"seg": 3}),
    "locust":      ("locust_habitat", {"seg": 3, "cd": 4}),
}


def check(domain: str, path: Path, deep: bool = True) -> bool:
    ok = True
    lay = LAYOUTS[domain]

    print(f"\n=== Sanity check: {domain} @ {path} ===")

    # 1. directory layout
    for d in lay["dirs"]:
        p = path / d
        n = len(list(p.glob("*"))) if p.is_dir() else 0
        status = "ok" if n > 0 else "MISSING/EMPTY"
        if n == 0:
            ok = False
        print(f"  dir {d:<8}: {n:>4} files  [{status}]")

    # 2. split files
    for split in ("train.txt", "val.txt", "test.txt"):
        f = path / split
        n = len(f.read_text().split()) if f.is_file() else 0
        status = "ok" if n > 0 else "MISSING"
        if n == 0:
            ok = False
        print(f"  split {split:<10}: {n:>4} ids  [{status}]")

    if not deep:
        return ok

    # 3. label value range (picks the first GT file)
    gt_dir = path / lay["gt_dir"]
    gt_files = sorted(gt_dir.glob("*")) if gt_dir.is_dir() else []
    if gt_files:
        try:
            import tifffile
            gt = np.asarray(tifffile.imread(gt_files[0]))
            vals = np.unique(gt[gt != 255])
            print(f"  GT[{gt_files[0].name}] unique labels (excl. 255): {vals.tolist()}")
        except Exception as e:  # noqa: BLE001
            print(f"  GT read failed: {e}")
            ok = False

    # 4. adapter round-trip
    try:
        from src.core.registry import dataset_libs
        name, tasks = ADAPTERS[domain]
        task = "seg" if "seg" in tasks else ("cd" if "cd" in tasks else "scd")
        cls_mod = dataset_libs[name]
        import importlib
        cells = [c.cell_contents for c in (cls_mod.__closure__ or [])]
        mod = importlib.import_module(cells[1])
        cls = getattr(mod, cells[0])

        kwargs = {"dataset_path": str(path),
                  "data_list_path": str(path / "train.txt"),
                  "split": "train"}
        if task in ("seg", "cd") and "seg" in tasks and "cd" in tasks:
            kwargs["task"] = task
        if task == "scd":
            kwargs["task"] = "scd"
        ds = cls(**kwargs)
        sample = ds[0]
        shapes = [getattr(s, "shape", None) for s in sample[:-1]]
        print(f"  adapter[{name}] task={task}: {len(sample)}-tuple, shapes={shapes}")
    except Exception as e:  # noqa: BLE001
        print(f"  adapter round-trip FAILED: {e}")
        ok = False

    print(f"  ==> {domain}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=sorted(LAYOUTS.keys()))
    ap.add_argument("--path", type=str, help="Dataset root directory")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for d, lay in LAYOUTS.items():
            print(f"  {d:<12} requires dirs: {', '.join(lay['dirs'])}")
        return

    if not args.domain or not args.path:
        ap.error("--domain and --path are required (or use --list)")

    ok = check(args.domain, Path(args.path))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
