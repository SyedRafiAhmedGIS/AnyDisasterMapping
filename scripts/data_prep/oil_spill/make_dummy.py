#!/usr/bin/env python3
"""Generate a synthetic dummy dataset for the Oil spill (SAR VV/VH; seg: sea/slick/look-alike or cd: new slick) domain.

Creates small 64x64 samples with correct bands/channels so the full
train -> test pipeline can be smoke-tested WITHOUT the real dataset.

Usage:
    python scripts/data_prep/oil_spill/make_dummy.py
    python scripts/data_prep/oil_spill/make_dummy.py --task cd   # binary change GT
    python scripts/data_prep/oil_spill/make_dummy.py --out data/oil_spill/DUMMY --size 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _dummy_common import base_parser, build_ids, gen_pair_domain


DEFAULT_OUT = "data/oil_spill/OILSPILL"

if __name__ == "__main__":
    args = base_parser("Oil spill (SAR VV/VH; seg: sea/slick/look-alike or cd: new slick) dummy-data generator", DEFAULT_OUT,
                       dual_mode=True).parse_args()
    # cd mode -> binary change GT in a dedicated _CD directory
    if args.task == "cd" and not args.out.endswith("_CD"):
        args.out = args.out.rstrip("/") + "_CD"
    out = Path(args.out)
    ids = build_ids(args, prefix="tile")
    n_cls = 2 if args.task == "cd" else 3
    gen_pair_domain(out, args.size, ch=2, n_classes=n_cls, ids=ids,
                    seg=(args.task == "seg"))
