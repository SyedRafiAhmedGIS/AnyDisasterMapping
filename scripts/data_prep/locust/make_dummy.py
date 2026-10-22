#!/usr/bin/env python3
"""Generate a synthetic dummy dataset for the Locust habitat (Sentinel-2 4-ch; seg: habitat/affected or cd: new affected) domain.

Creates small 64x64 samples with correct bands/channels so the full
train -> test pipeline can be smoke-tested WITHOUT the real dataset.

Usage:
    python scripts/data_prep/locust/make_dummy.py
    python scripts/data_prep/locust/make_dummy.py --task cd   # binary change GT
    python scripts/data_prep/locust/make_dummy.py --out data/locust/DUMMY --size 64
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _dummy_common import base_parser, build_ids, gen_pair_domain


DEFAULT_OUT = "data/locust/FAO_S2"

if __name__ == "__main__":
    args = base_parser("Locust habitat (Sentinel-2 4-ch; seg: habitat/affected or cd: new affected) dummy-data generator", DEFAULT_OUT,
                       dual_mode=True).parse_args()
    # cd mode -> binary change GT in a dedicated _CD directory
    if args.task == "cd" and not args.out.endswith("_CD"):
        args.out = args.out.rstrip("/") + "_CD"
    out = Path(args.out)
    ids = build_ids(args, prefix="tile")
    n_cls = 2 if args.task == "cd" else 2
    gen_pair_domain(out, args.size, ch=4, n_classes=n_cls, ids=ids,
                    seg=(args.task == "seg"))
