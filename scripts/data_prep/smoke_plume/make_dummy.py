#!/usr/bin/env python3
"""Generate a synthetic dummy dataset for the Smoke plumes (RGB, clear/smoke/cloud) domain.

Creates small 64x64 samples with correct bands/channels so the full
train -> test pipeline can be smoke-tested WITHOUT the real dataset.

Usage:
    python scripts/data_prep/smoke_plume/make_dummy.py
    python scripts/data_prep/smoke_plume/make_dummy.py --out data/smoke_plume/DUMMY --size 64 --n-train 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _dummy_common import base_parser, build_ids, gen_single_domain


DEFAULT_OUT = "data/SMOKE_MODIS"

if __name__ == "__main__":
    args = base_parser("Smoke plumes (RGB, clear/smoke/cloud) dummy-data generator", DEFAULT_OUT).parse_args()
    out = Path(args.out)
    ids = build_ids(args, prefix="tile")
    gen_single_domain(out, args.size, ch=3, n_classes=3, ids=ids)
