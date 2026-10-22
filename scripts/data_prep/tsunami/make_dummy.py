#!/usr/bin/env python3
"""Generate a synthetic dummy dataset for the Tsunami inundation (RGB pair, binary cd) domain.

Creates small 64x64 samples with correct bands/channels so the full
train -> test pipeline can be smoke-tested WITHOUT the real dataset.

Usage:
    python scripts/data_prep/tsunami/make_dummy.py
    python scripts/data_prep/tsunami/make_dummy.py --out data/tsunami/DUMMY --size 64 --n-train 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _dummy_common import base_parser, build_ids, gen_pair_domain


DEFAULT_OUT = "data/XBD_TSUNAMI"

if __name__ == "__main__":
    args = base_parser("Tsunami inundation (RGB pair, binary cd) dummy-data generator", DEFAULT_OUT).parse_args()
    out = Path(args.out)
    ids = build_ids(args, prefix="tile")
    gen_pair_domain(out, args.size, ch=3, n_classes=2, ids=ids, seg=False)
