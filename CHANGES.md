# CHANGES.md — Executive Summary

**What:** Extended the AnyDisasterMapping benchmark from 4 disaster domains to 13,
added a zero-edit Colab demo, copyright-safe illustrations, full documentation,
and GitHub-ready packaging.

**Before:** flood / infrastructure damage / landslide / wildfire; 31 models; 15 datasets;
no demo, no smoke tests, placeholder LICENSE.

**After:** everything above **plus**:

| Addition | Count / detail |
|----------|----------------|
| New disaster domains | 9 (oil spill, avalanche, volcano, tsunami, hurricane surge, GLOF, dust storm, smoke plume, locust) |
| New dataset adapters | 9 files + 1 shared helper (`src/datasets/`) |
| New configs | 20 YAMLs (lightweight + modern per domain; first-ever `task: scd` config) |
| New scripts | 9 dummy generators, sanity checker, `smoke_test_all_domains.py` (12-run matrix) |
| Colab notebook | `demo_colab.ipynb` — auto-setup, upload widget with inline preview, 3 honest modes, results dashboard, auto-export zip |
| Illustrations | 5 assets — AI hero + 4 programmatic diagrams, zero copyrighted material |
| Docs | README overhaul (badges/hero/Colab button/Next steps), `docs/ADDITIONAL_DISASTERS.md` (21 more candidates), CHANGELOG, CHANGES, GITHUB_UPLOAD_INSTRUCTIONS |
| License | Placeholder replaced with full Apache-2.0 text |

**Compatibility:** fully backward compatible. Only `src/core/registry.py`
(9 additive entries) and `README.md` were modified among original files.
`pip install -e .`, `train.py`, `test.py` work exactly as before.

**Verification:** `python scripts/smoke_test_all_domains.py` trains + tests every
new domain for 1 epoch on synthetic data — no real dataset download required.
