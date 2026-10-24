# Contributing to AnyDisasterMapping — extended

Thanks for helping improve the codebase! This guide gets you from clone to
green CI in a few minutes and explains the repo's house rules.

## 1. Development setup

```bash
git clone https://github.com/SyedRafiAhmedGIS/AnyDisasterMapping.git
cd AnyDisasterMapping
python -m venv .venv && source .venv/bin/activate   # 3.10–3.12
pip install -e . -c constraints.txt                 # constraints pin the verified set
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # dev on CPU
```

`constraints.txt` is load-bearing: it documents *why* every pin exists
(e.g. transformers 4.57.0 was yanked from PyPI; numpy stays on 1.26.x for
the vendored model code). If you must change a pin, change it **with** a
comment and re-run the full smoke suite.

## 2. What CI expects (run locally before pushing)

| command | what it guards |
|---------|----------------|
| `ruff check .` | first-party lint (vendored dirs excluded via `pyproject`) |
| `python scripts/validate_notebooks.py` | notebook JSON/AST, dedup contract, `KIT_REV` pin, README listing |
| `python -W error::SyntaxWarning -m compileall -q src scripts tests tools gradio_demo train.py test.py` | no syntax warnings anywhere |
| `pytest tests/ -q` | registry imports, safe-loading, trainer regressions, Colab-kit units |
| `python scripts/smoke_test_all_domains.py` | 1-epoch train + test.py on all CPU domains |
| CI placeholder sweep (`ci.yml`) | no leftover placeholder tokens in tracked files |

CI runs the first four on every push/PR across torch 2.5.1 + 2.14.1; the
smoke suite runs in the `smoke-one-domain` job.

## 3. House rules

### Vendored model code is (mostly) frozen
`src/models/<VendorDir>/` carries third-party implementations with
LICENSE-HEADER blocks and matching texts in `third_party_licenses/`.
Do **not** refactor vendored dirs for style; only touch them for correctness
fixes, and note the upstream repo in the PR. First-party code
(`src/core`, `src/datasets`, `src/utils`, `src/models/*.py` top-level,
`tools/`, `scripts/`, `tests/`) is ruff-gated.

### Notebooks share one core
The 17 Colab notebooks must not inline the shared engine anymore — they
import `tools/colab_kit.py` (torch-free, unit-tested in
`tests/test_colab_kit.py`). If you change kit behaviour, the equivalence
expectations are: same console output, same saved files, same STATE
transitions. `scripts/validate_notebooks.py` fails on duplicated shared
bodies, and the kit is pinned per release via `KIT_REV` — bump **both**
the kit version string and the notebooks' bootstrap pin together (one PR).

### Licensing is research-only
The bundle is Apache-2.0 core with non-commercial/GPL components; the
decision and inventory live in `README.md#license`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`. New vendored components need: verified upstream
license text in `third_party_licenses/`, a LICENSE-HEADER on each file,
and an entry in `THIRD_PARTY_NOTICES.md`.

### Honesty in demos
Any demo output that is not a trained model must stay labeled
`SYNTHETIC-DEMO`. Never present heuristic/synthetic results as model
predictions.

## 4. Adding a model / dataset / domain

Follow `docs/EXTENSION_GUIDE.md` (registry entries, config template,
per-domain synthetic scene in `tools/colab_kit.py` + `scripts/scene_gen.py`
kept in sync). The registry import-all test will fail if a new entry is
not importable; the smoke suite will fail if a new domain lacks a config.

## 5. Pull requests

1. Fork + branch (`feat/…`, `fix/…`).
2. Keep PRs focused; one feature or fix per PR.
3. Fill the PR template (what/why, tests run, license impact if any).
4. New tests for new behaviour — the suite is fast (≈40 s CPU).

## 6. Questions

Open a GitHub issue or email alsyedrafiahmed@gmail.com.
