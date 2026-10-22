"""Regression test: EVERY registered model / dataset / task module must import.

This test exists because `src/models/ChangeMamba/vmamba.py` shipped with an
IndentationError that silently killed both ChangeMamba factories: the smoke
suite skipped `glof_scd` on CPU-only hosts, so nothing ever imported the
broken module. Importing every registry target here makes any syntax error,
missing sibling module, or broken vendored dependency fail loudly in CI —
on every host, GPU or not.

Run:  pytest tests/test_registry_import_all.py -v
"""

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "src" / "core" / "registry.py"

_LAZY_CALL = re.compile(r'lazy_import\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')


def _registry_targets():
    """Extract every (module_path, class_name) pair from registry.py."""
    src = REGISTRY.read_text(encoding="utf-8")
    pairs = _LAZY_CALL.findall(src)
    assert pairs, "registry.py changed shape — update this extractor"
    return pairs


def test_registry_extractor_found_targets():
    pairs = _registry_targets()
    # >= 30 models + >= 20 datasets as of the 2026 extended release
    assert len(pairs) >= 50, f"only {len(pairs)} lazy targets found"


@pytest.mark.parametrize("module_path,class_name", _registry_targets())
def test_registry_module_imports(module_path, class_name):
    """Import the module; fail with the real traceback if it is broken."""
    mod = importlib.import_module(module_path)
    assert hasattr(mod, class_name), (
        f"{module_path} does not expose '{class_name}'"
    )


def test_task_registry_imports():
    from src.tasks import TASK_REGISTRY  # noqa: F401

    assert set(TASK_REGISTRY) >= {"seg", "cd", "scd"}


def test_changemamba_factories_import():
    """Explicit pins for the two factories killed by the vmamba.py bug."""
    m1 = importlib.import_module("src.models.ChangeMamba.ChangeMamba")
    m2 = importlib.import_module("src.models.ChangeMamba.ChangeMambaSCD")
    assert hasattr(m1, "ChangeMamba")
    assert hasattr(m2, "ChangeMambaSCD")
