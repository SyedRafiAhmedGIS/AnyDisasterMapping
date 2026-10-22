"""Tests for the opt-in training.deterministic flag (Round 4).

The flag must default OFF (no behaviour change for existing runs) and, when
enabled, actually flip the torch determinism knobs.
"""
import torch

from src.core.config import Config
from src.core.trainer import _apply_determinism


def _cfg(deterministic):
    return Config({"training": {"deterministic": deterministic}})


def test_default_is_off():
    """Default (flag absent) must be a no-op — existing runs unchanged."""
    cfg = Config({})
    torch.backends.cudnn.benchmark = True  # simulate a default-speed run
    assert _apply_determinism(cfg) is False
    assert torch.backends.cudnn.benchmark is True  # untouched
    assert torch.backends.cudnn.deterministic is False


def test_flag_off_is_noop():
    cfg = _cfg(False)
    torch.backends.cudnn.benchmark = True
    try:
        assert _apply_determinism(cfg) is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.backends.cudnn.benchmark = False


def test_flag_on_enables_determinism():
    cfg = _cfg(True)
    try:
        assert _apply_determinism(cfg) is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        # leave the process in the default state for other tests
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
