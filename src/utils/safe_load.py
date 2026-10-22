"""Safe checkpoint loading.

``torch.load`` unpickles arbitrary Python objects unless ``weights_only=True``,
so loading an untrusted ``.pth`` can execute attacker-controlled code. Every
checkpoint read in this repository goes through :func:`safe_torch_load`, which
uses ``weights_only=True`` and refuses to fall back to full unpickling unless
the user explicitly opts in:

* environment variable ``ANYDISASTER_TRUST_CHECKPOINTS=1``, or
* ``safe_torch_load(..., trusted=True)`` for call sites that have verified the
  file themselves.

Some third-party pretrained files (older MAE/SatMAE releases, for example)
embed non-tensor objects such as ``argparse.Namespace``; those need the opt-in
and should only be used with files from a source you trust.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any

import torch

_LOG = logging.getLogger(__name__)
_ENV = "ANYDISASTER_TRUST_CHECKPOINTS"


def _env_trusted() -> bool:
    return os.environ.get(_ENV, "").strip().lower() in {"1", "true", "yes"}


def safe_torch_load(f: Any, map_location: Any = "cpu", trusted: bool = False, **kwargs: Any) -> Any:
    """``torch.load`` with ``weights_only=True`` and an explicit trust opt-in."""
    kwargs.pop("weights_only", None)
    try:
        return torch.load(f, map_location=map_location, weights_only=True, **kwargs)
    except (pickle.UnpicklingError, RuntimeError, AttributeError) as exc:
        # Only treat genuine "not loadable as plain weights" failures as such.
        msg = str(exc)
        looks_unsafe = isinstance(exc, pickle.UnpicklingError) or "weights_only" in msg
        if not looks_unsafe:
            raise
        if not (trusted or _env_trusted()):
            raise RuntimeError(
                "Refusing to unpickle this checkpoint: it contains objects that are not "
                "plain tensors/containers, and loading it with full pickle could execute "
                "arbitrary code. If you trust the file's source, set "
                f"{_ENV}=1 and retry.\nOriginal error: {msg[:300]}"
            ) from exc
        _LOG.warning("Loading checkpoint with weights_only=False (trusted opt-in).")
        if hasattr(f, "seek"):
            f.seek(0)
        return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
