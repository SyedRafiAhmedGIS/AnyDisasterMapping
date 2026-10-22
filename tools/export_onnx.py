#!/usr/bin/env python3
"""Export any registered model to ONNX and/or TorchScript.

Bridges the gap between training (train.py) and deployment: builds the model
through the *same* registry path the trainer uses, loads a trained checkpoint,
and traces the forward pass exactly as the task handlers invoke it
(`src/tasks/{seg,cd,scd}.py`):

  seg      -> model(image)                       (dual-kwargs models honoured)
  cd / scd -> model(**{arg_names: (pre, post)})  for DUAL_INPUT_MODELS entries
           -> model(torch.cat([pre, post], 1))   otherwise

Usage:
    # from a trained experiment directory (config.yaml + best.pth)
    python tools/export_onnx.py --exp_path results/xbd/unet

    # from a config only (random weights, architecture check)
    python tools/export_onnx.py --config configs/oil_spill/oilspill_sar/cd_bit.yaml

    # dynamic H/W axes + onnxruntime verification + TorchScript
    python tools/export_onnx.py --exp_path results/_smoke/oil_spill \
        --dynamic --check --torchscript

Outputs land in --outdir (default exports/<name>/):
    model.onnx            ONNX graph (opset 18)
    model.torchscript.pt  TorchScript module (with --torchscript)
    export_manifest.json  shapes, hashes, export metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from src.utils.safe_load import safe_torch_load


# ---------------------------------------------------------------------------
# Config / model construction (mirrors src/core/trainer.py::_build_model)
# ---------------------------------------------------------------------------

def load_config(path: Path):
    from src.core.config import Config
    return Config.from_yaml(str(path))


def build_model(config):
    from src.core.registry import model_libs
    model_cfg = config.get("model", {})
    name = model_cfg.get("name") if isinstance(model_cfg, dict) else config.get("model_name")
    kwargs = model_cfg.get("kwargs", {}) if isinstance(model_cfg, dict) else config.get("model_kwargs", {})
    factory = model_libs[name]
    return name, factory(**kwargs)


def load_checkpoint(model, ckpt_path: Path, strict: bool = True) -> dict:
    state = safe_torch_load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict", "weights"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    try:
        model.load_state_dict(cleaned, strict=strict)
        return {"loaded": ckpt_path.name, "strict": strict, "tensors": len(cleaned)}
    except RuntimeError as e:
        if not strict:
            raise
        print(f"  ! strict load failed ({str(e)[:120]}...) — retrying non-strict")
        return load_checkpoint(model, ckpt_path, strict=False)


# ---------------------------------------------------------------------------
# Input spec (mirrors src/tasks/*.py run_model conventions)
# ---------------------------------------------------------------------------

def _in_channels(kwargs: dict) -> int:
    for key in ("in_channels", "input_nc", "in_ch", "num_channels"):
        if key in kwargs:
            return int(kwargs[key])
    return 3


def make_inputs(config, name: str, size: int, device: str = "cpu"):
    """Return (args_tuple, kw_names|None, input_names, example_tensors)."""
    from src.core.registry import DUAL_INPUT_MODELS

    task = config.get("task")
    model_cfg = config.get("model", {})
    kwargs = model_cfg.get("kwargs", {}) if isinstance(model_cfg, dict) else {}
    c = _in_channels(kwargs)
    info = DUAL_INPUT_MODELS.get(name, {})
    dual_kwargs = bool(info) and info.get("forward") == "dual"

    if task == "seg":
        img = torch.randn(1, c, size, size, device=device)
        if dual_kwargs:  # e.g. HyperSigma takes the image twice under 2 names
            names = list(info["arg_names"])
            return ({names[0]: img, names[1]: img},), names, [f"image_{n}" for n in names], [img]
        return (img,), None, ["image"], [img]

    # cd / scd
    pre = torch.randn(1, c, size, size, device=device)
    post = torch.randn(1, c, size, size, device=device)
    if dual_kwargs:
        names = list(info["arg_names"])
        return ({names[0]: pre, names[1]: post},), names, names, [pre, post]
    concat = torch.cat([pre, post], dim=1)
    return (concat,), None, ["pre_post_concat"], [concat]


def run_forward(model, args, kw_names):
    model.eval()
    with torch.no_grad():
        if kw_names:  # dict-style example
            return model(**args[0])
        return model(*args)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def _positional_wrapper(model: torch.nn.Module, names: list) -> torch.nn.Module:
    """Wrap a kwargs-forwarding model so the exported graph is positional."""
    class _Pos(torch.nn.Module):
        def __init__(self, m, order):
            super().__init__()
            self.m, self.order = m, order

        def forward(self, *tensors):
            return self.m(**dict(zip(self.order, tensors)))

    return _Pos(model.eval(), names).eval()


def export_onnx(model, args, kw_names, input_names, out_path: Path,
                dynamic: bool, opset: int) -> bool:
    """Returns True when the graph really carries dynamic axes."""
    if kw_names:
        export_module = _positional_wrapper(model, kw_names)
        export_args = tuple(args[0][n] for n in kw_names)
        in_names = kw_names
    else:
        export_module = model.eval()
        export_args = args
        in_names = input_names

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {n: {0: "batch", 2: "height", 3: "width"} for n in in_names}
        dynamic_axes["logits"] = {0: "batch"}

    try:
        torch.onnx.export(
            export_module, export_args, str(out_path),
            input_names=in_names, output_names=["logits"],
            opset_version=opset, do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    except RuntimeError as e:
        # torch>=2.9 cannot express dynamic H/W for reshape/attention-heavy
        # models via dynamic_axes. Fall back to a static graph (always
        # deployable) instead of hard-failing the export.
        if not dynamic:
            raise
        print(f"  ! dynamic-axes export not supported for this architecture "
              f"({str(e).splitlines()[0][:90]}) — falling back to STATIC shapes")
        dynamic = False
        torch.onnx.export(
            export_module, export_args, str(out_path),
            input_names=in_names, output_names=["logits"],
            opset_version=opset, do_constant_folding=True,
        )
    _fold_external_weights(out_path)
    return dynamic


def _fold_external_weights(out_path: Path) -> None:
    """torch>=2.9 new exporter may spill weights to <model>.onnx.data.

    Deployment-friendly default: fold external data back into a single-file
    ONNX protobuf (only possible while the model is < 2 GB, which covers
    every model in this zoo — the guard below keeps >2GB exports intact).
    """
    data_file = out_path.with_name(out_path.name + ".data")
    if not data_file.exists():
        return
    try:
        import onnx
        from onnx.external_data_helper import convert_model_from_external_data
        proto = onnx.load(str(out_path))  # pulls external tensors into memory
        convert_model_from_external_data(proto)
        onnx.save(proto, str(out_path), save_as_external_data=False)
        data_file.unlink()
        print("  • folded external weights into single-file ONNX")
    except Exception as e:  # noqa: BLE001 — keep external-data layout instead
        print(f"  • kept external-data layout ({data_file.name}); "
              f"reason: {type(e).__name__}: {str(e)[:80]}")


def export_torchscript(model, args, kw_names, out_path: Path):
    if kw_names:
        # torch>=2.14 removed trace(example_kwargs=...) — reuse the
        # positional wrapper so the traced graph keeps a clean signature.
        wrapped = _positional_wrapper(model, kw_names)
        traced = torch.jit.trace(wrapped, tuple(args[0][n] for n in kw_names))
    else:
        traced = torch.jit.trace(model.eval(), args)
    traced.save(str(out_path))
    return traced


def verify_torchscript(path: Path, args, kw_names) -> None:
    reloaded = torch.jit.load(str(path))
    reloaded.eval()
    with torch.no_grad():
        if kw_names:
            reloaded(*(args[0][n] for n in kw_names))
        else:
            reloaded(*args)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(p: Path) -> str:
    """Repo-relative when possible, absolute otherwise."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--exp_path", help="Trained experiment dir (config.yaml + checkpoint)")
    src.add_argument("--config", help="Config yaml (random weights — architecture check)")
    ap.add_argument("--checkpoint", default="best.pth")
    ap.add_argument("--outdir", default=None, help="Output dir (default exports/<name>)")
    ap.add_argument("--size", type=int, default=256, help="Square input size (default 256)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dynamic", action="store_true",
                    help="Export with dynamic batch/H/W axes (static by default)")
    ap.add_argument("--torchscript", action="store_true", help="Also trace a TorchScript module")
    ap.add_argument("--check", action="store_true",
                    help="Verify ONNX with onnxruntime + TorchScript reload (if available)")
    args = ap.parse_args()

    torch.manual_seed(0)

    if args.exp_path:
        exp = Path(args.exp_path)
        cfg_path = exp / "config.yaml"
        ckpt_path = exp / args.checkpoint
        if not cfg_path.exists():
            sys.exit(f"✗ {cfg_path} not found — is this an experiment dir?")
        config = load_config(cfg_path)
        name_tag = exp.name
    else:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            sys.exit(f"✗ {cfg_path} not found")
        config = load_config(cfg_path)
        ckpt_path = None
        name_tag = cfg_path.stem

    outdir = Path(args.outdir).resolve() if args.outdir else REPO / "exports" / name_tag
    outdir.mkdir(parents=True, exist_ok=True)

    name, model = build_model(config)
    task = config.get("task")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"◆ model={name}  task={task}  params={n_params / 1e6:.1f}M  size={args.size}px")

    ckpt_info = {"loaded": None}
    if ckpt_path and Path(ckpt_path).exists():
        ckpt_info = load_checkpoint(model, Path(ckpt_path))
        print(f"◆ checkpoint: {ckpt_path} (strict={ckpt_info.get('strict')})")
    elif ckpt_path:
        print(f"  ! checkpoint {ckpt_path} not found — exporting random weights")

    args_t, kw_names, input_names, tensors = make_inputs(config, name, args.size)
    ref = run_forward(model, args_t, kw_names)
    ref_out = ref[0] if isinstance(ref, (tuple, list)) else ref
    print(f"◆ forward OK — logits {tuple(ref_out.shape)}  inputs={input_names}")

    manifest = {
        "model": name, "task": task, "params_M": round(n_params / 1e6, 2),
        "input_names": input_names, "input_size": args.size, "opset": args.opset,
        "dynamic_axes": args.dynamic, "output_classes": int(ref_out.shape[1]),
        "checkpoint": ckpt_info, "outputs": {},
    }

    onnx_path = outdir / "model.onnx"
    print(f"◆ exporting ONNX (opset {args.opset}, {'dynamic' if args.dynamic else 'static'}) …")
    used_dynamic = export_onnx(model, args_t, kw_names, input_names, onnx_path,
                               args.dynamic, args.opset)
    manifest["dynamic_axes"] = used_dynamic
    manifest["outputs"]["onnx"] = {"path": _display_path(onnx_path),
                                   "bytes": onnx_path.stat().st_size,
                                   "sha256": sha256(onnx_path)}
    print(f"  ✓ {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if args.torchscript:
        ts_path = outdir / "model.torchscript.pt"
        print("◆ tracing TorchScript …")
        export_torchscript(model, args_t, kw_names, ts_path)
        manifest["outputs"]["torchscript"] = {"path": _display_path(ts_path),
                                              "bytes": ts_path.stat().st_size,
                                              "sha256": sha256(ts_path)}
        print(f"  ✓ {ts_path}  ({ts_path.stat().st_size / 1e6:.1f} MB)")

    if args.check:
        try:
            import numpy as np
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            feeds = {n: t.detach().numpy() for n, t in zip(input_names, tensors)}
            out = sess.run(None, feeds)[0]
            assert out.shape == tuple(ref_out.detach().numpy().shape), \
                f"shape mismatch: onnx {out.shape} vs torch {tuple(ref_out.shape)}"
            diff = float(np.abs(out - ref_out.detach().numpy()).max())
            print(f"  ✓ onnxruntime parity — max|Δ| = {diff:.2e}")
            manifest["outputs"]["onnx"]["parity_max_abs_diff"] = diff
        except ImportError:
            print("  ! onnxruntime not installed — skipped ONNX parity check "
                  "(pip install onnxruntime)")
        if args.torchscript:
            verify_torchscript(ts_path, args_t, kw_names)
            print("  ✓ TorchScript reload + forward OK")

    mpath = outdir / "export_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"◆ manifest → {mpath}")
    print("✓ export complete")


if __name__ == "__main__":
    main()
