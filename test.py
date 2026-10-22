"""
Usage:
    python test.py --exp_path results/xbd/unet
    python test.py --exp_path results/xbd/unet --checkpoint latest.pth
    python test.py --exp_path results/xbd/unet --config configs/infra/xbd/unet.yaml
    python test.py --exp_path results/xbd/unet --force    # re-run even if results exist
    python test.py --exp_path results/urbansar/unet --test_splits data/flood/urbansar/test_jubba.txt data/flood/urbansar/test_nova.txt data/flood/urbansar/test_weihui.txt
"""
import argparse
import hashlib
import json
import os
from src.core.config import Config
from src.core.trainer import Trainer


def _serialize(metrics):
    """Convert metrics dict to JSON-serializable form."""
    out = {}
    for k, v in metrics.items():
        if hasattr(v, 'tolist'):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _save_and_print(metrics, save_path, label=""):
    """Save metrics to JSON and print scalar values."""
    with open(save_path, "w") as f:
        json.dump(_serialize(metrics), f, indent=2)
    prefix = f"  [{label}]" if label else " "
    print(f"{prefix} Saved to {save_path}")
    for k, v in metrics.items():
        if not hasattr(v, '__len__'):
            print(f"    {k}: {v:.4f}")


def _meta_path(metrics_path):
    return metrics_path + ".meta"


def _write_meta(metrics_path, meta):
    """Record what produced a metrics file (checkpoint, split, config hash)."""
    with open(_meta_path(metrics_path), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _is_fresh(metrics_path, ckpt_path, meta=None):
    """True if metrics_path is newer than ckpt_path AND was produced by the
    same checkpoint name / test split / config (when `meta` is given).

    Timestamps alone are not enough: re-running with a different --test_splits
    or --config would otherwise silently reuse stale numbers. A metrics file
    without a matching `.meta` sidecar (e.g. from an older version) is stale.
    """
    if not os.path.isfile(metrics_path):
        return False
    if not os.path.isfile(ckpt_path):
        return False
    if os.path.getmtime(metrics_path) < os.path.getmtime(ckpt_path):
        return False
    if meta is not None:
        try:
            with open(_meta_path(metrics_path)) as f:
                return json.load(f) == meta
        except (OSError, ValueError):
            return False
    return True


def _config_sha(config):
    blob = json.dumps(config._data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _set_test_split(config, split_path):
    """Point dataset.test at `split_path`, creating the section if absent
    (a detached `.get(..., {})` would silently drop the override)."""
    ds_cfg = config._data.setdefault("dataset", {})
    ds_cfg.setdefault("test", {})["data_list_path"] = split_path


def main():
    parser = argparse.ArgumentParser(description="Unified Disaster Tester")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Experiment output dir (loads config.yaml + checkpoint from here)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config (e.g. to use updated YAML with inference settings)")
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Checkpoint filename (default: best.pth)")
    parser.add_argument("--test_splits", type=str, nargs="+", default=None,
                        help="Multiple test split files to evaluate sequentially. "
                             "Results are saved per-split AND aggregated via confusion "
                             "matrix accumulation across all evaluated splits.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if test_metrics.json is newer than checkpoint")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.exp_path, args.checkpoint)

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_yaml(os.path.join(args.exp_path, "config.yaml"))

    cfg_sha = _config_sha(config)  # hash before test-split overrides / resume
    config.resume = ckpt_path

    def meta_for(split=None):
        return {"checkpoint": args.checkpoint, "split": split, "config_sha": cfg_sha}

    if args.test_splits and len(args.test_splits) > 1:
        # Multi-split evaluation with confusion matrix aggregation

        # Skip check: all per-split JSONs + average must be fresh
        if not args.force:
            all_fresh = True
            for sp in args.test_splits:
                sname = os.path.splitext(os.path.basename(sp))[0]
                mp = os.path.join(args.exp_path, f"test_metrics_{sname}.json")
                if not _is_fresh(mp, ckpt_path, meta_for(sp)):
                    all_fresh = False
                    break
            avg_path = os.path.join(args.exp_path, "test_metrics_average.json")
            avg_meta = meta_for(sorted(args.test_splits))
            if all_fresh and _is_fresh(avg_path, ckpt_path, avg_meta):
                print(f"[SKIP] All test results are fresh (newer than {args.checkpoint}). "
                      f"Use --force to re-run.")
                return

        trainer = Trainer(config)

        # Check if any split is stale — if so, force re-eval ALL splits
        # because we can't restore confusion matrices from saved scalar JSONs.
        any_stale = args.force
        if not args.force:
            for sp in args.test_splits:
                sname = os.path.splitext(os.path.basename(sp))[0]
                mp = os.path.join(args.exp_path, f"test_metrics_{sname}.json")
                if not _is_fresh(mp, ckpt_path, meta_for(sp)):
                    any_stale = True
                    break
            if any_stale:
                print("[INFO] Some splits are stale — re-evaluating ALL splits for correct aggregation.")

        all_metrics = {}
        cm_total = None

        for split_path in args.test_splits:
            split_name = os.path.splitext(os.path.basename(split_path))[0]

            # Per-split skip: only when ALL splits are fresh (any_stale is False)
            per_split_path = os.path.join(args.exp_path, f"test_metrics_{split_name}.json")
            if not any_stale and _is_fresh(per_split_path, ckpt_path, meta_for(split_path)):
                print(f"\n--- Test split: {split_name} [SKIP, fresh] ---")
                with open(per_split_path) as f:
                    all_metrics[split_name] = json.load(f)
                continue

            print(f"\n--- Test split: {split_name} ---")
            _set_test_split(config, split_path)

            trainer.task_handler.reset_metrics()
            metrics = trainer.evaluate("test")
            all_metrics[split_name] = metrics

            _save_and_print(metrics, per_split_path, label=split_name)
            _write_meta(per_split_path, meta_for(split_path))

            cm = getattr(trainer.task_handler.metrics_mgr, 'evaluator_total', None)
            if cm is not None and hasattr(cm, 'confusion_matrix'):
                cm_arr = cm.confusion_matrix.copy()
                cm_total = cm_arr if cm_total is None else cm_total + cm_arr

        # Aggregate: recompute metrics from summed confusion matrix
        if cm_total is not None:
            from src.common.evaluation_metrics import Evaluator
            from src.core.metrics import per_class_metrics

            num_classes = cm_total.shape[0]
            agg_eval = Evaluator(num_classes)
            agg_eval.confusion_matrix = cm_total

            pcm = per_class_metrics(cm_total)
            agg_metrics = {
                "OA": agg_eval.Pixel_Accuracy(),
                "mIoU": agg_eval.Mean_Intersection_over_Union(),
                "per_class_iou": pcm["iou"].tolist(),
                "per_class_f1": pcm["f1"].tolist(),
                "per_class_precision": pcm["precision"].tolist(),
                "per_class_recall": pcm["recall"].tolist(),
            }
            print("\n--- Test Average (confusion matrix aggregation) ---")
            avg_path = os.path.join(args.exp_path, "test_metrics_average.json")
            _save_and_print(agg_metrics, avg_path, label="average")
            _write_meta(avg_path, meta_for(sorted(args.test_splits)))
    else:
        # Single test split. With an explicit --test_splits the result goes to
        # test_metrics_<split>.json so different splits never overwrite each
        # other; without it the classic test_metrics.json is used.
        split = args.test_splits[0] if args.test_splits else None
        if split:
            sname = os.path.splitext(os.path.basename(split))[0]
            metrics_path = os.path.join(args.exp_path, f"test_metrics_{sname}.json")
        else:
            metrics_path = os.path.join(args.exp_path, "test_metrics.json")
        meta = meta_for(split)

        # Skip check
        if not args.force and _is_fresh(metrics_path, ckpt_path, meta):
            print(f"[SKIP] {metrics_path} is fresh (same checkpoint/split/config, newer than "
                  f"{args.checkpoint}). Use --force to re-run.")
            return

        trainer = Trainer(config)

        if split:
            _set_test_split(config, split)

        metrics = trainer.evaluate("test")
        _save_and_print(metrics, metrics_path)
        _write_meta(metrics_path, meta)


if __name__ == "__main__":
    main()
