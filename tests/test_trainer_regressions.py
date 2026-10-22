"""End-to-end regressions for trainer / test.py fixes (tiny GLOF dummy data, CPU)."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/glof/icimod/seg_unet.yaml"


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{args}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    return r


def _train(tmp_path, *extra):
    data = tmp_path / "data"
    _run("scripts/data_prep/glof/make_dummy.py", "--out", data, "--size", "64",
         "--n-train", "4", "--n-val", "2", "--n-test", "2")
    out = tmp_path / "run"
    _run("train.py", "--config", CONFIG, "--override",
         "dataset.train.dataset_path", data, "dataset.train.data_list_path", data / "train.txt",
         "dataset.val.dataset_path", data, "dataset.val.data_list_path", data / "val.txt",
         "dataset.test.dataset_path", data, "dataset.test.data_list_path", data / "test.txt",
         "training.num_epochs", "1", "training.train_batch_size", "2",
         "training.eval_batch_size", "1", "training.num_workers", "0",
         "scheduler.name", "null",
         "augmentation.train.RandomCrop.height", "64", "augmentation.train.RandomCrop.width", "64",
         "output_dir", out, *extra)
    return data, out


def test_max_steps_smaller_than_epoch_still_validates_and_saves_best(tmp_path):
    # 4 samples / bs 2 = 2 batches per epoch; max_steps=1 used to skip eval entirely.
    _, out = _train(tmp_path, "training.max_steps", "1")
    assert (out / "best.pth").exists()
    assert (out / "latest.pth").exists()


def test_test_py_does_not_reuse_stale_results_across_splits(tmp_path):
    data, out = _train(tmp_path)
    _run("test.py", "--exp_path", out, "--checkpoint", "best.pth",
         "--test_splits", data / "test.txt")
    assert (out / "test_metrics_test.json").exists()
    assert (out / "test_metrics_test.json.meta").exists()
    # a different split must not be answered with the cached numbers
    r = _run("test.py", "--exp_path", out, "--checkpoint", "best.pth",
             "--test_splits", data / "val.txt")
    assert "[SKIP]" not in r.stdout
    assert (out / "test_metrics_val.json").exists()
    json.loads((out / "test_metrics_val.json").read_text())
