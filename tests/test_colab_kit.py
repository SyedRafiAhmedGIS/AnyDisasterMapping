"""Torch-free regression tests for tools/colab_kit.py — the shared core that
the 17 Colab notebooks import.  Guards the Round-5 deduplication refactor:
if the kit ever drifts from the behaviour the notebooks rely on, these fail."""
import importlib
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
colab_kit = importlib.import_module("colab_kit")

DOMAINS = ["flood", "wildfire", "tsunami", "hurricane", "landslide", "volcano",
           "oil_spill", "avalanche", "glof", "dust_storm", "smoke_plume",
           "locust", "infra_damage", "generic"]


# ---------------------------------------------------------------------------
# Synthetic scene engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("domain", DOMAINS)
def test_make_synthetic_shapes_and_determinism(domain):
    pre, post, mask = colab_kit.make_synthetic(domain, 96, 128, seed=7)
    assert pre.shape == (128, 96, 3) and pre.dtype in (np.float32, np.float64)
    assert post.shape == (128, 96, 3) and post.dtype in (np.float32, np.float64)
    assert mask.shape == (128, 96) and mask.dtype == bool
    assert pre.min() >= 0 and pre.max() <= 1 and post.min() >= 0 and post.max() <= 1
    pre2, post2, mask2 = colab_kit.make_synthetic(domain, 96, 128, seed=7)
    assert np.array_equal(pre, pre2) and np.array_equal(post, post2)
    assert np.array_equal(mask, mask2)


def test_make_synthetic_changes_something():
    """The hazard must actually alter the scene (post != pre, mask non-trivial)."""
    for d in ("flood", "wildfire", "tsunami"):
        pre, post, mask = colab_kit.make_synthetic(d, 96, 128, seed=7)
        assert not np.array_equal(pre, post)
        assert 0 < mask.sum() < mask.size


def test_make_synthetic_unknown_domain_falls_back():
    pre, post, mask = colab_kit.make_synthetic("not_a_domain", 64, 64, seed=1)
    pre_f, _, _ = colab_kit.make_synthetic("flood", 64, 64, seed=1)
    assert np.array_equal(pre, pre_f)


# ---------------------------------------------------------------------------
# Heuristics / stats
# ---------------------------------------------------------------------------
def test_stats_for():
    mask = np.zeros((10, 10), bool)
    mask[:5] = True
    st = colab_kit.stats_for(mask, gsd_m=10.0)
    assert st["changed_px"] == 50
    assert st["changed_pct"] == 50.0
    assert st["km2"] == pytest.approx(0.005)  # 50 px * 10 m * 10 m = 5000 m^2


def test_otsu_bimodal():
    rng = np.random.default_rng(0)
    x = np.concatenate([0.2 + rng.normal(0, 0.05, 500),
                        0.8 + rng.normal(0, 0.05, 500)]).clip(0, 1)
    t = colab_kit._otsu(x)
    assert 0.35 < t < 0.65


def test_infer_returns_labeled_demo_mask():
    pre, post, _ = colab_kit.make_synthetic("flood", 64, 64, seed=3)
    mask, tag = colab_kit.infer("flood", "cd", pre, post)
    assert tag == "SYNTHETIC-DEMO"
    assert mask.dtype == bool and mask.shape == (64, 64)
    seg_mask, tag2 = colab_kit.infer("flood", "seg", None, post)
    assert tag2 == "SYNTHETIC-DEMO" and seg_mask.shape == (64, 64)


def test_overlay_and_heatmap_ranges():
    pre, post, mask = colab_kit.make_synthetic("flood", 48, 48, seed=5)
    ov = colab_kit.overlay(post, mask)
    hm = colab_kit.heatmap(mask)
    assert ov.shape == post.shape and ov.min() >= 0 and ov.max() <= 1
    assert hm.shape == mask.shape + (3,)
    u8 = colab_kit.to_u8(post)
    assert u8.dtype == np.uint8 and u8.min() >= 0 and u8.max() <= 255


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def test_read_any_png(tmp_path):
    import imageio.v2 as imageio
    p = tmp_path / "x.png"
    imageio.imwrite(p, (np.random.rand(32, 32, 3) * 255).astype(np.uint8))
    arr = colab_kit.read_any(str(p))
    assert arr.shape == (32, 32, 3) and arr.min() >= 0 and arr.max() <= 1


def test_read_any_grayscale_expands(tmp_path):
    import imageio.v2 as imageio
    p = tmp_path / "g.png"
    imageio.imwrite(p, (np.random.rand(16, 16) * 255).astype(np.uint8))
    assert colab_kit.read_any(str(p)).shape == (16, 16, 3)


# ---------------------------------------------------------------------------
# Drive mount (non-Colab branches)
# ---------------------------------------------------------------------------
def test_mount_and_prepare_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root, in_colab = colab_kit.mount_and_prepare(mount=False)
    assert in_colab is False
    assert root == "outputs"


def test_mount_and_prepare_on_without_colab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root, in_colab = colab_kit.mount_and_prepare(mount=True)
    assert in_colab is False
    assert root == "outputs" and (tmp_path / "outputs").is_dir()


# ---------------------------------------------------------------------------
# Export ports (variant A demo, variant B hub)
# ---------------------------------------------------------------------------
def _demo_ns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "dashboard_x.png").write_bytes(b"png")
    return {"NOTEBOOK": "flood_colab.ipynb",
            "STATE": {"domain": "flood", "task": "cd", "mode_tag": "SYNTHETIC-DEMO",
                      "source": "synthetic"},
            "OUT_DIR": "outputs", "AUTO_RUN": False}


def test_export_results_zip_a(tmp_path, monkeypatch):
    ns = _demo_ns(tmp_path, monkeypatch)
    zp = colab_kit.export_results_zip_a(ns, use_drive=False)
    zp = zp if os.path.isabs(zp) else str(tmp_path / zp)
    assert os.path.isfile(zp)
    assert sorted(zipfile.ZipFile(zp).namelist()) == ["dashboard_x.png", "manifest.json"]
    man = json.loads((tmp_path / "outputs" / "manifest.json").read_text())
    assert man["notebook"] == "flood_colab.ipynb" and man["domain"] == "flood"
    assert man["task"] == "cd" and man["mode"] == "SYNTHETIC-DEMO"
    assert "dashboard_x.png" in man["files"]
    # a second export must never swallow the first zip
    zp2 = colab_kit.export_results_zip_a(ns, use_drive=False)
    zp2 = zp2 if os.path.isabs(zp2) else str(tmp_path / zp2)
    assert sorted(zipfile.ZipFile(zp2).namelist()) == ["dashboard_x.png", "manifest.json"]


def test_export_results_zip_hub(tmp_path, monkeypatch):
    ns = _demo_ns(tmp_path, monkeypatch)
    ns.update({"MOUNT_DRIVE": False, "IN_COLAB": False, "DRIVE_ROOT": "outputs",
               "NOTEBOOK": "checkpoint_zoo_colab.ipynb", "STATE": {"k": "v"}})
    zp = colab_kit.export_results_zip_hub(ns)
    zp = zp if os.path.isabs(zp) else str(tmp_path / zp)
    names = zipfile.ZipFile(zp).namelist()
    assert "dashboard_x.png" in names
    assert not any(n.endswith(".zip") for n in names)  # never zips zips
    man = json.loads((tmp_path / "outputs" / "manifest.json").read_text())
    assert man["notebook"] == "checkpoint_zoo_colab.ipynb"
    assert man["drive_root"] == "outputs" and man["state"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Upload-helper ports
# ---------------------------------------------------------------------------
def _upload_ns(tmp_path, monkeypatch):
    ns = _demo_ns(tmp_path, monkeypatch)
    ns.update({"TASK": "both", "SHOW_PREVIEW_ON_UPLOAD": False,
               "STATE": {"domain": "flood", "task": "seg", "pre": None,
                         "post": None, "gt": None, "mask": None, "source": "none"},
               "AUTO_RUN": False})

    class _Box:
        value = 10.0

    class _Out:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ns["gsd_box"], ns["out_up"] = _Box(), _Out()
    return ns


def test_nb_save_upload(tmp_path, monkeypatch):
    ns = _upload_ns(tmp_path, monkeypatch)
    p = colab_kit.nb_save_upload(ns, "img.png", b"data")
    assert Path(p).read_bytes() == b"data"
    assert Path(p).parent.resolve() == (tmp_path / "outputs" / "uploads").resolve()


def test_nb_apply_updates_state(tmp_path, monkeypatch):
    ns = _upload_ns(tmp_path, monkeypatch)
    arr = np.zeros((4, 4, 3), np.float32)
    colab_kit.nb_apply(ns, "pre", arr)
    assert ns["STATE"]["pre"] is arr
    assert ns["STATE"]["source"] == "user" and ns["STATE"]["gsd"] == 10.0
    colab_kit.nb_apply(ns, "post", arr + 1)
    assert ns["STATE"]["post"] is not None


def test_nb_use_synthetic(tmp_path, monkeypatch):
    ns = _upload_ns(tmp_path, monkeypatch)
    colab_kit.nb_use_synthetic(ns)
    assert ns["STATE"]["source"] == "synthetic"
    assert ns["STATE"]["pre"] is not None and ns["STATE"]["gt"] is not None


# ---------------------------------------------------------------------------
# Standalone-fetch contract
# ---------------------------------------------------------------------------
def test_kit_rev_pin_consistent_across_notebooks():
    """Every notebook bootstrap must pin exactly colab_kit.KIT_REV."""
    for nb in sorted(REPO.glob("*_colab.ipynb")):
        text = nb.read_text().replace('\\"', '"')  # unescape JSON quotes
        if "_ensure_colab_kit" not in text:
            continue
        assert f'for _rev in ("{colab_kit.KIT_REV}", "main"):' in text, \
               f"{nb.name}: bootstrap rev does not match colab_kit.KIT_REV"


def test_kit_importable_without_torch():
    """The kit must stay torch-free (notebooks bootstrap before torch exists)."""
    src = (REPO / "tools" / "colab_kit.py").read_text()
    assert "import torch" not in src
