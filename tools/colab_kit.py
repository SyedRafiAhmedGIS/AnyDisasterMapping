#!/usr/bin/env python3
"""Shared core for the AnyDisasterMapping Colab notebook collection.

Single source of truth for the code that used to be duplicated inside every
``*_colab.ipynb`` (setup engine, synthetic-scene generator, demo heuristics,
visualisation, upload / run / export helpers).

Design notes
------------
* **Torch-free** — everything here runs on numpy + Pillow (+ optional
  matplotlib / tifffile / h5py / imageio), so notebooks stay lightweight and
  the module is unit-tested in CI (tests/test_colab_kit.py).
* **Notebook-namespace ports** — interactive helpers take an ``ns`` dict
  (pass ``globals()`` from the notebook cell) so per-notebook state
  (``STATE``, ``OUT_DIR``, widgets, ...) keeps working exactly as before,
  while the *logic* lives in this one tested place.
* **Standalone fetch** — thin notebook cells locate this file next to the
  repository; when a notebook runs standalone in Colab,
  :func:`ensure_colab_kit` downloads the file pinned to ``KIT_REV``.
* Engine functions below are byte-identical ports of the originals (kept in
  sync with ``scripts/scene_gen.py``); two latent notebook bugs are fixed in
  the interactive ports: a missing ``clear_output`` import and a hardcoded
  ``/content`` export path outside Colab.
"""
from __future__ import annotations

import datetime
import json
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

__all__ = [
    "KIT_REV", "pip_install", "ensure_colab_kit", "mount_and_prepare",
    "make_synthetic", "infer", "overlay", "heatmap", "to_u8", "stats_for",
    "stylize_poster", "read_any", "verdict", "show_pair",
    "run_inference", "run_dashboard",
    "export_results_zip_a", "export_results_zip_hub",
    "nb_save_upload", "nb_apply", "nb_upload_pre", "nb_upload_post",
    "nb_use_synthetic",
]

# Public constants -----------------------------------------------------------
KIT_REV = "v5.0.0"          # git tag this release must carry (see
                            # GITHUB_UPLOAD_INSTRUCTIONS.md); notebooks pin
                            # their standalone fetch of this file to it.
_REPO = "SyedRafiAhmedGIS/AnyDisasterMapping"


# ---------------------------------------------------------------------------
# Environment bootstrap
# --------------------------------------------------------------------------- #
def pip_install(*pkgs: str) -> None:
    """Quiet, idempotent, restart-safe ``pip install`` (same as notebooks used)."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


def ensure_colab_kit() -> tuple[str, str]:
    """Make ``colab_kit`` importable; returns ("local"|"fetched", tools_dir).

    Search order: current dir and upwards for ``tools/colab_kit.py`` (repo
    checkout), then a pinned download from GitHub raw (``KIT_REV`` tag, with a
    clearly-warned fallback to ``main`` for bootstrap-before-first-release).
    """
    d = os.path.abspath(os.getcwd())
    for _ in range(8):
        p = os.path.join(d, "tools")
        if os.path.isfile(os.path.join(p, "colab_kit.py")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return "local", p
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    dest = os.path.join(tempfile.gettempdir(), "adm_colab_kit")
    os.makedirs(dest, exist_ok=True)
    fp = os.path.join(dest, "colab_kit.py")
    if not os.path.isfile(fp):
        import urllib.request
        for rev in (KIT_REV, "main"):
            url = f"https://raw.githubusercontent.com/{_REPO}/{rev}/tools/colab_kit.py"
            try:
                urllib.request.urlretrieve(url, fp)
                if rev != KIT_REV:
                    print(f"⚠️ colab_kit @ {KIT_REV} not published yet — fetched 'main'; "
                          "pin will activate once the tag is pushed.")
                else:
                    print(f"📥 fetched tools/colab_kit.py @ {rev}")
                break
            except Exception:
                continue
        else:
            raise RuntimeError(
                "colab_kit.py not found locally and could not be fetched from GitHub. "
                f"Run this notebook from the repository checkout, or check {_REPO}.")
    if dest not in sys.path:
        sys.path.insert(0, dest)
    return "fetched", dest


# ---------------------------------------------------------------------------
# Google Drive mount — port of the identical cell used by all 17 notebooks
# --------------------------------------------------------------------------- #
def mount_and_prepare(project: str = "AnyDisasterMapping",
                      mount: bool = True) -> tuple[str, bool]:
    """Mount Google Drive in Colab; returns ``(DRIVE_ROOT, IN_COLAB)``.

    Same three branches and prints as the original notebook cell.
    """
    drive_root = os.path.join("/content/drive/MyDrive", project)
    in_colab = False
    try:
        import google.colab  # noqa: F401
        in_colab = True
    except Exception:
        in_colab = False

    if mount and in_colab:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            print("🔐 Google Drive is about to ask for authorization —")
            print("   click the link, allow access, paste the code back here.")
            drive.mount("/content/drive")
        os.makedirs(drive_root, exist_ok=True)
        print("☁️ Drive mounted →", drive_root)
        print("   anything saved there survives this Colab runtime being recycled.")
        return drive_root, in_colab
    if mount and not in_colab:
        local = "outputs"
        os.makedirs(local, exist_ok=True)
        print("☁️ Not running in Colab — Drive skipped, using local folder:", local)
        return local, in_colab
    local = "/content" if in_colab else "outputs"
    print("☁️ Drive mount OFF — using local folder:", local)
    return local, in_colab



# --------------------------------------------------------------------------- #
# Synthetic scene engine — verbatim from the notebooks (single source now)
# (procedural, AI-style illustration data; no real photos anywhere)
# --------------------------------------------------------------------------- #
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC")

def _field(w, h, cells, seed):
    """Smooth value-noise field in [0,1] (bicubic-upscaled random grid, 1 octave)."""
    rng = np.random.default_rng(seed)
    gh, gw = max(2, h // cells), max(2, w // cells)
    g = rng.random((gh, gw)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8)).resize((w, h), _RESAMPLE)
    a = np.asarray(im, np.float32) / 255.0
    return (a - a.min()) / (a.max() - a.min() + 1e-6)

def _fbm(w, h, seed):
    """Two-octave fractal field."""
    a = _field(w, h, 6, seed) * 0.65 + _field(w, h, 22, seed + 7) * 0.35
    return (a - a.min()) / (a.max() - a.min() + 1e-6)

def _stretch(a, fx, fy=1.0):
    """Horizontally/vertically stretch a field; output is always >= original size."""
    hh, ww = a.shape
    tw, th = max(ww, int(ww * fx)), max(hh, int(hh * fy))
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((tw, th), _RESAMPLE)
    out = np.asarray(im, np.float32) / 255.0
    return out[:hh, :ww]

def _blur(a, r=2):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(r)), np.float32) / 255.0

def _median_mask(m, k=5):
    im = Image.fromarray((m * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MedianFilter(k)), np.float32) > 127

def _terrain(w, h, seed, arid=0.0, cropland=False, snowline=1.2, sea=False):
    """Base 'pre-disaster' scene + elevation f + vegetation g."""
    f = _fbm(w, h, seed)
    SEA = 0.58 if sea else 0.24
    if sea:
        f = f * 0.55 + 0.42                        # push land high; sea cut below
    g = _field(w, h, 26, seed + 31)
    veg = np.clip(0.30 + 0.55 * g - 0.18 * f, 0, 1) * (1 - arid)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.20 + 0.34 * f - 0.10 * veg + 0.16 * arid
    img[..., 1] = 0.24 + 0.42 * veg + 0.10 * f * (1 - arid)
    img[..., 2] = 0.13 + 0.16 * f + 0.05 * veg
    if cropland:
        q = (f * 5).astype(np.int32)
        fq = q / 5.0
        band = (np.sin(fq * 12.9898) * 43758.5453) % 1.0
        tint = (band > 0.5)[..., None]
        img = img * 0.82 + tint * np.array([0.30, 0.38, 0.12]) * 0.18
    water = _blur(f, 3) < SEA
    depth = np.clip(SEA - f, 0, 1)
    wc = np.array([0.07, 0.20, 0.44], np.float32)
    img[water] = wc * (0.55 + 0.45 * (1 - depth[water, None]))
    shore = (~water) & (f < SEA + 0.02)
    img[shore] *= np.array([1.18, 1.08, 0.82], np.float32)
    if snowline < 1.0:
        snow = f > snowline
        img[snow] = np.array([0.90, 0.92, 0.95], np.float32)
    return np.clip(img, 0, 1), f, g, water

# --------------------------------------------------------------------------- #
# Per-domain synthetic scenes:  make_synthetic(domain) -> (pre, post, mask)
# --------------------------------------------------------------------------- #
def make_synthetic(domain, w=512, h=512, seed=7):
    d = str(domain).lower()
    if d in ("flood", "generic", "quickstart", "any"):
        pre, f, g, wat = _terrain(w, h, seed)
        hz = _blur(_fbm(w, h, seed + 101), 1.5)
        flood = (hz > 0.58) & (~wat)
        post = pre.copy()
        post[flood] = np.array([0.05, 0.17, 0.40], np.float32) * (0.7 + 0.5 * hz[flood, None])
        return pre, np.clip(post, 0, 1), flood

    if d == "wildfire":
        pre, f, g, wat = _terrain(w, h, seed, arid=0.55)
        hz = _blur(_fbm(w, h, seed + 202), 1.5)
        burn = (hz > 0.56) & (~wat)
        edge = _blur(burn.astype(np.float32), 3)
        ring = (edge > 0.25) & (edge < 0.55)
        post = pre.copy()
        post[burn] = np.array([0.13, 0.11, 0.10], np.float32) * (0.6 + 0.8 * hz[burn, None])
        post[ring] = post[ring] * 0.5 + np.array([0.85, 0.38, 0.08], np.float32) * 0.5
        return pre, np.clip(post, 0, 1), burn

    if d == "tsunami":
        pre, f, g, wat = _terrain(w, h, seed, sea=True)
        hz = _fbm(w, h, seed + 303)
        land = ~wat
        inund = (f < 0.66) & (hz > 0.46) & land
        post = pre.copy()
        post[inund] = np.array([0.06, 0.19, 0.38], np.float32) * (0.75 + 0.4 * hz[inund, None])
        debris = (hz > 0.72) & land & ~inund
        post[debris] *= np.array([0.75, 0.65, 0.55])
        return pre, np.clip(post, 0, 1), inund

    if d in ("hurricane", "cyclone"):
        pre, f, g, wat = _terrain(w, h, seed)
        cy, cx = h * 0.45, w * 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.42 * h)
        th = np.arctan2(yy - cy, xx - cx)
        spiral = np.sin(3 * th + 7.0 * r) * np.exp(-1.4 * r)
        dmg = (spiral > 0.28) & (r < 1.0)
        post = pre.copy()
        post[dmg] = post[dmg] * np.array([0.72, 0.60, 0.52]) + 0.10
        spec = _field(w, h, 90, seed + 404) > 0.985
        post[spec & dmg] = 0.92
        return pre, np.clip(post, 0, 1), dmg & (r < 0.95)

    if d == "landslide":
        pre, f, g, wat = _terrain(w, h, seed)
        gy, gx = np.gradient(f)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        hz = _fbm(w, h, seed + 505)
        scar = (slope > np.quantile(slope, 0.80)) & (hz > 0.50) & (f > 0.35)
        post = pre.copy()
        post[scar] = np.array([0.46, 0.33, 0.19], np.float32) * (0.7 + 0.6 * hz[scar, None])
        return pre, np.clip(post, 0, 1), scar

    if d in ("volcano", "volcanic"):
        pre, f, g, wat = _terrain(w, h, seed)
        vy, vx = int(h * 0.58), int(w * 0.66)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((yy - vy) ** 2 + (xx - vx) ** 2) / (0.5 * h)
        hz = _fbm(w, h, seed + 606)
        ash = (dist < 0.42) & (hz > 0.42)
        lava = (np.abs(f - 0.5) < 0.035) & (hz > 0.55) & (dist < 0.75)
        post = pre.copy()
        post[ash] = post[ash] * 0.35 + np.array([0.32, 0.30, 0.30]) * 0.65
        post[lava] = np.array([0.95, 0.34, 0.05], np.float32) * (0.7 + 0.5 * hz[lava, None])
        return pre, np.clip(post, 0, 1), ash | lava

    if d in ("oil_spill", "oil"):
        pre, f, g, wat = _terrain(w, h, seed, sea=True)
        hz = _stretch(_fbm(w, h, seed + 707), 2.6, 1.0)
        slick = wat & (hz > 0.56)
        post = pre.copy()
        post[slick] = pre[slick] * np.array([0.38, 0.42, 0.50])
        lookalike = wat & (hz < 0.30) & (hz > 0.22)
        post[lookalike] *= np.array([0.82, 0.84, 0.88])
        return pre, np.clip(post, 0, 1), slick

    if d == "avalanche":
        pre, f, g, wat = _terrain(w, h, seed, snowline=0.68)
        gy, gx = np.gradient(f)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        hz = _fbm(w, h, seed + 808)
        slide = (slope > np.quantile(slope, 0.72)) & (f > 0.62) & (hz > 0.52)
        post = pre.copy()
        post[slide] = np.array([0.42, 0.34, 0.26], np.float32) * (0.75 + 0.5 * hz[slide, None])
        return pre, np.clip(post, 0, 1), slide

    if d == "glof":
        pre, f, g, wat = _terrain(w, h, seed, snowline=0.80)
        hz = _blur(_fbm(w, h, seed + 909), 1.5)
        lake0 = f < 0.26
        grow = (f < 0.33) & (hz > 0.55)
        post = pre.copy()
        newlake = grow & ~lake0
        post[newlake] = np.array([0.06, 0.20, 0.42], np.float32)
        spill = (hz > 0.72) & (f < 0.45) & ~grow
        post[spill] = post[spill] * np.array([0.55, 0.70, 1.0])
        return pre, np.clip(post, 0, 1), newlake | spill

    if d in ("dust_storm", "dust"):
        pre, f, g, wat = _terrain(w, h, seed, arid=0.9)
        hz = _stretch(_fbm(w, h, seed + 111), 2.2)
        yy = np.mgrid[0:h, 0:w][0] / h
        band = (hz > 0.55) & (yy > 0.15) & (yy < 0.85)
        post = pre.copy()
        post[band] = post[band] * 0.45 + np.array([0.78, 0.64, 0.42]) * 0.55
        return pre, np.clip(post, 0, 1), band

    if d in ("smoke_plume", "smoke"):
        pre, f, g, wat = _terrain(w, h, seed, arid=0.35)
        fy, fx = int(h * 0.72), int(w * 0.30)
        hz = _field(w, h, 5, seed + 121)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        drift = (xx - fx) * 0.9 + (yy - fy)
        plume = (drift > 0) & (hz * 260 > drift * 0.85) & (drift < h * 0.75)
        post = pre.copy()
        a = np.clip(1.0 - drift / (h * 0.9), 0.25, 0.85)
        post[plume] = post[plume] * (1 - a[plume, None]) + np.array([0.82, 0.80, 0.78]) * a[plume, None]
        return pre, np.clip(post, 0, 1), plume

    if d == "locust":
        pre, f, g, wat = _terrain(w, h, seed, cropland=True)
        hz = _blur(_fbm(w, h, seed + 131), 1.5)
        hit = (hz > 0.60)
        post = pre.copy()
        post[hit] = post[hit] * np.array([0.62, 0.55, 0.45]) + 0.05
        swarm = _field(w, h, 120, seed + 132) > 0.995
        post[swarm] = 0.25
        return pre, np.clip(post, 0, 1), hit

    if d in ("infra_damage", "infra", "earthquake"):
        pre, f, g, wat = _terrain(w, h, seed, cropland=False)
        q = (g * 6).astype(np.int32) / 6.0
        urban = ((q % 1 == 0) | (q > 0.8)) & (f > 0.3) & (f < 0.7)
        pre = pre.copy()
        pre[urban] = pre[urban] * 0.6 + np.array([0.52, 0.48, 0.44]) * 0.4
        cy, cx = h * 0.5, w * 0.5
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.4 * h)
        hz = _fbm(w, h, seed + 141)
        dmg = urban & (dist < 1.0) & (hz > 0.55 - 0.25 * (1 - dist))
        post = pre.copy()
        post[dmg] = post[dmg] * np.array([0.70, 0.62, 0.55]) + 0.06
        rub = (hz > 0.80) & (dist < 1.1)
        post[rub] = 0.30
        return pre, np.clip(post, 0, 1), dmg

    # fallback
    return make_synthetic("flood", w, h, seed)

# --------------------------------------------------------------------------- #
# Heuristic "inference" (labeled SYNTHETIC-DEMO — never presented as a model)
# --------------------------------------------------------------------------- #
def _hsv(img):
    mx = img.max(-1); mn = img.min(-1)
    s = (mx - mn) / (mx + 1e-6)
    return mx, s

def segment_heuristic(img, domain):
    """Color-rule segmentation per domain. Demo-grade, clearly labeled."""
    d = str(domain).lower()
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    V, S = _hsv(img)
    if d in ("flood", "tsunami", "glof", "generic", "quickstart", "any"):
        return (B > R + 0.04) & (B > G) & (B > 0.22) & (V < 0.75)
    if d == "wildfire":
        return (V < 0.32) & (S < 0.42)
    if d in ("landslide", "avalanche"):
        return (R > G + 0.06) & (G > B) & (R > 0.38) & (V < 0.66)
    if d in ("volcano", "volcanic"):
        glow = (R > 0.55) & (G < 0.55) & (B < 0.35)
        ash = (S < 0.15) & (V > 0.22) & (V < 0.55)
        return glow | ash
    if d in ("oil_spill", "oil"):
        return (V < 0.20) & (B > G)
    if d in ("dust_storm", "dust", "smoke_plume", "smoke"):
        return (V > 0.55) & (S < 0.32)
    if d in ("hurricane", "cyclone"):
        return (V > 0.62) & (S < 0.25)
    if d in ("infra_damage", "infra", "earthquake"):
        return (R > G) & (G > B) & (S < 0.45) & (V > 0.35)
    if d == "locust":
        return (R > G) & (G > B) & (V > 0.30)
    return (B > R + 0.04) & (B > 0.22)

def change_heuristic(pre, post, k=2.5):
    """|pre-post| difference + blur + Otsu threshold + median cleanup."""
    d = np.abs(post - pre).mean(-1)
    d = _blur(d, 2)
    return _median_mask(d > _otsu(d), 5)

def _otsu(x, nbins=32):
    """Otsu threshold on a float array in [0,1]."""
    hist, edges = np.histogram(np.clip(x, 0, 1), bins=nbins, range=(0, 1))
    hist = hist.astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom <= 0] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    idx = int(np.argmax(sigma_b))
    return float(edges[idx + 1])

def infer(domain, task, pre, post=None):
    """task 'seg' → segment post image; 'cd' → change mask. Returns (mask, tag)."""
    if task == "cd" and post is not None:
        return change_heuristic(pre, post), "SYNTHETIC-DEMO"
    return segment_heuristic(post if post is not None else pre, domain), "SYNTHETIC-DEMO"

# --------------------------------------------------------------------------- #
# Visualization helpers
# --------------------------------------------------------------------------- #
OVERLAY_COLOR = (1.00, 0.23, 0.19)

def overlay(img, mask, color=OVERLAY_COLOR, alpha=0.55):
    m = mask[..., None]
    c = np.array(color, np.float32)
    return np.clip(img * (1 - m * alpha) + c * m * alpha, 0, 1)

def heatmap(mask):
    h = _blur(mask.astype(np.float32), 3)
    return np.stack([h * 1.15 + 0.05, h * 0.25, h * 0.30], -1).clip(0, 1)

def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def stats_for(mask, gsd_m=10.0):
    px = int(mask.sum())
    tot = mask.size
    return {"changed_px": px, "changed_pct": round(100.0 * px / tot, 2),
            "km2": round(px * gsd_m * gsd_m / 1e6, 3)}

def stylize_poster(img, k=6):
    """AI-illustration style: posterize + saturate + grain + vignette (no real photos)."""
    im = Image.fromarray(to_u8(img))
    q = im.quantize(colors=k, method=Image.MEDIANCUT).convert("RGB")
    q = q.filter(ImageFilter.SMOOTH_MORE)
    a = np.asarray(q, np.float32) / 255.0
    mean = a.mean(-1, keepdims=True)
    a = np.clip(mean + (a - mean) * 1.45, 0, 1)          # saturation boost
    hh, ww = a.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    r = np.sqrt(((yy / hh - 0.5) ** 2 + (xx / ww - 0.5) ** 2) * 2)
    a *= (1.0 - 0.45 * np.clip(r - 0.45, 0, 1))[..., None]  # vignette
    rng = np.random.default_rng(3)
    a += (rng.random((hh, ww, 1)).astype(np.float32) - 0.5) * 0.05  # grain
    return np.clip(a, 0, 1)

# --------------------------------------------------------------------------- #
# Readers / preview helpers
# --------------------------------------------------------------------------- #
def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            arr = np.asarray(f[next(iter(f.keys()))], dtype=np.float32)
    else:
        import imageio.v2 as imageio
        arr = np.asarray(imageio.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr.astype(np.float32), 0, 1)

def verdict(path, arr):
    mb = os.path.getsize(path) / 1e6
    h, w = arr.shape[:2]
    ok = 96 <= min(h, w) <= 8192
    print(f"   🔎 {os.path.basename(path)} | {w}x{h} px | {mb:.2f} MB | "
          f"{'✅ suitable' if ok else '⚠️ unusual dimensions — auto-resized for the demo'}")

def show_pair(pre, post=None, title="Uploaded preview"):
    cols = 2 if post is not None else 1
    import matplotlib.pyplot as plt  # lazy: kit stays light
    fig, ax = plt.subplots(1, cols, figsize=(4.2 * cols, 4.4))
    ax = np.atleast_1d(ax)
    ax[0].imshow(pre); ax[0].set_title("PRE" if post is not None else "input"); ax[0].axis("off")
    if post is not None:
        ax[1].imshow(post); ax[1].set_title("POST"); ax[1].axis("off")
    plt.suptitle(title); plt.show()

# ---------------------------------------------------------------------------
# Notebook-namespace ports: RUN pipeline (was an identical cell in 7 notebooks)
# ---------------------------------------------------------------------------
def run_inference(ns: dict, try_checkpoint=None):
    """Port of ``run_inference``; ``ns`` is the notebook ``globals()``."""
    pre, post = ns["STATE"].get("pre"), ns["STATE"].get("post")
    if pre is None or (ns["STATE"]["task"] == "cd" and post is None):
        print("🛟 no usable pair in STATE — falling back to bundled synthetic sample")
        pre, post, gt = make_synthetic(ns["STATE"]["domain"], 512, 512, seed=7)
        ns["STATE"].update(pre=pre, post=post, gt=gt, source="synthetic")
    ck = try_checkpoint() if try_checkpoint is not None else None
    if ck is not None:
        fn, mode_tag = ck
        ns["MODE_TAG"] = mode_tag
        mask = fn(ns["STATE"]["pre"], ns["STATE"]["post"])
    else:
        mask, mode_tag = infer(ns["STATE"]["domain"], ns["STATE"]["task"],
                               ns["STATE"]["pre"], ns["STATE"]["post"])
        ns["MODE_TAG"] = mode_tag
    ns["STATE"]["mask"] = mask
    ns["STATE"]["mode_tag"] = ns["MODE_TAG"]
    return mask


def run_dashboard(ns: dict, try_checkpoint=None):
    """Port of ``run_pipeline`` (inference + dashboard + file outputs)."""
    import matplotlib.pyplot as plt
    mask = run_inference(ns, try_checkpoint)
    gsd = ns["STATE"].get("gsd", ns["GSD_M"])
    st = stats_for(mask, gsd)
    st["mode"] = ns["STATE"].get("mode_tag", ns.get("MODE_TAG", "SYNTHETIC-DEMO"))
    st["domain"] = ns["STATE"]["domain"]
    st["task"] = ns["STATE"]["task"]
    st["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    pre, post, tag = ns["STATE"]["pre"], ns["STATE"]["post"], st["mode"]
    if ns["STATE"]["task"] == "cd":
        panels = [pre, post, heatmap(mask), overlay(post, mask)]
        titles = ["PRE", "POST", "change heatmap", "changed-area overlay"]
    else:
        panels = [post, heatmap(mask), overlay(post, mask)]
        titles = ["input image", "heatmap", "overlay"]
    fig, ax = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.6))
    for a, im, t in zip(np.atleast_1d(ax), panels, titles):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle(f"{ns['NOTEBOOK']} — mode: {tag}", fontsize=12, fontweight="bold")
    dash = os.path.join(ns["OUT_DIR"], "dashboard_" + ns["STATE"]["domain"] + ".png")
    fig.savefig(dash, dpi=110, bbox_inches="tight"); plt.show()
    Image.fromarray(to_u8(post)).save(os.path.join(ns["OUT_DIR"], "post_" + ns["STATE"]["domain"] + ".png"))
    Image.fromarray(to_u8(overlay(post, mask))).save(
        os.path.join(ns["OUT_DIR"], "overlay_" + ns["STATE"]["domain"] + ".png"))
    np.save(os.path.join(ns["OUT_DIR"], "mask_" + ns["STATE"]["domain"] + ".npy"), mask)
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(ns["OUT_DIR"], "mask_" + ns["STATE"]["domain"] + ".png"))
    with open(os.path.join(ns["OUT_DIR"], "stats.json"), "w") as f:
        json.dump(st, f, indent=2)
    print("┌" + "─" * 46 + "┐")
    print(("│ 📊 RESULT — mode: " + tag).ljust(47) + "│")
    print(("│   changed: " + str(st["changed_pct"]) + " % of scene").ljust(47) + "│")
    print(("│   area: " + str(st["km2"]) + " km² (GSD " + str(gsd) + " m/px)").ljust(47) + "│")
    if ns["STATE"].get("gt") is not None:
        gt = ns["STATE"]["gt"]
        iou = float((mask & gt).sum()) / max(1, float((mask | gt).sum()))
        print(("│   IoU vs synthetic ground truth: " + f"{iou:.3f}").ljust(47) + "│")
    print(("│   saved: " + dash).ljust(47) + "│")
    print("└" + "─" * 46 + "┘")
    if ns.get("AUTO_RUN") and "run_export" in ns:
        ns["run_export"]()
    return mask


# ---------------------------------------------------------------------------
# Notebook-namespace ports: EXPORT cells (variant A: 12 demo notebooks;
# variant B: 4 hub notebooks)
# ---------------------------------------------------------------------------
def export_results_zip_a(ns: dict, use_drive: bool = False) -> str:
    """Port of EXPORT variant A.  Fix vs original: outside Colab the zip is
    written next to OUT_DIR instead of the nonexistent /content."""
    zip_root = "/content" if os.path.isdir("/content") else ns["OUT_DIR"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"notebook": ns["NOTEBOOK"], "domain": ns["STATE"]["domain"],
                "task": ns["STATE"]["task"],
                "mode": ns["STATE"].get("mode_tag", "SYNTHETIC-DEMO"),
                "source": ns["STATE"].get("source", "none"),
                "exported_at": ts,
                "files": sorted(os.listdir(ns["OUT_DIR"]))}
    with open(os.path.join(ns["OUT_DIR"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    zpath = os.path.join(zip_root, "results_{}_{}_{}.zip").format(
        ns["STATE"]["domain"], ns["STATE"]["task"], ts)
    import zipfile as _zf
    with _zf.ZipFile(zpath, "w", _zf.ZIP_DEFLATED) as z:
        for root, _, fns in os.walk(ns["OUT_DIR"]):
            for fn in fns:
                if fn.endswith(".zip"):
                    continue  # when zip_root == OUT_DIR, never zip ourselves
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, ns["OUT_DIR"]))
    print("📦 exported:", zpath, f"({os.path.getsize(zpath)/1e6:.2f} MB)")
    if use_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            dst = "/content/drive/MyDrive/AnyDisasterMapping_results"
            os.makedirs(dst, exist_ok=True)
            import shutil; shutil.copy(zpath, dst)
            print("☁️ copied to Google Drive:", dst)
        except Exception as e:
            print("Drive copy skipped:", e)
    try:
        from google.colab import files as _dl
        _dl.download(zpath)
    except Exception:
        print("(auto-download unavailable outside Colab — file is in %s)" % zip_root)
    return zpath


def export_results_zip_hub(ns: dict) -> str:
    """Port of EXPORT variant B (checkpoint_zoo / dataset_download /
    evaluation_metrics / scd notebooks)."""
    import zipfile as _zf
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"notebook": ns["NOTEBOOK"],
                "exported_at": ts,
                "drive_root": ns["DRIVE_ROOT"],
                "state": {k: (v if isinstance(v, (str, int, float, bool, list)) else "…")
                          for k, v in ns["STATE"].items()},
                "files": sorted(os.listdir(ns["OUT_DIR"]))}
    mpath = os.path.join(ns["OUT_DIR"], "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    zpath = os.path.join(ns["OUT_DIR"], f"results_{ns['NOTEBOOK'].replace('.ipynb', '')}_{ts}.zip")
    with _zf.ZipFile(zpath, "w", _zf.ZIP_DEFLATED) as z:
        for root, _, fns in os.walk(ns["OUT_DIR"]):
            for fn in fns:
                if fn.endswith(".zip"):
                    continue
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, ns["OUT_DIR"]))
    print("📦 exported:", zpath, f"({os.path.getsize(zpath)/1e6:.2f} MB)")
    if ns.get("MOUNT_DRIVE") and ns.get("IN_COLAB"):
        try:
            import shutil
            dst = os.path.join(ns["DRIVE_ROOT"], os.path.basename(zpath))
            shutil.copy(zpath, dst)
            print("☁️ copied to Google Drive:", dst)
        except Exception as e:
            print("Drive copy skipped:", e)
    if ns.get("IN_COLAB"):
        try:
            from google.colab import files as _dl
            _dl.download(zpath)
        except Exception as e:
            print("Browser download skipped:", e)
    return zpath


# ---------------------------------------------------------------------------
# Notebook-namespace ports: upload helpers (were identical in 12 notebooks).
# Fixes vs originals: clear_output is imported properly; google.colab is
# optional so the notebooks also run end-to-end outside Colab.
# ---------------------------------------------------------------------------
def _notebook_gfiles():
    try:
        from google.colab import files
        return files
    except Exception:
        return None


def nb_save_upload(ns: dict, name: str, data: bytes) -> str:
    d = os.path.join(ns["OUT_DIR"], "uploads"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def nb_apply(ns: dict, kind: str, arr) -> None:
    if kind == "pre":
        ns["STATE"]["pre"] = arr
    else:
        ns["STATE"]["post"] = arr
    ns["STATE"]["source"] = "user"
    if ns["gsd_box"].value:
        ns["STATE"]["gsd"] = float(ns["gsd_box"].value)
    if ns["SHOW_PREVIEW_ON_UPLOAD"]:
        with ns["out_up"]:
            show_pair(ns["STATE"]["pre"], ns["STATE"]["post"] if ns["TASK"] != "seg" else None,
                      "📤 uploaded preview")
    ready = ns["STATE"]["pre"] is not None and (ns["TASK"] == "seg" or ns["STATE"]["post"] is not None)
    if ns["AUTO_RUN"] and ready and "run_pipeline" in ns:
        with ns["out_up"]:
            print("↻ re-running pipeline on your data …")
            ns["run_pipeline"]()


def nb_upload_pre(ns: dict, _=None) -> None:
    from IPython.display import clear_output
    gfiles = _notebook_gfiles()
    with ns["out_up"]:
        clear_output()
        print("📂 file picker opened — choose your PRE image …")
    try:
        if gfiles is None:
            print("(outside Colab — place files into %s and re-run)" % ns["OUT_DIR"])
            return
        for name, data in gfiles.upload().items():
            p = nb_save_upload(ns, name, data)
            arr = read_any(p)
            with ns["out_up"]:
                verdict(p, arr)
            nb_apply(ns, "pre", arr)
    except Exception as e:
        with ns["out_up"]:
            print("upload failed:", e)


def nb_upload_post(ns: dict, _=None) -> None:
    from IPython.display import clear_output
    gfiles = _notebook_gfiles()
    with ns["out_up"]:
        clear_output()
        print("📂 file picker opened — choose your POST image …")
    try:
        if gfiles is None:
            print("(outside Colab — place files into %s and re-run)" % ns["OUT_DIR"])
            return
        for name, data in gfiles.upload().items():
            p = nb_save_upload(ns, name, data)
            arr = read_any(p)
            with ns["out_up"]:
                verdict(p, arr)
            nb_apply(ns, "post", arr)
    except Exception as e:
        with ns["out_up"]:
            print("upload failed:", e)


def nb_use_synthetic(ns: dict, _=None) -> None:
    from IPython.display import clear_output
    seed = random.randint(0, 9999)
    pre, post, gt = make_synthetic(ns["STATE"]["domain"], 512, 512, seed=seed)
    ns["STATE"].update(pre=pre, post=post, gt=gt, source="synthetic")
    with ns["out_up"]:
        clear_output()
        print(f"🛟 synthetic {ns['STATE']['domain']} scene (seed {seed}) — labeled SYNTHETIC-DEMO")
        if ns["SHOW_PREVIEW_ON_UPLOAD"]:
            show_pair(pre, post, "🛟 synthetic sample (AI-generated illustration, no real photos)")
    if ns["AUTO_RUN"] and "run_pipeline" in ns:
        ns["run_pipeline"]()

