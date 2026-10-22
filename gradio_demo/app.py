"""
AnyDisasterMapping — Gradio web demo.
Same honest contract as the Colab notebooks: every output is labeled
SYNTHETIC-DEMO / QUICK-TRAIN / CHECKPOINT — this demo runs the labeled
heuristic engine (no heavy weights required) and synthetic AI-style scenes.
Run:  pip install -r requirements.txt && python app.py
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
# Noise / terrain primitives
# --------------------------------------------------------------------------- #
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


DOMAINS = ["flood", "wildfire", "tsunami", "hurricane", "landslide", "volcano",
           "oil_spill", "avalanche", "glof", "dust_storm", "smoke_plume", "locust",
           "infra_damage"]


def _u8(a):
    return (np.clip(a, 0, 1) * 255).astype("uint8")


def _stats(mask, gsd):
    px = int(mask.sum())
    return (f"**changed:** {100.0 * px / mask.size:.2f} %  |  "
            f"**area:** {px * gsd * gsd / 1e6:.2f} km² (GSD {gsd} m/px)  |  "
            "mode: **SYNTHETIC-DEMO** (labeled heuristic)")


def seg_fn(img, domain, gsd):
    if img is None:
        raise gr.Error("Upload an image or use the Synthetic demo tab")
    arr = np.asarray(img).astype("float32")
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = arr[..., :3]
    mask, tag = infer(domain, "seg", arr, None)
    ov = overlay(arr, mask)
    return [arr, ov], _stats(mask, gsd)


def cd_fn(pre, post, domain, gsd):
    if pre is None or post is None:
        raise gr.Error("Upload BOTH pre and post images (or use the Synthetic demo tab)")
    a = np.asarray(pre).astype("float32")
    b = np.asarray(post).astype("float32")
    if a.max() > 1.5:
        a /= 255.0
    if b.max() > 1.5:
        b /= 255.0
    if a.shape[:2] != b.shape[:2]:
        b = np.asarray(Image.fromarray(_u8(b)).resize((a.shape[1], a.shape[0]), _RESAMPLE),
                       "float32") / 255.0
    mask, tag = infer(domain, "cd", a[..., :3], b[..., :3])
    ov = overlay(b[..., :3], mask)
    return [a[..., :3], b[..., :3], heatmap(mask), ov], _stats(mask, gsd)


def synth_fn(domain, seed, gsd):
    pre, post, gt = make_synthetic(domain, 512, 384, seed=int(seed))
    mask, tag = infer(domain, "cd", pre, post)
    ov = overlay(post, mask)
    iou = float((mask & gt).sum()) / max(1, float((mask | gt).sum()))
    return [pre, post, heatmap(mask), ov], (
        _stats(mask, gsd) + f"  |  IoU vs synthetic ground truth: {iou:.3f}")


def _watermark(img_u8, text):
    im = Image.fromarray(img_u8)
    d = ImageDraw.Draw(im)
    d.rectangle([0, im.height - 24, im.width, im.height], fill=(12, 12, 14))
    d.text((9, im.height - 19), text, fill=(255, 255, 255))
    return np.asarray(im)


def ai_fn(domain, seed, poster):
    outs = []
    for k in range(4):
        s = int(seed) + k * 17
        _, scene, _ = make_synthetic(domain, 512, 384, seed=s)
        art = stylize_poster(scene, k=5 + k) if poster else scene
        outs.append(_watermark(_u8(art),
                    "AI-GENERATED ILLUSTRATION | " + domain + " | seed " + str(s) +
                    " | NOT A REAL PHOTO"))
    return outs


with gr.Blocks(title="AnyDisasterMapping demo") as demo:
    gr.Markdown(
        "# 🌍 AnyDisasterMapping — Gradio demo\n"
        "**Honest-demo policy:** results are labeled with the mode that produced them. "
        "This space runs the **SYNTHETIC-DEMO** heuristic engine + AI-style synthetic "
        "scenes (no real photos). For model inference, use the Colab notebooks "
        "(`change_detection_colab.ipynb`, `satellite_finetune_colab.ipynb`).")
    with gr.Tab("🛟 Synthetic demo (no upload needed)"):
        d0 = gr.Dropdown(DOMAINS, value="flood", label="Disaster domain")
        s0 = gr.Slider(0, 9999, value=7, step=1, label="Seed")
        g0 = gr.Number(10, label="GSD (m/px)")
        btn0 = gr.Button("Generate pre/post + change map", variant="primary")
        gal0 = gr.Gallery(columns=4, height=220)
        st0 = gr.Markdown()
        btn0.click(synth_fn, [d0, s0, g0], [gal0, st0])
    with gr.Tab("🆚 Change detection (upload pre/post)"):
        d1 = gr.Dropdown(DOMAINS, value="flood", label="Disaster domain")
        pre1 = gr.Image(label="PRE image", type="pil")
        post1 = gr.Image(label="POST image", type="pil")
        g1 = gr.Number(10, label="GSD (m/px)")
        btn1 = gr.Button("Run change detection", variant="primary")
        gal1 = gr.Gallery(columns=4, height=220)
        st1 = gr.Markdown()
        btn1.click(cd_fn, [pre1, post1, d1, g1], [gal1, st1])
    with gr.Tab("🖼️ Segmentation (single image)"):
        d2 = gr.Dropdown(DOMAINS, value="flood", label="Disaster domain")
        img2 = gr.Image(label="Scene", type="pil")
        g2 = gr.Number(10, label="GSD (m/px)")
        btn2 = gr.Button("Segment", variant="primary")
        gal2 = gr.Gallery(columns=2, height=300)
        st2 = gr.Markdown()
        btn2.click(seg_fn, [img2, d2, g2], [gal2, st2])
    with gr.Tab("🎨 AI illustrations (no real photos)"):
        d3 = gr.Dropdown(DOMAINS, value="flood", label="Disaster domain")
        s3 = gr.Slider(0, 9999, value=42, step=1, label="Seed")
        p3 = gr.Checkbox(True, label="Poster style")
        btn3 = gr.Button("Generate 4 illustrations", variant="primary")
        gal3 = gr.Gallery(columns=2, height=300)
        btn3.click(ai_fn, [d3, s3, p3], [gal3])

if __name__ == "__main__":
    import os

    # GRADIO_SERVER_NAME / GRADIO_SERVER_PORT are honoured by Gradio itself
    # (the Docker `demo` target sets 0.0.0.0 so the published port is reachable).
    # Default stays loopback-only for safety on a developer machine.
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
