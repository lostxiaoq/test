"""Ink-wash (水墨) rendering primitives.

Everything is float32 in [0, 1]. Painted elements are described by an ink
*density* map plus an *arrival-time* map (T-map): the moment the brush or the
spreading ink reaches each pixel. A frame at time t only has to threshold the
T-map, so expensive shapes are generated once and animated cheaply.
"""
import numpy as np
import cv2

W, H = 1920, 1080

PAPER = np.array([0.938, 0.908, 0.842], np.float32)      # 宣纸
INK = np.array([0.075, 0.074, 0.082], np.float32)        # 松烟墨
INK_SKY = np.array([0.085, 0.110, 0.175], np.float32)    # 墨 + 花青
VERMILION = np.array([0.74, 0.16, 0.10], np.float32)     # 朱砂
GOLD = np.array([1.00, 0.80, 0.42], np.float32)          # 月光金


# ----------------------------------------------------------------- math utils
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sstep(e0, e1, x):
    """Scalar smoothstep."""
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def blur(img, sigma):
    if sigma <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------- noise
def value_noise(h, w, cells_y, cells_x, seed):
    """Bicubic-interpolated gaussian lattice noise covering an h x w image."""
    g = np.random.default_rng(seed).standard_normal((cells_y + 4, cells_x + 4)).astype(np.float32)
    bw = int(round(w * (cells_x + 4) / cells_x))
    bh = int(round(h * (cells_y + 4) / cells_y))
    big = cv2.resize(g, (bw, bh), interpolation=cv2.INTER_CUBIC)
    oy, ox = (bh - h) // 2, (bw - w) // 2
    return big[oy:oy + h, ox:ox + w]


def fbm(h, w, scale=200.0, octaves=5, persistence=0.5, seed=0, aspect=1.0):
    """Fractal noise, roughly unit std. `aspect` > 1 stretches features horizontally."""
    out = np.zeros((h, w), np.float32)
    amp, s = 1.0, float(scale)
    for o in range(octaves):
        cy = max(1, int(np.ceil(h / s)))
        cx = max(1, int(np.ceil(w / (s * aspect))))
        out += amp * value_noise(h, w, cy, cx, seed * 7919 + o * 104729)
        amp *= persistence
        s /= 2.0
        if s < 1.2:
            break
    out -= out.mean()
    return out / (out.std() + 1e-6)


def noise1d(n, scale, octaves=5, persistence=0.5, seed=0):
    """1-D fractal noise of length n, unit std."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float32)
    amp, s = 1.0, float(scale)
    x = np.arange(n, dtype=np.float32)
    for _ in range(octaves):
        k = int(np.ceil(n / s)) + 4
        g = rng.standard_normal(k).astype(np.float32)
        # cubic-ish interpolation through smooth resampling
        xs = np.arange(k, dtype=np.float32) * s - 2 * s
        out += amp * np.interp(x, xs, g).astype(np.float32)
        amp *= persistence
        s /= 2.0
        if s < 1.0:
            break
    out = cv2.GaussianBlur(out.reshape(1, -1), (0, 0), 1.2).ravel()
    out -= out.mean()
    return out / (out.std() + 1e-6)


# --------------------------------------------------------------------- paper
def make_paper_texture(h=H, w=W, seed=11):
    """Luminance multiplier (~1.0) with rice-paper fibres, mottling and grain."""
    rng = np.random.default_rng(seed)
    lum = np.ones((h, w), np.float32)
    lum += fbm(h, w, 520, 4, seed=seed) * 0.016
    lum += fbm(h, w, 90, 4, seed=seed + 1) * 0.007
    grain = rng.standard_normal((h, w)).astype(np.float32)
    lum += blur(grain, 0.55) * 0.010

    # fibres: long pale wisps and short dark ones, drawn as smooth random walks
    fib = np.zeros((h, w), np.uint8)
    dark = np.zeros((h, w), np.uint8)
    n = int(h * w / 520)
    for i in range(n):
        x, y = rng.uniform(-20, w + 20), rng.uniform(-20, h + 20)
        ang = rng.uniform(0, np.pi * 2)
        L = rng.uniform(12, 90)
        steps = max(3, int(L / 4))
        pts = []
        for _ in range(steps):
            pts.append((x, y))
            ang += rng.normal(0, 0.35)
            x += np.cos(ang) * 4
            y += np.sin(ang) * 4
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        if rng.random() < 0.82:
            cv2.polylines(fib, [pts], False, int(rng.uniform(60, 200)), 1, cv2.LINE_AA)
        else:
            cv2.polylines(dark, [pts], False, int(rng.uniform(40, 140)), 1, cv2.LINE_AA)
    lum += blur(fib.astype(np.float32) / 255.0, 0.6) * 0.030
    lum -= blur(dark.astype(np.float32) / 255.0, 0.6) * 0.018
    # sparse specks
    sp = (rng.random((h, w)) > 0.99965).astype(np.float32)
    lum -= blur(sp, 0.8) * 0.35
    return lum


def make_vignette(h=H, w=W):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    r = np.sqrt(nx * nx * 0.85 + ny * ny)
    v = 1.0 - 0.30 * smoothstep(0.55, 1.45, r) ** 1.3
    # warm, slightly darker corners like aged paper
    vig = np.stack([v, v * 0.995 - 0.004, v * 0.985 - 0.010], -1)
    return np.clip(vig, 0, 1).astype(np.float32)


# ----------------------------------------------------------- compositing ops
def over(dst, rgb_premul, a):
    """In-place 'over' of a premultiplied layer onto dst (H,W,3)."""
    dst *= (1.0 - a)[..., None]
    dst += rgb_premul
    return dst


def over_ink(dst, a, ink=INK):
    """Paint an ink density map `a` with colour `ink`."""
    a3 = a[..., None]
    dst *= (1.0 - a3)
    dst += a3 * ink
    return dst


def screen(dst, light):
    """Screen blend light (H,W,3, >=0) onto dst, in place."""
    dst[...] = 1.0 - (1.0 - dst) * (1.0 - np.clip(light, 0.0, 1.0))
    return dst


def over_roi(dst, x0, y0, rgb_premul, a):
    """Premultiplied 'over' of a small sprite at integer position (x0, y0), clipped."""
    h, w = a.shape
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    sx, sy = X0 - x0, Y0 - y0
    aa = a[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
    region = dst[Y0:Y1, X0:X1]
    region *= (1.0 - aa)[..., None]
    region += rgb_premul[sy:sy + Y1 - Y0, sx:sx + X1 - X0]


def add_roi(dst, x0, y0, light):
    """Screen-blend a small light sprite at integer position (x0, y0)."""
    h, w = light.shape[:2]
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, dst.shape[1]), min(y0 + h, dst.shape[0])
    if X1 <= X0 or Y1 <= Y0:
        return
    sx, sy = X0 - x0, Y0 - y0
    l = light[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
    region = dst[Y0:Y1, X0:X1]
    region[...] = 1.0 - (1.0 - region) * (1.0 - np.clip(l, 0, 1))


# ------------------------------------------------------------ brush strokes
def render_strokes(h, w, strokes, ss=1.0):
    """Rasterise brush strokes with dry-brush (飞白) bristle texture.

    Each stroke: dict(pts=(N,2), widths=(N,), t0, t1, dens, dry, seed,
    dens_profile=(N,) optional).  Returns (density, arrival_time) maps.
    Arrival time interpolates t0 -> t1 along the stroke's arc length, so a
    stroke can later be revealed as if a brush were drawing it.  Bristles
    leave streaks of finite length; gaps appear toward the stroke's end and
    along its edges, where a real brush runs dry first.
    """
    dens = np.zeros((h, w), np.float32)
    tmap = np.full((h, w), np.inf, np.float32)
    for st in strokes:
        pts = np.asarray(st["pts"], np.float32) * ss
        wid = np.asarray(st["widths"], np.float32) * ss
        if len(pts) < 2:
            continue
        prof = np.asarray(st.get("dens_profile", np.ones(len(pts))), np.float32)
        rng = np.random.default_rng(st.get("seed", 0))
        nb = 40
        b_phase = rng.uniform(0, 2 * np.pi, nb).astype(np.float32)
        b_period = rng.uniform(14.0, 46.0, nb).astype(np.float32) * ss
        b_bias = rng.uniform(-0.35, 0.35, nb).astype(np.float32)
        dry = st.get("dry", 0.3)
        seg = np.diff(pts, axis=0)
        seglen = np.sqrt((seg ** 2).sum(1)) + 1e-6
        cum = np.concatenate([[0], np.cumsum(seglen)])
        total = cum[-1]
        t0, t1 = st.get("t0", 0.0), st.get("t1", 1.0)
        base_d = st.get("dens", 0.9)
        fade_end = st.get("fade_end", 0.22)
        for i in range(len(seg)):
            p0, p1 = pts[i], pts[i + 1]
            w0, w1 = wid[i], wid[i + 1]
            r = max(w0, w1) + 2
            xa, xb = int(np.floor(min(p0[0], p1[0]) - r)), int(np.ceil(max(p0[0], p1[0]) + r))
            ya, yb = int(np.floor(min(p0[1], p1[1]) - r)), int(np.ceil(max(p0[1], p1[1]) + r))
            xa, ya = max(xa, 0), max(ya, 0)
            xb, yb = min(xb, w - 1), min(yb, h - 1)
            if xb < xa or yb < ya:
                continue
            yy, xx = np.mgrid[ya:yb + 1, xa:xb + 1].astype(np.float32)
            d = seg[i] / seglen[i]
            rx, ry = xx - p0[0], yy - p0[1]
            tt = np.clip((rx * d[0] + ry * d[1]) / seglen[i], 0, 1)
            nx = rx - tt * seg[i][0]
            ny = ry - tt * seg[i][1]
            nd = nx * (-d[1]) + ny * d[0]            # signed normal distance
            dist = np.sqrt(nx * nx + ny * ny)
            ww = w0 + (w1 - w0) * tt
            inside = dist < ww
            if not inside.any():
                continue
            u = np.clip(nd / (ww + 1e-3) * 0.5 + 0.5, 0, 0.9999)   # across-stroke 0..1
            edge = smoothstep(ww, ww - 1.1 * ss, dist)
            s_px = cum[i] + tt * seglen[i]
            s_along = s_px / total
            bi = (u * nb).astype(np.int32)
            bv = 0.5 + 0.5 * np.sin(s_px / b_period[bi] * 2 * np.pi + b_phase[bi]) + b_bias[bi]
            rim = np.abs(u - 0.5) * 2.0
            dryness = dry * (0.25 + 0.75 * s_along ** 1.4) * (0.55 + 0.9 * rim ** 2)
            tex = 0.12 + 0.88 * smoothstep(dryness - 0.10, dryness + 0.10, bv)
            pr = prof[i] + (prof[i + 1] - prof[i]) * tt
            tone = 1.0 - fade_end * s_along
            val = base_d * pr * edge * tex * tone
            sub = dens[ya:yb + 1, xa:xb + 1]
            np.maximum(sub, val, out=sub)
            tsub = tmap[ya:yb + 1, xa:xb + 1]
            tv = t0 + (t1 - t0) * s_along
            upd = inside & (tv < tsub)
            tsub[upd] = tv[upd]
    return dens, tmap


def reveal(dens, tmap, t, ramp=0.08):
    """Density visible at time t given an arrival-time map."""
    return dens * smoothstep(0.0, ramp, t - tmap)
