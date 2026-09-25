"""The painted world: parallax layers, camera, and landscape construction."""
import numpy as np
import cv2
from ink import (W, H, PAPER, INK, INK_SKY, GOLD, fbm, noise1d, smoothstep, sstep, blur)
from sky import Sky, Moon
from mountains import paint_range, draw_far_trees

CX, CY = W / 2.0, H / 2.0


class Camera:
    def __init__(self, x=0.0, y=0.0, dz=0.0):
        self.x, self.y, self.dz = x, y, dz

    def matrix(self, depth, origin):
        """Affine matrix mapping layer pixels to screen for a layer at `depth`."""
        s = depth / max(depth - self.dz, 1e-3)
        px, py = self.x / depth, self.y / depth
        ox, oy = origin
        return np.float32([[s, 0, s * (ox - CX - px) + CX],
                           [0, s, s * (oy - CY - py) + CY]]), s

    def to_screen(self, depth, wx, wy):
        s = depth / max(depth - self.dz, 1e-3)
        return (s * (wx - CX - self.x / depth) + CX,
                s * (wy - CY - self.y / depth) + CY, s)


def warp(img, M, flags=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rgba_from(dens, alpha, ink=INK):
    """Premultiplied RGBA for 'paper silhouette + ink'."""
    A = 1.0 - (1.0 - alpha) * (1.0 - dens)
    rgb = PAPER[None, None, :] * (alpha * (1.0 - dens))[..., None] + ink[None, None, :] * dens[..., None]
    return np.concatenate([rgb, A[..., None]], -1).astype(np.float32)


class Layer:
    """A static painted layer with an optional reveal (arrival-time) map."""

    def __init__(self, name, depth, rgba, origin, T=None, ramp=0.6):
        self.name, self.depth, self.rgba, self.origin = name, depth, rgba, origin
        self.ramp = ramp
        self.set_T(T)

    def set_T(self, T):
        self.T = T
        if T is None:
            self.t_first = self.t_done = None
        else:
            fin = T[np.isfinite(T)]
            self.t_first = float(fin.min())
            self.t_done = float(fin.max()) + self.ramp

    def draw(self, frame, t, cam):
        if self.T is not None and t < self.t_first:
            return
        img = self.rgba
        if self.T is not None and t < self.t_done:
            r = smoothstep(0.0, self.ramp, t - self.T)
            img = img * r[..., None]
        M, _ = cam.matrix(self.depth, self.origin)
        out = warp(img, M)
        frame *= (1.0 - out[..., 3:4])
        frame += out[..., :3]


def sweep_T(h, w, t0, dur, dens, seed, dir_x=1.0, down=0.35, jitter=0.12):
    """Arrival map for a brush sweep: left->right along ridges, bleeding downward."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / w if dir_x > 0 else 1.0 - xx / w
    # distance below the top of the painted area in each column
    painted = dens > 0.02
    top = np.where(painted.any(0), painted.argmax(0), h).astype(np.float32)
    below = np.clip((yy - top[None, :]) / 220.0, 0, 3)
    n = fbm(h, w, 120, 4, seed=seed)
    T = t0 + dur * ((1 - down) * nx + down * below / 3.0) + jitter * dur * n
    return T.astype(np.float32)


# --------------------------------------------------------------- landscape
HORIZON = 700


def build_landscape(seed=100):
    layers = []

    # ---- far range (远山): pale, soft, wide
    m = 260
    w, h = W + 2 * m, 460
    oy = HORIZON - h + 20
    peaks = [(-150 + m, 150, 260, 2.0, 0.2), (180 + m, 110, 200, 2.0, -0.2),
             (820 + m, 90, 260, 1.8, 0.1), (1100 + m, 170, 230, 2.0, 0.25),
             (1350 + m, 120, 200, 2.0, -0.1), (1700 + m, 200, 260, 1.9, 0.2),
             (2050 + m, 150, 240, 2.0, 0.0), (500 + m, 70, 230, 1.8, 0.0)]
    d, a, _ = paint_range(h, w, h - 20, peaks, seed + 1, dens=0.30, fade=0.6, edge=2.2,
                          tex=0.12, line=0.05, fold=0.0, mist_h=120, blur_s=1.4, shade=0.18,
                          streak_w=16.0)
    T = sweep_T(h, w, 2.3, 2.6, d, seed + 2)
    layers.append(Layer("far", 7.0, rgba_from(d, a), (-m, oy), T))

    # ---- mid-far range: karst groups at both sides
    m = 300
    w, h = W + 2 * m, 620
    oy = HORIZON + 30 - h + 20
    peaks = [(-60 + m, 300, 150, 2.3, 0.1), (190 + m, 230, 120, 2.4, -0.2),
             (1240 + m, 150, 110, 2.2, 0.0), (1480 + m, 230, 120, 2.4, 0.2),
             (1880 + m, 330, 150, 2.4, -0.1), (2130 + m, 260, 150, 2.2, 0.1)]
    d, a, _ = paint_range(h, w, h - 20, peaks, seed + 3, dens=0.46, fade=0.5, edge=1.6,
                          tex=0.14, line=0.10, fold=0.0, mist_h=160, blur_s=0.7, shade=0.2,
                          streak_w=14.0, strokes=25, stroke_w=(1.2, 2.4), stroke_len=(0.08, 0.2),
                          stroke_d=(0.12, 0.25))
    T = sweep_T(h, w, 3.0, 2.6, d, seed + 4)
    layers.append(Layer("midfar", 4.5, rgba_from(d, a), (-m, oy), T))

    # ---- mid range (中景): tall cliffs framing the left and right
    m = 380
    w, h = W + 2 * m, 900
    base = HORIZON + 90
    oy = base - h + 30
    peaks = [(-140 + m, 560, 170, 2.4, 0.15), (110 + m, 430, 130, 2.5, -0.25),
             (300 + m, 250, 110, 2.2, -0.1),
             (1830 + m, 380, 120, 2.5, 0.2), (2060 + m, 540, 170, 2.4, -0.1),
             (2280 + m, 440, 150, 2.3, 0.0)]
    d, a, ridges = paint_range(h, w, h - 30, peaks, seed + 5, dens=0.74, fade=0.42, edge=1.1,
                               tex=0.16, line=0.28, fold=0.0, moss=60, mist_h=240, blur_s=0.0,
                               shade=0.24, streak_w=11.0, dry=0.25, strokes=70,
                               stroke_w=(1.0, 2.6), stroke_len=(0.08, 0.26), stroke_d=(0.25, 0.5))
    rng = np.random.default_rng(seed + 6)
    for (xc, hg, wd, _, _), ridge in zip(peaks, ridges):
        for _ in range(2):
            px = rng.uniform(xc - 0.7 * wd, xc + 0.7 * wd)
            ix = int(np.clip(px, 0, w - 1))
            if h - 30 - ridge[ix] < 0.5 * hg:
                continue
            draw_far_trees(d, px, ridge[ix] + 4, rng.uniform(9, 15), rng, n=int(rng.integers(2, 5)))
    T = sweep_T(h, w, 3.8, 2.8, d, seed + 7)
    layers.append(Layer("mid", 2.6, rgba_from(d, a), (-m, oy), T))
    return layers


# --------------------------------------------------------------------- water
def build_water(seed=300):
    m = 420
    w, h = W + 2 * m, 520
    oy = HORIZON - 30
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wy = yy + oy                                   # world y
    depth01 = np.clip((wy - HORIZON) / (H - HORIZON), 0, 1.3)
    alpha = smoothstep(HORIZON - 20, HORIZON + 40, wy)
    # perspective ripples: horizontal broken strokes, larger toward the viewer
    rip = np.zeros((h, w), np.float32)
    rng = np.random.default_rng(seed)
    for i in range(900):
        y = HORIZON + (H + 40 - HORIZON) * rng.random() ** 1.6
        py = y - oy
        f = (y - HORIZON) / (H - HORIZON)
        L = rng.uniform(20, 90) * (0.4 + 2.2 * f)
        x = rng.uniform(0, w)
        th = max(1, int(round(0.6 + 2.4 * f * rng.random())))
        val = float(rng.uniform(0.25, 1.0))
        # slightly wavy stroke
        n = max(3, int(L / 8))
        xs = np.linspace(x, x + L, n)
        ys = py + np.sin(np.linspace(0, np.pi * rng.uniform(1, 3), n)) * rng.uniform(0.5, 2.0) * (0.5 + f)
        pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(rip, [pts], False, val, th, cv2.LINE_AA)
    brk = smoothstep(-0.3, 0.6, fbm(h, w, 40, 4, seed=seed + 1, aspect=4.0))
    rip = blur(rip, 0.6) * brk
    dens = rip * (0.10 + 0.16 * depth01)
    # faint wash: water darkens a touch toward the viewer
    dens += 0.05 * smoothstep(0.2, 1.1, depth01) * (1 + 0.3 * fbm(h, w, 200, 3, seed=seed + 2, aspect=5.0))
    dens = np.clip(dens, 0, 1)
    return Layer("water", 3.2, rgba_from(dens, alpha), (-m, oy), None)


# ---------------------------------------------------------------------- mist
class Mist:
    """Drifting mist bands painted as paper white over the landscape."""

    def __init__(self, y, thick, depth, speed, strength, seed, aspect=7.0):
        m = 700
        self.w, self.h = W + 2 * m, int(thick * 3)
        self.depth, self.speed = depth, speed
        self.origin = (-m, y - self.h / 2)
        yy = np.arange(self.h, dtype=np.float32)[:, None]
        band = np.exp(-((yy - self.h / 2) / (thick * 0.55)) ** 2)
        n = fbm(self.h, self.w, 160, 5, seed=seed, aspect=aspect)
        a = smoothstep(-0.6, 1.0, n + 0.25) * band * strength
        # make the texture tile horizontally by crossfading the ends
        self.a = a.astype(np.float32)
        rgb = PAPER[None, None, :] * self.a[..., None]
        self.rgba = np.concatenate([rgb, self.a[..., None]], -1).astype(np.float32)

    def draw(self, frame, t, cam, fade=1.0):
        if fade <= 0.001:
            return
        ox, oy = self.origin
        M, _ = cam.matrix(self.depth, (ox + self.speed * t, oy))
        out = warp(self.rgba, M)
        if fade < 1.0:
            out *= fade
        frame *= (1.0 - out[..., 3:4])
        frame += out[..., :3]
