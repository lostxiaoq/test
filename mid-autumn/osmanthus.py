"""An osmanthus branch (桂枝) painted stroke by stroke, then golden florets bloom."""
import numpy as np
import cv2
from ink import INK, fbm, smoothstep, sstep, blur, render_strokes, over_roi, add_roi, ease_out
from world import Layer, rgba_from

LEAF_INK = np.array([0.085, 0.115, 0.095], np.float32)
GOLD_FL = np.array([0.99, 0.74, 0.20], np.float32)
DEPTH = 1.0
S3, CAMX3, CAMY3 = 1.0 / (1.0 - 0.13), -130.0, -50.0     # camera mid-way through the osmanthus scene


def to_world(sx, sy):
    """Screen position (as framed in scene 3) -> world position on the branch layer."""
    return ((sx - 960) / S3 + 960 + CAMX3, (sy - 540) / S3 + 540 + CAMY3)


def _bez(pts, n):
    pts = np.float32(pts)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    if len(pts) == 3:
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]
    return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1] + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])


def _floret_cluster(n, rng, S=64):
    """RGBA sprite of a cluster of 4-petalled florets."""
    ss = 3
    a = np.zeros((S * ss, S * ss), np.float32)
    core = np.zeros_like(a)
    c = S * ss / 2
    for i in range(n):
        r = rng.uniform(0, S * ss * 0.28) * np.sqrt(rng.random())
        th = rng.uniform(0, 2 * np.pi)
        fx, fy = c + np.cos(th) * r, c + np.sin(th) * r * 0.8
        fs = S * ss * rng.uniform(0.075, 0.10)
        rot = rng.uniform(0, np.pi / 2)
        for k in range(4):
            ang = rot + k * np.pi / 2
            px, py = fx + np.cos(ang) * fs * 0.62, fy + np.sin(ang) * fs * 0.62
            cv2.ellipse(a, (int(px), int(py)), (int(fs * 0.62), int(fs * 0.44)), np.degrees(ang), 0, 360,
                        float(rng.uniform(0.85, 1.0)), -1, cv2.LINE_AA)
        cv2.circle(core, (int(fx), int(fy)), max(1, int(fs * 0.28)), 1.0, -1, cv2.LINE_AA)
    a = cv2.resize(a, (S, S), interpolation=cv2.INTER_AREA)
    core = cv2.resize(core, (S, S), interpolation=cv2.INTER_AREA)
    a = blur(a, 0.5)
    core = blur(core, 0.6)
    col = np.ones((S, S, 3), np.float32) * GOLD_FL
    col = col * (1 - core[..., None] * 0.35) + core[..., None] * np.array([0.85, 0.42, 0.08], np.float32) * 0.35
    rgb = col * a[..., None]
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32)


class Osmanthus:
    def __init__(self, t0=19.4, seed=90):
        rng = np.random.default_rng(seed)
        self.t0 = t0
        m = 300
        w, h = 1400, 900
        self.origin = (-m, -m + 20)
        ox, oy = self.origin

        def L(sx, sy):
            wx, wy = to_world(sx, sy)
            return (wx - ox, wy - oy)

        strokes = []
        leaves = []
        nodes = []
        # main branch, entering from the upper left and drooping past the moon
        main = _bez([L(-12, 26), L(250, 62), L(530, 122), L(790, 300)], 34)
        s = np.linspace(0, 1, len(main))
        strokes.append(dict(pts=main, widths=11.0 * (1 - s) ** 0.8 + 2.0, t0=t0, t1=t0 + 1.2,
                            dens=0.9, dry=0.45, seed=seed + 1))
        subs = [  # (from index on main, control, end, width, t)
            (10, L(230, 150), L(250, 330), 5.5, t0 + 0.7),
            (19, L(440, 190), L(470, 330), 4.6, t0 + 0.9),
            (25, L(640, 110), L(730, 70), 4.0, t0 + 1.0),
            (29, L(690, 260), L(640, 420), 3.8, t0 + 1.1),
            (5, L(90, 150), L(60, 250), 4.2, t0 + 0.55),
        ]
        branches = [main]
        for si, ctrl, end, w0, tb in subs:
            p0 = main[si]
            pts = _bez([p0, ctrl, end], 16)
            ss_ = np.linspace(0, 1, len(pts))
            strokes.append(dict(pts=pts, widths=w0 * (1 - ss_) ** 0.8 + 1.2, t0=tb, t1=tb + 0.6,
                                dens=0.88, dry=0.4, seed=int(rng.integers(1 << 30))))
            branches.append(pts)
        # leaves: pairs along each branch, pointing outward and down
        lt = t0 + 1.0
        for bi, pts in enumerate(branches):
            n_pairs = 5 if bi == 0 else 3
            for j in range(n_pairs):
                f = 0.25 + 0.7 * (j + rng.uniform(0, 0.6)) / n_pairs
                k = min(int(f * (len(pts) - 1)), len(pts) - 2)
                p = pts[k]
                tg = pts[k + 1] - pts[k]
                tg /= np.linalg.norm(tg) + 1e-6
                nodes.append((p.copy(), lt + 0.8 + 0.05 * len(nodes)))
                for side in (-1, 1):
                    if rng.random() < 0.15:
                        continue
                    ang = np.arctan2(tg[1], tg[0]) + side * rng.uniform(0.45, 1.25) + rng.uniform(-0.25, 0.35)
                    Ll = rng.uniform(40, 56)
                    d = np.float32([np.cos(ang), np.sin(ang)])
                    e = p + d * Ll
                    ctrl = p + d * Ll * 0.5 + np.float32([-d[1], d[0]]) * side * rng.uniform(3, 8)
                    lp = _bez([p + d * 4, ctrl, e], 14)
                    prof = np.sin(np.linspace(0.06, np.pi - 0.08, 14)) ** 0.55
                    prof *= np.linspace(1.0, 0.8, 14)
                    ws = rng.uniform(9.0, 12.0) * prof
                    ws[0] = 1.0
                    dens = float(rng.uniform(0.5, 0.82))
                    leaves.append(dict(pts=lp, widths=ws, t0=lt, t1=lt + 0.28, dens=dens, dry=0.18,
                                       seed=int(rng.integers(1 << 30)), fade_end=0.25))
                    # petiole and midrib vein (勾筋)
                    leaves.append(dict(pts=_bez([p, p + d * 3, p + d * 6], 4), widths=np.full(4, 1.2),
                                       t0=lt, t1=lt + 0.05, dens=0.9, dry=0.0, seed=int(rng.integers(1 << 30))))
                    vp = lp[1:-2]
                    leaves.append(dict(pts=vp, widths=np.full(len(vp), 0.8), t0=lt + 0.2, t1=lt + 0.4,
                                       dens=min(dens + 0.2, 0.97), dry=0.15, seed=int(rng.integers(1 << 30))))
                    lt += 0.07
        d1, t1 = render_strokes(h, w, strokes)
        d2, t2 = render_strokes(h, w, leaves)
        # leaves in a green-tinged ink: composite as a separate colour layer
        self.branch = Layer("branch", DEPTH, rgba_from(d1, np.zeros_like(d1)), self.origin, t1, ramp=0.12)
        self.leaves = Layer("leaves", DEPTH, rgba_from(d2, np.zeros_like(d2), LEAF_INK), self.origin, t2, ramp=0.12)
        # flower clusters at the nodes
        self.clusters = []
        for p, tb in nodes:
            for _ in range(int(rng.integers(1, 3))):
                q = p + np.float32([rng.uniform(-10, 10), rng.uniform(-4, 12)])
                spr = _floret_cluster(int(rng.integers(6, 12)), rng)
                self.clusters.append(dict(wx=q[0] + ox, wy=q[1] + oy, t=tb + rng.uniform(0, 0.4), spr=spr,
                                          size=rng.uniform(34, 50), ph=rng.uniform(0, 6.28)))
        G = 96
        gy, gx = np.mgrid[0:G, 0:G].astype(np.float32) - G / 2
        rr = np.sqrt(gx * gx + gy * gy) / (G / 2)
        self.glow = (np.exp(-(rr / 0.45) ** 2)[..., None] * np.array([1.0, 0.72, 0.28], np.float32)).astype(np.float32)

    def draw(self, frame, t, cam, alpha=1.0):
        if t < self.t0:
            return
        if alpha < 0.999:
            tmp = frame.copy()
            self._draw(tmp, t, cam)
            frame += (tmp - frame) * alpha
        else:
            self._draw(frame, t, cam)

    def _draw(self, frame, t, cam):
        self.branch.draw(frame, t, cam)
        self.leaves.draw(frame, t, cam)
        for c in self.clusters:
            tt = t - c["t"]
            if tt <= 0:
                continue
            p = min(tt / 0.7, 1.0)
            grow = 0.25 + 0.75 * ease_out(p) + 0.08 * np.sin(min(p, 1.0) * np.pi)
            sx, sy, s = cam.to_screen(DEPTH, c["wx"], c["wy"])
            size = max(int(c["size"] * s * grow), 3)
            spr = cv2.resize(c["spr"], (size, size), interpolation=cv2.INTER_AREA)
            k = min(1.0, p * 1.6)
            a = spr[..., 3] * k
            over_roi(frame, int(sx - size / 2), int(sy - size / 2), spr[..., :3] * k, a)
            gs = int(size * 2.2)
            gl = cv2.resize(self.glow, (gs, gs), interpolation=cv2.INTER_AREA)
            pulse = 0.8 + 0.2 * np.sin(t * 2.0 + c["ph"])
            add_roi(frame, int(sx - gs / 2), int(sy - gs / 2), gl * (0.30 * k * pulse))
