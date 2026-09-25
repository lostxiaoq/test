"""Animated elements: fishing boat, moon path on the water, wild geese,
sky lanterns (孔明灯) and drifting golden petals."""
import numpy as np
import cv2
from ink import (INK, PAPER, GOLD, fbm, smoothstep, sstep, blur, render_strokes, over_roi, add_roi)


def _supersample_poly(canvas, pts, val, ss):
    cv2.fillPoly(canvas, [np.int32(np.asarray(pts) * ss)], val, cv2.LINE_AA)


# ---------------------------------------------------------------------- boat
class Boat:
    """A small covered fishing boat with a boatman and a bow lantern."""

    def __init__(self, seed=41):
        ss = 4
        w, h = 260, 150
        self.w, self.h = w, h
        self.water = 108                       # waterline row inside the sprite
        c = np.zeros((h * ss, w * ss), np.float32)
        rng = np.random.default_rng(seed)
        # hull: a crescent with raised bow and stern
        top = [(24, 92), (60, 101), (120, 105), (180, 102), (222, 90)]
        bot = [(210, 104), (170, 113), (120, 116), (70, 113), (36, 104)]
        _supersample_poly(c, top + bot, 0.88, ss)
        # canopy (乌篷): an arch of woven bamboo
        can = np.zeros_like(c)
        cv2.ellipse(can, (118 * ss, 101 * ss), (36 * ss, 27 * ss), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
        stripes = (np.sin(np.arange(w * ss) / (3.2 * ss)) > 0.2).astype(np.float32)[None, :]
        can = can * (0.55 + 0.25 * stripes)
        cv2.ellipse(can, (118 * ss, 101 * ss), (36 * ss, 27 * ss), 0, 180, 360, 0.95, int(2.2 * ss), cv2.LINE_AA)
        c = np.maximum(c, can)
        # boatman at the stern: cloak, conical hat, a long pole into the water
        _supersample_poly(c, [(172, 100), (170, 84), (174, 72), (183, 72), (188, 86), (190, 100)], 0.9, ss)
        _supersample_poly(c, [(160, 73), (178, 60), (198, 73), (179, 70)], 0.95, ss)
        cv2.line(c, (int(196 * ss), int(40 * ss)), (int(150 * ss), int(140 * ss)), 0.9, int(1.4 * ss), cv2.LINE_AA)
        # bow post for the lantern
        cv2.line(c, (int(30 * ss), int(95 * ss)), (int(30 * ss), int(80 * ss)), 0.9, int(1.2 * ss), cv2.LINE_AA)
        a = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        a = blur(a, 0.5)
        tex = fbm(h, w, 10, 3, seed=seed)
        a = np.clip(a * (0.92 + 0.08 * tex), 0, 1)
        self.a = a.astype(np.float32)
        self.rgb = (self.a[..., None] * INK).astype(np.float32)
        # reflection: flipped about the waterline, pale and blurred
        wl = self.water
        ref = np.zeros_like(self.a)
        part = self.a[:wl][::-1]
        ref[wl:wl + part.shape[0]] = part[:h - wl]
        ref = blur(ref, 1.2) * 0.28
        ref *= smoothstep(h, wl, np.arange(h, dtype=np.float32))[:, None]
        self.ref = ref.astype(np.float32)
        # lantern glow sprite
        gs = 70
        yy, xx = np.mgrid[0:2 * gs, 0:2 * gs].astype(np.float32) - gs
        r = np.sqrt(xx * xx + yy * yy)
        g = 0.85 * np.exp(-(r / 3.5) ** 2) + 0.35 * np.exp(-r / 12.0) + 0.12 * np.exp(-r / 34.0)
        self.glow = (g[..., None] * np.array([1.0, 0.72, 0.36], np.float32)).astype(np.float32)
        self.gs = gs
        self.lantern_xy = (30, 78)

    def draw(self, frame, t, sx, sy, scale, alpha=1.0, ripple_t=0.0):
        """Draw with the waterline anchored at screen (sx, sy)."""
        if alpha <= 0.0:
            return
        a, ref = self.a, self.ref
        if abs(scale - 1.0) > 0.01:
            nw, nh = int(self.w * scale), int(self.h * scale)
            a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            ref = cv2.resize(ref, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        h, w = a.shape
        # animated ripples in the reflection
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        mx = xx + 2.2 * np.sin(yy * 0.55 + ripple_t * 3.1) * smoothstep(self.water * scale, h, yy)
        ref = cv2.remap(ref, mx, yy, cv2.INTER_LINEAR)
        x0 = int(round(sx - w / 2))
        y0 = int(round(sy - self.water * scale))
        ra = ref * alpha
        over_roi(frame, x0, y0, ra[..., None] * INK, ra)
        aa = a * alpha
        over_roi(frame, x0, y0, aa[..., None] * INK, aa)
        # lantern
        lx = int(round(x0 + self.lantern_xy[0] * scale)) - self.gs
        ly = int(round(y0 + self.lantern_xy[1] * scale)) - self.gs
        fl = 0.85 + 0.15 * np.sin(t * 7.3) * np.sin(t * 3.1 + 1)
        add_roi(frame, lx, ly, self.glow * (alpha * fl))
        # its reflection on the water
        add_roi(frame, lx, ly + int(2 * (self.water - self.lantern_xy[1]) * scale), self.glow * (0.35 * alpha * fl))


# ------------------------------------------------------------ moon on water
class MoonPath:
    """Broken golden strokes of moonlight on the water under the moon."""

    def __init__(self, seed=51, n=34):
        rng = np.random.default_rng(seed)
        self.rows = []
        for i in range(n):
            f = (i + rng.uniform(0, 0.6)) / n              # 0 at horizon .. 1 near viewer
            self.rows.append(dict(f=f, w=(10 + 120 * f ** 1.3) * rng.uniform(0.5, 1.2),
                                  dx=rng.normal(0, 8 + 30 * f), ph=rng.uniform(0, 6.28),
                                  sp=rng.uniform(1.2, 3.2), th=1.2 + 3.2 * f * rng.uniform(0.6, 1.2)))

    def draw(self, frame, t, mx, y_top, y_bot, strength=1.0):
        if strength <= 0.001:
            return
        H_, W_ = frame.shape[:2]
        pad = 200
        x0 = int(mx) - pad
        y0 = int(y_top) - 10
        hh = int(y_bot - y_top) + 20
        light = np.zeros((hh, 2 * pad), np.float32)
        for r in self.rows:
            y = r["f"] * (y_bot - y_top) + 10
            wig = 6 * np.sin(t * r["sp"] + r["ph"])
            cx = pad + r["dx"] + wig
            flick = 0.55 + 0.45 * np.sin(t * r["sp"] * 1.7 + r["ph"] * 2)
            ww = r["w"] * (0.75 + 0.25 * np.sin(t * r["sp"] * 0.8 + r["ph"]))
            cv2.ellipse(light, (int(cx * 4), int(y * 4)), (int(ww / 2 * 4), max(1, int(r["th"] / 2 * 4))),
                        0, 0, 360, float(flick * (1.0 - 0.35 * r["f"])), -1, cv2.LINE_AA, 2)
        light = blur(light, 1.3)
        col = light[..., None] * np.array([1.0, 0.86, 0.55], np.float32) * (0.85 * strength)
        # soft column of glow
        yy = np.arange(hh, dtype=np.float32)[:, None]
        xx = np.arange(2 * pad, dtype=np.float32)[None, :] - pad
        colg = np.exp(-(xx / 70.0) ** 2) * smoothstep(0, 40, yy) * np.exp(-yy / 260.0) * 0.20 * strength
        col += colg[..., None] * np.array([1.0, 0.8, 0.45], np.float32)
        add_roi(frame, x0, y0, col)


# ---------------------------------------------------------------------- geese
class Geese:
    """A skein of wild geese (雁) in loose V formation, wings flapping."""

    def __init__(self, n=7, seed=61, size=58):
        rng = np.random.default_rng(seed)
        self.size = size
        self.frames = [self._goose(ph / 16.0 * 2 * np.pi, size) for ph in range(16)]
        # formation offsets (leader first, flying left)
        self.birds = []
        for i in range(n):
            side = 1 if i % 2 else -1
            k = (i + 1) // 2
            self.birds.append(dict(dx=k * 64 + rng.uniform(-8, 8), dy=side * k * 30 + rng.uniform(-6, 6),
                                   ph=rng.uniform(0, 1), rate=rng.uniform(2.2, 2.7),
                                   sc=rng.uniform(0.85, 1.05)))

    @staticmethod
    def _goose(phase, size):
        ss = 4
        S = size * ss
        c = np.zeros((S, S), np.float32)
        cx, cy = S / 2, S / 2
        u = size * ss / 46.0
        # body, neck and head facing left
        cv2.ellipse(c, (int(cx), int(cy)), (int(9 * u), int(3.2 * u)), -4, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.line(c, (int(cx - 7 * u), int(cy - 0.5 * u)), (int(cx - 17 * u), int(cy - 2.5 * u)), 1.0, max(1, int(1.8 * u)), cv2.LINE_AA)
        cv2.ellipse(c, (int(cx - 18 * u), int(cy - 2.8 * u)), (int(2.4 * u), int(1.7 * u)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.fillPoly(c, [np.int32([(cx + 8 * u, cy - 1 * u), (cx + 14 * u, cy + 1.5 * u), (cx + 8 * u, cy + 2 * u)])], 1.0, cv2.LINE_AA)
        # wings: shoulder -> elbow -> tip, tip swings with the beat
        s = np.sin(phase)
        for side, off, val in ((-1, 0.0, 0.75), (1, 1.2, 1.0)):
            sh = np.float32([cx + 1 * u, cy - 1 * u])
            tip = np.float32([cx + (6 + 5 * abs(s)) * u, cy - (20 * s + 2) * u * (0.8 if side < 0 else 1.0)])
            el = (sh + tip) / 2 + np.float32([-3 * u, 3 * u * s])
            poly = np.int32([sh + np.float32([-4 * u, 1 * u]), el + np.float32([-1.5 * u, 0]), tip,
                             el + np.float32([1.5 * u, 1.2 * u]), sh + np.float32([4 * u, 1 * u])])
            cv2.fillPoly(c, [poly], val, cv2.LINE_AA)
        a = cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA)
        return blur(a, 0.45).astype(np.float32)

    def draw(self, frame, t, lead_x, lead_y, scale=1.0, alpha=1.0, dens=0.86):
        for b in self.birds:
            ph = (t * b["rate"] + b["ph"]) % 1.0
            spr = self.frames[int(ph * 16) % 16]
            sc = scale * b["sc"]
            if abs(sc - 1.0) > 0.02:
                s = max(int(self.size * sc), 6)
                spr = cv2.resize(spr, (s, s), interpolation=cv2.INTER_AREA)
            h, w = spr.shape
            x = lead_x + b["dx"] * scale - w / 2
            y = lead_y + b["dy"] * scale - h / 2 + 3 * np.sin(t * 1.3 + b["ph"] * 6)
            a = spr * (alpha * dens)
            over_roi(frame, int(round(x)), int(round(y)), a[..., None] * INK, a)


# ------------------------------------------------------------------ lanterns
class Lanterns:
    """Sky lanterns drifting up from the water with warm, flickering light."""

    def __init__(self, n=40, seed=71, t0=27.5):
        rng = np.random.default_rng(seed)
        self.sprite, self.glow = self._sprite(96)
        self.items = []
        for i in range(n):
            depth = rng.uniform(0.0, 1.0) ** 1.4           # 0 = far, 1 = near
            size = 12 + 48 * depth
            if depth < 0.3:
                x = rng.uniform(380, 1540)
            else:
                x = rng.uniform(60, 1880)
            if 470 < x < 760 and rng.random() < 0.6:      # keep the moon mostly clear
                x += rng.choice([-1, 1]) * 260
            start = t0 + rng.uniform(0, 4.5) + (1 - depth) * 0.6
            y0 = rng.uniform(760, 880) if depth < 0.35 else rng.uniform(1000, 1160)
            self.items.append(dict(x=x, y0=y0, size=size, depth=depth, t0=start,
                                   v=16 + 58 * depth, sway=rng.uniform(4, 16), ph=rng.uniform(0, 6.28),
                                   fl=rng.uniform(5, 9)))
        self.items.sort(key=lambda d: d["depth"])

    @staticmethod
    def _sprite(S):
        h = int(S * 1.3)
        w = S
        ss = 4
        c = np.zeros((h * ss, w * ss), np.float32)
        top_w, bot_w = 0.92, 0.66
        poly = []
        for k in range(21):     # rounded shoulders
            a = np.pi * k / 20
            poly.append((w / 2 - np.cos(a) * top_w * w / 2, h * 0.10 - np.sin(a) * h * 0.08))
        poly += [(w / 2 + bot_w * w / 2, h * 0.94), (w / 2 - bot_w * w / 2, h * 0.94)]
        poly = [(x, y + h * 0.04) for x, y in poly]
        cv2.fillPoly(c, [np.int32(np.array(poly) * ss)], 1.0, cv2.LINE_AA)
        a = cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - w / 2) / (w / 2)
        ny = yy / h
        heat = np.exp(-((nx / 0.55) ** 2) - ((ny - 0.92) / 0.45) ** 2)       # flame at the mouth
        edge = np.abs(nx) ** 3
        r = 0.95 + 0.05 * heat
        g = 0.48 + 0.46 * heat - 0.14 * edge
        b = 0.16 + 0.52 * heat ** 2 - 0.08 * edge
        rgb = np.stack([r, g, b], -1)
        rgb *= (1.0 - 0.25 * (1 - ny) ** 3)[..., None]
        # thin paper ribs
        ribs = (np.abs(np.sin(nx * np.pi * 2.0)) < 0.06).astype(np.float32) * 0.18
        rgb *= (1.0 - ribs)[..., None]
        rgb = np.clip(rgb, 0, 1) * a[..., None]
        # glow sprite, 3x the lantern
        G = int(S * 4)
        gy, gx = np.mgrid[0:G, 0:G].astype(np.float32) - G / 2
        rr = np.sqrt(gx * gx + (gy - S * 0.15) ** 2) / S
        glow = 0.60 * np.exp(-(rr / 0.6) ** 2) + 0.30 * np.exp(-rr / 0.75)
        glow = glow[..., None] * np.array([1.0, 0.62, 0.25], np.float32)
        return (np.concatenate([rgb, a[..., None]], -1).astype(np.float32), glow.astype(np.float32))

    def draw(self, frame, t, cam_dx=0.0, cam_dy=0.0, alpha=1.0, depth_range=(0.0, 1.01), avoid=()):
        """avoid: [(x0, y0, x1, y1, strength, max_depth)] screen boxes lanterns fade out of."""
        for it in self.items:
            if not (depth_range[0] <= it["depth"] < depth_range[1]):
                continue
            tt = t - it["t0"]
            if tt < 0:
                continue
            k = sstep(0.0, 1.2, tt) * alpha
            if k <= 0:
                continue
            par = 0.3 + 0.9 * it["depth"]
            y = it["y0"] - it["v"] * tt - 4 * tt * tt * (0.3 + it["depth"]) + cam_dy * par
            x = it["x"] + it["sway"] * np.sin(0.7 * tt + it["ph"]) + cam_dx * par
            s = it["size"] * (1.0 - 0.10 * sstep(0, 12, tt))
            if y < -3 * s:
                continue
            for (ax0, ay0, ax1, ay1, st, md) in avoid:
                if it["depth"] < md:
                    inside = min(sstep(ax0 - 40, ax0 + 30, x), 1 - sstep(ax1 - 30, ax1 + 40, x)) * \
                             min(sstep(ay0 - 40, ay0 + 30, y), 1 - sstep(ay1 - 30, ay1 + 40, y))
                    k *= 1.0 - st * inside
            if k <= 0.003:
                continue
            fl = 0.82 + 0.18 * np.sin(t * it["fl"] + it["ph"]) * np.sin(t * it["fl"] * 0.37 + 2)
            sw, sh = max(int(s), 3), max(int(s * 1.3), 4)
            spr = cv2.resize(self.sprite, (sw, sh), interpolation=cv2.INTER_AREA)
            if it["depth"] < 0.25:
                spr = blur(spr, 0.6)
            gsz = max(int(s * 4), 8)
            gl = cv2.resize(self.glow, (gsz, gsz), interpolation=cv2.INTER_AREA)
            gl *= k * fl * (0.55 + 0.45 * it["depth"])
            add_roi(frame, int(x - gsz / 2), int(y - gsz / 2), gl)
            a = spr[..., 3] * k * 0.96
            rgb = spr[..., :3] * (k * 0.96) * (0.85 + 0.15 * fl)
            over_roi(frame, int(x - sw / 2), int(y - sh / 2), rgb, a)


# ------------------------------------------------------------------- petals
class Petals:
    """Tiny golden osmanthus florets drifting down, a few big out-of-focus ones."""

    def __init__(self, n=150, seed=81, t0=22.5, src=((100, 40), (950, 300))):
        rng = np.random.default_rng(seed)
        self.items = []
        (x0, y0), (x1, y1) = src
        for i in range(n):
            near = rng.random() < 0.12
            self.items.append(dict(
                x=rng.uniform(x0, x1), y=rng.uniform(y0, y1), t0=t0 + rng.uniform(0, 12),
                vy=rng.uniform(28, 60) * (1.8 if near else 1.0), vx=rng.uniform(18, 50),
                sway=rng.uniform(10, 30), ph=rng.uniform(0, 6.28), rot=rng.uniform(2, 6),
                size=rng.uniform(3.0, 5.2) * (3.2 if near else 1.0), near=near))
        self.dot = self._floret(64)

    @staticmethod
    def _floret(S):
        ss = 4
        c = np.zeros((S * ss, S * ss), np.float32)
        cc = S * ss / 2
        for k in range(4):
            a = k * np.pi / 2 + 0.3
            px, py = cc + np.cos(a) * S * ss * 0.2, cc + np.sin(a) * S * ss * 0.2
            cv2.ellipse(c, (int(px), int(py)), (int(S * ss * 0.2), int(S * ss * 0.13)), np.degrees(a), 0, 360, 1.0, -1, cv2.LINE_AA)
        a = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)
        return blur(a, 0.8)

    def draw(self, frame, t, alpha=1.0, near=None):
        col = np.array([0.98, 0.72, 0.22], np.float32)
        for it in self.items:
            if near is not None and it["near"] != near:
                continue
            tt = t - it["t0"]
            if tt < 0 or tt > 14:
                continue
            x = it["x"] + it["vx"] * tt + it["sway"] * np.sin(1.3 * tt + it["ph"])
            y = it["y"] + it["vy"] * tt
            if y > 1150 or x > 2000:
                continue
            flip = abs(np.cos(it["rot"] * tt * 0.5 + it["ph"]))
            s = it["size"]
            sw, sh = max(int(2 * s * (0.35 + 0.65 * flip)), 2), max(int(2 * s), 2)
            spr = cv2.resize(self.dot, (sw, sh), interpolation=cv2.INTER_AREA)
            if it["near"]:
                spr = blur(spr, s * 0.35)
            k = sstep(0, 0.6, tt) * (1 - sstep(10, 14, tt)) * alpha * (0.55 if it["near"] else 0.95)
            a = spr * k
            over_roi(frame, int(x - sw / 2), int(y - sh / 2), a[..., None] * col, a)
            add_roi(frame, int(x - sw / 2), int(y - sh / 2), (a * 0.35)[..., None] * col)
