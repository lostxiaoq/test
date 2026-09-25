"""Calligraphy that writes itself, and vermilion seals (印章)."""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ink import INK, VERMILION, PAPER, fbm, smoothstep, sstep, blur, ease_out

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def font_path(name):
    return os.path.join(FONT_DIR, name + ".ttf")


def glyph_mask(ch, font, size, ss=3):
    """Anti-aliased glyph mask cropped to its ink bbox; returns (mask, ox, oy)."""
    fnt = ImageFont.truetype(font, size * ss)
    pad = size * ss // 2
    canvas = Image.new("L", (size * ss * 2, size * ss * 2), 0)
    ImageDraw.Draw(canvas).text((pad, pad), ch, font=fnt, fill=255)
    a = np.asarray(canvas, np.float32) / 255.0
    a = cv2.resize(a, (a.shape[1] // ss, a.shape[0] // ss), interpolation=cv2.INTER_AREA)
    # layout box = the em square at (pad/ss, pad/ss)
    return a, pad // ss, pad // ss


class Glyph:
    def __init__(self, ch, font, size, cx, cy, t0, dur, seed):
        m, ox, oy = glyph_mask(ch, font, size)
        ys, xs = np.nonzero(m > 0.02)
        if len(xs) == 0:
            self.empty = True
            return
        self.empty = False
        # centre the ink bbox on (cx, cy)
        x0, x1, y0, y1 = xs.min() - 6, xs.max() + 7, ys.min() - 6, ys.max() + 7
        m = m[y0:y1, x0:x1]
        h, w = m.shape
        self.x = int(round(cx - w / 2))
        self.y = int(round(cy - h / 2))
        self.m = m.astype(np.float32)
        binm = (m > 0.5).astype(np.uint8)
        D = cv2.distanceTransform(binm, cv2.DIST_L2, 5).astype(np.float32)
        ref = max(np.percentile(D[binm > 0], 92), 1.0) if binm.any() else 1.0
        self.Dn = np.clip(D / ref, 0, 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        n = fbm(h, w, max(size * 0.4, 8), 3, seed=seed)
        self.T = (t0 + dur * (0.64 * yy / h + 0.36 * xx / w) + 0.05 * dur * n).astype(np.float32)
        self.t_first = float(self.T.min())
        self.t_last = float(self.T.max()) + 0.3
        self.halo = blur(m, max(size / 36.0, 1.2))
        self.diss = (0.5 + 0.5 * np.clip(fbm(h, w, max(size * 0.25, 6), 4, seed=seed + 1), -1, 1)).astype(np.float32)
        self.size = size

    def alpha(self, t, t_out=None, out_dur=0.9):
        if self.empty or t < self.t_first:
            return None
        if t < self.t_last:
            p = np.clip((t - self.T) / 0.22, 0, 1)
            th = 1.1 * (1.0 - p) - 0.1
            core = self.m * smoothstep(th - 0.12, th + 0.06, self.Dn)
            wet = self.halo * np.clip((t - self.T - 0.05) / 0.5, 0, 1) * 0.16
        else:
            core = self.m
            wet = self.halo * 0.16
        a = 1.0 - (1.0 - core) * (1.0 - wet)
        if t_out is not None and t > t_out:
            q = (t - t_out) / out_dur
            if q >= 1.0:
                return None
            # ink dissolves into the paper, lightest parts first
            a = a * smoothstep(q * 1.2 - 0.1, q * 1.2 + 0.05, self.diss) * (1.0 - 0.35 * q)
        return a


class Calligraphy:
    """A block of text written glyph by glyph; vertical columns right-to-left."""

    def __init__(self, columns, font, size, x_right, y_top, t0, *, col_gap=1.25, spacing=1.10,
                 char_dur=0.42, char_gap=0.26, col_pause=0.35, ink=INK, dens=0.93,
                 t_out=None, out_stagger=0.06, out_dur=0.9, seed=0, glow=None):
        self.glyphs = []
        self.ink = np.asarray(ink, np.float32)
        self.dens = dens
        self.t_out, self.out_stagger, self.out_dur = t_out, out_stagger, out_dur
        self.glow = glow
        t = t0
        k = 0
        for ci, col in enumerate(columns):
            cx = x_right - ci * size * col_gap
            y = y_top
            for ch in col:
                if ch == " ":
                    y += size * spacing * 0.5
                    continue
                g = Glyph(ch, font, size, cx, y + size / 2, t, char_dur, seed + k * 17)
                g.k = k
                self.glyphs.append(g)
                y += size * spacing
                t += char_gap
                k += 1
            t += col_pause
        self.t_end_write = t

    def draw(self, frame, t):
        H_, W_ = frame.shape[:2]
        for g in self.glyphs:
            if g.empty:
                continue
            t_out = None if self.t_out is None else self.t_out + g.k * self.out_stagger
            a = g.alpha(t, t_out, self.out_dur)
            if a is None:
                continue
            h, w = a.shape
            X0, Y0 = max(g.x, 0), max(g.y, 0)
            X1, Y1 = min(g.x + w, W_), min(g.y + h, H_)
            if X1 <= X0 or Y1 <= Y0:
                continue
            aa = a[Y0 - g.y:Y1 - g.y, X0 - g.x:X1 - g.x] * self.dens
            sub = frame[Y0:Y1, X0:X1]
            if self.glow is not None:
                gl = blur(aa, g.size / 9.0)[..., None] * self.glow
                sub[...] = 1.0 - (1.0 - sub) * (1.0 - np.clip(gl, 0, 1))
            sub *= (1.0 - aa)[..., None]
            sub += aa[..., None] * self.ink


def make_seal(chars, size, seed=5, font=None, relief="yin"):
    """Vermilion seal. relief='yin' = white characters cut into red (阴文)."""
    ss = 3
    S = size * ss
    font = font or font_path("NotoSerifSCwght900")
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    border = int(S * 0.07)
    n = len(chars)
    if n == 4:     # read right column top-down, then left column
        cells = [(1, 0), (1, 1), (0, 0), (0, 1)]
        cs = (S - 2 * border) // 2
        fnt = ImageFont.truetype(font, int(cs * 0.92))
        for ch, (cx, cy) in zip(chars, cells):
            bb = d.textbbox((0, 0), ch, font=fnt)
            gw, gh = bb[2] - bb[0], bb[3] - bb[1]
            x = border + cx * cs + (cs - gw) / 2 - bb[0]
            y = border + cy * cs + (cs - gh) / 2 - bb[1]
            d.text((x, y), ch, font=fnt, fill=255)
    elif n == 2:   # vertical pair
        cs = (S - 2 * border) // 2
        fnt = ImageFont.truetype(font, int(cs * 0.95))
        for i, ch in enumerate(chars):
            bb = d.textbbox((0, 0), ch, font=fnt)
            gw, gh = bb[2] - bb[0], bb[3] - bb[1]
            x = (S - gw) / 2 - bb[0]
            y = border + i * cs + (cs - gh) / 2 - bb[1]
            d.text((x, y), ch, font=fnt, fill=255)
    else:
        fnt = ImageFont.truetype(font, int((S - 2 * border) * 0.9))
        bb = d.textbbox((0, 0), chars, font=fnt)
        d.text(((S - bb[2] + bb[0]) / 2 - bb[0], (S - bb[3] + bb[1]) / 2 - bb[1]), chars, font=fnt, fill=255)
    txt = np.asarray(img, np.float32) / 255.0
    # slightly swollen strokes like carved stone
    txt = np.clip(blur(txt, 1.2 * ss / 3) * 1.3, 0, 1)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    n1 = fbm(S, S, S / 6, 4, seed=seed)
    n2 = fbm(S, S, S / 40, 3, seed=seed + 1)
    # rough outer edge
    e = np.minimum(np.minimum(xx, S - 1 - xx), np.minimum(yy, S - 1 - yy))
    outer = smoothstep(0, 2.5 * ss, e + 2.2 * ss * n1)
    if relief == "yin":
        red = outer * (1.0 - txt)
    else:
        frame_line = outer * (1.0 - smoothstep(border * 0.55, border * 0.75, e + ss * n1))
        red = np.maximum(frame_line, txt * outer)
    # stamp texture: specks where the paste did not take
    specks = smoothstep(0.55, 0.9, n2 * 0.6 + 0.4 * rng.standard_normal((S, S)).astype(np.float32) * 0.5)
    red = red * (1.0 - 0.55 * specks) * (0.88 + 0.12 * np.clip(n1, -1, 1))
    red = cv2.resize(red, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(red, 0, 1).astype(np.float32)


class Seal:
    def __init__(self, chars, size, cx, cy, t_stamp, seed=5, relief="yin"):
        self.a = make_seal(chars, size, seed, relief=relief)
        self.cx, self.cy, self.t, self.size = cx, cy, t_stamp, size
        self.color = VERMILION

    def draw(self, frame, t, fade=1.0):
        if t < self.t:
            return
        p = min((t - self.t) / 0.16, 1.0)
        scale = 1.0 + 0.22 * (1.0 - ease_out(p))
        a = self.a
        if scale != 1.0:
            s = max(int(self.size * scale), 2)
            a = cv2.resize(a, (s, s), interpolation=cv2.INTER_LINEAR)
        a = a * min(1.0, p * 1.4) * 0.94 * fade
        h, w = a.shape
        x0, y0 = int(self.cx - w / 2), int(self.cy - h / 2)
        H_, W_ = frame.shape[:2]
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, W_), min(y0 + h, H_)
        aa = a[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None]
        sub = frame[Y0:Y1, X0:X1]
        # pigment: multiply-like so paper texture shows through the red
        sub *= (1.0 - aa) + aa * (self.color / PAPER)
