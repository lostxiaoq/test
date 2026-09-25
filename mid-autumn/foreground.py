"""Foreground: dark rocky banks, an old pine leaning over the water, reeds."""
import numpy as np
import cv2
from ink import W, H, INK, fbm, smoothstep, blur, render_strokes
from mountains import paint_range
from world import Layer, rgba_from, sweep_T


def _curve(p0, p1, bend, n=24, wiggle=0.0, rng=None):
    """Quadratic curve from p0 to p1 bent sideways by `bend` (fraction of length)."""
    p0, p1 = np.float32(p0), np.float32(p1)
    mid = (p0 + p1) / 2
    d = p1 - p0
    nrm = np.float32([-d[1], d[0]])
    ctrl = mid + nrm * bend
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
    if wiggle and rng is not None:
        L = float(np.linalg.norm(d))
        tt = t[:, 0]
        off = (np.sin(2 * np.pi * (rng.uniform(1.0, 1.8) * tt) + rng.uniform(0, 6.28)) * 0.6 +
               np.sin(2 * np.pi * (rng.uniform(2.2, 3.4) * tt) + rng.uniform(0, 6.28)) * 0.4)
        off = (off - (1 - tt) * off[0] - tt * off[-1]).astype(np.float32)
        nn = nrm / (np.linalg.norm(nrm) + 1e-6)
        pts += (off[:, None] * wiggle * 0.08 * L) * nn
    return pts


def _needles(dens, tmap, cx, cy, r, t0, rng, ss=4, up=True, n=None):
    """A fan of pine needles (松针) radiating from (cx, cy)."""
    h, w = dens.shape
    pad = int(r * 1.2) + 4
    x0, y0 = int(cx - pad), int(cy - pad)
    if x0 < 0 or y0 < 0 or x0 + 2 * pad >= w or y0 + 2 * pad >= h:
        return
    size = 2 * pad
    spr = np.zeros((size * ss, size * ss), np.float32)
    c = np.float32([(cx - x0) * ss, (cy - y0) * ss])
    n = n or int(rng.integers(28, 40))
    a0, a1 = (-172, -8) if up else (-200, 20)
    for i in range(n):
        ang = np.radians(rng.uniform(a0, a1))
        L = r * rng.uniform(0.7, 1.05) * ss
        e = c + np.float32([np.cos(ang) * L, np.sin(ang) * L * 0.55])
        s = c + np.float32([np.cos(ang) * L * 0.12, np.sin(ang) * L * 0.1])
        cv2.line(spr, tuple(np.int32(s)), tuple(np.int32(e)), float(rng.uniform(0.75, 1.0)),
                 max(1, int(0.9 * ss)), cv2.LINE_AA)
    cv2.ellipse(spr, tuple(np.int32(c)), (int(r * 0.22 * ss), int(r * 0.12 * ss)), 0, 0, 360, 0.95, -1, cv2.LINE_AA)
    spr = cv2.resize(spr, (size, size), interpolation=cv2.INTER_AREA)
    spr = blur(spr, 0.4)
    sub = dens[y0:y0 + size, x0:x0 + size]
    tsub = tmap[y0:y0 + size, x0:x0 + size]
    upd = spr > 0.02
    tsub[upd] = np.minimum(tsub[upd], t0 + rng.uniform(0, 0.12))
    np.maximum(sub, spr * 0.92, out=sub)


def _soft_wash(dens, cx, cy, rx, ry, val, sigma):
    h, w = dens.shape
    m = np.zeros_like(dens)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    m = blur(m, sigma) * val
    np.copyto(dens, 1.0 - (1.0 - dens) * (1.0 - m))


def build_near(seed=700, t_start=5.0):
    m = 460
    w, h = W + 2 * m, 700
    oy = H + 60 - h            # layer bottom sits below the frame
    base = h - 10
    rng = np.random.default_rng(seed)

    def lx(wx):
        return wx + m

    def ly(wy):
        return wy - oy

    # ---- rocky banks
    peaks = [(lx(-160), 470, 230, 2.1, 0.15), (lx(150), 330, 200, 2.0, -0.1),
             (lx(420), 190, 170, 2.2, -0.2), (lx(640), 70, 150, 2.0, 0.0),
             (lx(1780), 150, 150, 2.2, 0.2), (lx(2000), 290, 190, 2.1, -0.1),
             (lx(2260), 380, 200, 2.0, 0.0)]
    d, a, ridges = paint_range(h, w, base, peaks, seed + 1, dens=0.80, fade=0.30, edge=0.9,
                               tex=0.18, line=0.60, fold=0.0, moss=50, mist_h=60,
                               shade=0.28, ridge_amp=0.06, streak_w=9.0, dry=0.5, strokes=55,
                               stroke_w=(2.5, 7.0), stroke_len=(0.10, 0.30), stroke_d=(0.35, 0.75))
    T = sweep_T(h, w, t_start, 1.8, d, seed + 2)

    # ---- the old pine on the left bank (outlined trunk + wash + bark scales)
    t0 = t_start + 0.9
    sd, st, tips = _paint_pine(h, w, rng, t0, seed, lx, ly)
    # needle clusters with a pale wash underneath for volume
    nd = np.zeros((h, w), np.float32)
    nt = np.full((h, w), np.inf, np.float32)
    wash = np.zeros((h, w), np.float32)
    for pts, tb in tips:
        for f in (0.5, 0.68, 0.84, 1.0):
            q = pts[min(int(f * (len(pts) - 1)), len(pts) - 1)]
            cx = q[0] + rng.uniform(-8, 8)
            cy = q[1] + rng.uniform(-5, 3)
            r = rng.uniform(17, 27)
            _soft_wash(wash, cx, cy + 4, r * 1.6, r * 0.45, 0.10, 9)
            _needles(nd, nt, cx, cy, r, tb + 0.1 * f, rng)
    np.maximum(sd, nd, out=sd)
    np.minimum(st, nt, out=st)
    wt = np.where(wash > 0.01, t0 + 1.4, np.inf).astype(np.float32)
    d = 1.0 - (1.0 - d) * (1.0 - wash)
    np.minimum(T, wt, out=T)
    d = 1.0 - (1.0 - d) * (1.0 - sd)
    np.minimum(T, st, out=T)

    # ---- reeds on the right bank (蒹葭): bending stems, feathery plumes
    rs = []
    rng2 = np.random.default_rng(seed + 90)
    for i in range(15):
        bx = lx(1760 + rng2.normal(0, 55))
        by = ly(1040 + rng2.uniform(-15, 25))
        L = rng2.uniform(170, 290)
        lean = rng2.uniform(-0.5, -0.05)
        end_ = (bx + lean * L, by - L)
        pts = _curve((bx, by), end_, rng2.uniform(-0.16, -0.04), 18)
        tr = t_start + 1.2 + 0.05 * i
        rs.append(dict(pts=pts, widths=np.linspace(1.9, 0.6, len(pts)), t0=tr, t1=tr + 0.5,
                       dens=0.78, dry=0.25, seed=seed + 100 + i))
        if i % 3 == 0:      # one long thin leaf, bent over
            q = pts[int(len(pts) * rng2.uniform(0.25, 0.45))]
            e = (q[0] + rng2.uniform(-110, -40), q[1] + rng2.uniform(-40, 30))
            lp = _curve(q, e, rng2.uniform(0.15, 0.35), 14)
            wl = np.sin(np.linspace(0.1, np.pi * 0.92, len(lp))) ** 0.7 * 2.4
            rs.append(dict(pts=lp, widths=wl, t0=tr + 0.2, t1=tr + 0.6, dens=0.6, dry=0.35,
                           seed=seed + 140 + i))
        # plume: fine filaments hanging from the top fifth of the stem
        for j in range(14):
            q = pts[len(pts) - 1 - int(rng2.uniform(0, 0.22) * len(pts))]
            e = (q[0] + rng2.uniform(-26, 4), q[1] + rng2.uniform(8, 28))
            pp = _curve(q, e, rng2.uniform(0.1, 0.4), 6)
            rs.append(dict(pts=pp, widths=np.linspace(1.0, 0.35, len(pp)), t0=tr + 0.45, t1=tr + 0.7,
                           dens=0.30, dry=0.5, seed=seed + 300 + 14 * i + j))
    rd, rt = render_strokes(h, w, rs)
    d = np.maximum(d, rd)
    np.minimum(T, rt, out=T)
    a = np.maximum(a, 0.0)
    return Layer("near", 1.35, rgba_from(d, a), (-m, oy), T, ramp=0.25)


def _bezier(pts, n):
    pts = np.float32(pts)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    if len(pts) == 3:
        return (1 - t) ** 2 * pts[0] + 2 * (1 - t) * t * pts[1] + t ** 2 * pts[2]
    return ((1 - t) ** 3 * pts[0] + 3 * (1 - t) ** 2 * t * pts[1] +
            3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3])


def _paint_pine(h, w, rng, t0, seed, lx, ly):
    """写意 pine: a bold wet trunk stroke with dry edges, a darker shade stroke,
    bark dabs, and limbs that droop then lift."""
    strokes = []
    c = _bezier([(lx(200), ly(948)), (lx(345), ly(875)), (lx(325), ly(715)), (lx(528), ly(596))], 44)
    tang = np.gradient(c, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-6
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    s = np.linspace(0, 1, len(c))
    width = (17.0 - 11.0 * s ** 0.7) * (1 + 0.10 * np.sin(s * 21 + 1.3) + 0.05 * np.sin(s * 53))
    strokes.append(dict(pts=c, widths=width, t0=t0, t1=t0 + 0.8, dens=0.62, dry=0.55,
                        seed=seed + 21, fade_end=0.1))
    shade = c - nrm * (width * 0.45)[:, None]
    strokes.append(dict(pts=shade, widths=width * 0.42, t0=t0 + 0.1, t1=t0 + 0.9, dens=0.95,
                        dry=0.6, seed=seed + 22, fade_end=0.15))
    lit = c + nrm * (width * 0.62)[:, None]
    strokes.append(dict(pts=lit[4:-6], widths=np.full(len(lit) - 10, 1.6), t0=t0 + 0.2, t1=t0 + 0.9,
                        dens=0.9, dry=0.7, seed=seed + 23))
    # bark dabs and knots
    for i in range(16):
        k = int(rng.uniform(0.06, 0.85) * len(c))
        wv = width[k]
        ctr = c[k] + nrm[k] * wv * rng.uniform(-0.5, 0.45)
        sz = wv * rng.uniform(0.25, 0.45)
        a0 = ctr - nrm[k] * sz
        a1 = ctr + nrm[k] * sz
        mid = ctr - tang[k] * sz * 0.8
        arc = _bezier([a0, mid, a1], 7)
        strokes.append(dict(pts=arc, widths=np.full(7, rng.uniform(1.0, 1.6)), t0=t0 + 0.5, t1=t0 + 0.6,
                            dens=float(rng.uniform(0.6, 0.9)), dry=0.2, seed=seed + 60 + i))
    # roots gripping the rock
    for dx, dy in [(-50, 16), (42, 22), (-10, 30)]:
        r0 = c[2]
        pts = _bezier([r0, r0 + np.float32([dx * 0.4, dy * 0.1]), r0 + np.float32([dx, dy])], 8)
        strokes.append(dict(pts=pts, widths=np.linspace(5.5, 1.0, 8), t0=t0 - 0.1, t1=t0 + 0.15,
                            dens=0.85, dry=0.5, seed=seed + 80 + dx))
    # limbs: out, droop, then lift at the tip (pine 'hand')
    limbs = [  # (index on trunk, dx, droop, rise, width)
        (26, 300, 30, 36, 5.5),
        (40, 265, 24, 30, 5.0),
        (34, -170, 22, 50, 4.2),
        (30, -185, 28, 16, 4.0),
        (43, 70, -30, 105, 3.6),
    ]
    tips = []
    for k, (si, dx, droop, rise, w0) in enumerate(limbs):
        p0 = c[si] + nrm[si] * width[si] * 0.3 * (1 if dx < 0 else -1)
        p3 = p0 + np.float32([dx, -rise])
        c1 = p0 + np.float32([dx * 0.30, droop])
        c2 = p0 + np.float32([dx * 0.70, droop * 0.8])
        pts = _bezier([p0, c1, c2, p3], 22)
        pts[:, 1] += np.sin(np.linspace(0, np.pi * 3, 22)) * 2.0
        ss_ = np.linspace(0, 1, len(pts))
        tb = t0 + 0.6 + 0.12 * k
        wd = w0 * (1.0 - ss_) ** 0.8 + 1.1
        strokes.append(dict(pts=pts, widths=wd, t0=tb, t1=tb + 0.55, dens=0.93, dry=0.45,
                            seed=seed + 10 + k))
        tips.append((pts[len(pts) // 3:], tb + 0.35))
        for j in range(2):
            q = pts[int(len(pts) * rng.uniform(0.4, 0.75))]
            e = q + np.float32([rng.uniform(18, 50) * np.sign(dx), rng.uniform(-40, -8)])
            tp = _bezier([q, (q + e) / 2 + np.float32([0, 7]), e], 8)
            strokes.append(dict(pts=tp, widths=np.linspace(2.2, 0.9, 8), t0=tb + 0.3, t1=tb + 0.55,
                                dens=0.9, dry=0.3, seed=seed + 40 + 2 * k + j))
            tips.append((tp, tb + 0.5))
    sd, st = render_strokes(h, w, strokes)
    return sd, st, tips
