"""Layered ink mountains (远山/中山/近石) with radiating hemp-fibre texture
(披麻皴), moss dots (苔点), distant trees and mist at their feet."""
import numpy as np
import cv2
from ink import fbm, noise1d, smoothstep, blur, render_strokes


def _peak_profile(x, xc, hgt, wid, p=2.0, skew=0.0):
    u = (x - xc) / wid
    u = np.where(u < 0, u * (1.0 + skew), u * (1.0 - skew))
    return hgt * np.exp(-np.abs(u) ** p)


def paint_range(h, w, base_y, peaks, seed, *, dens=0.8, fade=0.5, edge=1.2,
                tex=0.25, line=0.25, moss=0, mist_h=90.0, blur_s=0.0,
                fold=0.0, shade=0.22, ridge_amp=0.035, streak_w=12.0, dry=0.0,
                strokes=0, stroke_w=(1.2, 3.0), stroke_len=(0.10, 0.32), stroke_d=(0.25, 0.55)):
    """Paint a mountain range into a layer of size (h, w).

    peaks: list of (xc, height, width, p, skew) in layer x coords, drawn back
    to front. Returns (density, alpha, ridges).
    """
    rng = np.random.default_rng(seed)
    x = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)[:, None]
    dens_map = np.zeros((h, w), np.float32)
    alpha = np.zeros((h, w), np.float32)
    streak = fbm(h, w, streak_w * 6, 5, seed=seed + 1, aspect=0.2)
    ridged = 1.0 - np.abs(fbm(h, w, 130, 4, seed=seed + 2, aspect=0.16))
    blotch = fbm(h, w, 160, 4, seed=seed + 3)
    fine = fbm(h, w, 12, 3, seed=seed + 4)
    warpn = fbm(h, w, 90, 3, seed=seed + 8)
    dryn = fbm(h, w, 5, 2, seed=seed + 9, aspect=0.35)
    ridge_noise = noise1d(w, 150, 6, 0.6, seed=seed + 5)
    grid_x = np.tile(x[None, :], (h, 1))
    ridges = []
    for k, (xc, hgt, wid, p, skew) in enumerate(peaks):
        prof = _peak_profile(x, xc, hgt, wid, p, skew)
        prof = prof + ridge_noise * (ridge_amp * hgt) * np.clip(prof / (0.2 * hgt), 0, 1)
        ridge = base_y - prof
        dist = yy - ridge[None, :]
        sil = smoothstep(-edge, edge, dist + 0.8 * edge * fine)
        sil = sil * (prof > 2.0)[None, :]
        L = fade * max(hgt, 40.0)
        ph = np.maximum(prof, 1.0)[None, :]
        dpos = np.maximum(dist, 0)
        # texture coordinates bent so strokes radiate from the summit
        u = xc + (grid_x - xc) / (1.0 + dpos / (0.9 * ph + 30.0)) + 4.0 * warpn + k * 131.0
        mapx = np.mod(u, w - 1).astype(np.float32)
        mapy = np.clip(np.broadcast_to(dist, (h, w)) + k * 57.0, 0, h - 1).astype(np.float32)
        st = cv2.remap(streak, mapx, mapy, cv2.INTER_LINEAR)
        rd = cv2.remap(ridged, mapx, mapy, cv2.INTER_LINEAR)
        fall = np.exp(-dpos / L)
        side = np.tanh((grid_x - xc) / (0.45 * wid)) * (1 if k % 2 == 0 else -1)
        d = dens * fall
        d *= 1.0 + tex * st + 0.20 * blotch + shade * side
        d += fold * dens * fall ** 0.5 * smoothstep(0.80, 0.97, rd) * 0.5
        d += line * np.exp(-dpos / 3.2) * (0.55 + 0.45 * np.clip(fine, -1, 1))
        if dry > 0:   # dry-brush gaps (飞白) in the upper strokes
            d *= 1.0 - dry * smoothstep(0.2, 0.9, dryn) * fall
        if strokes:
            sts = []
            side_sign = 1 if k % 2 == 0 else -1
            for j in range(strokes):
                x0 = rng.uniform(xc - 1.15 * wid, xc + 1.15 * wid)
                ix = int(np.clip(x0, 0, w - 1))
                pr = prof[ix]
                if pr < 0.22 * hgt:
                    continue
                y0 = ridge[ix] + rng.uniform(1.0, 0.45 * pr) ** 1.0 * rng.uniform(0.2, 1.0)
                rel = (x0 - xc) / wid
                dx, dy = 0.85 * rel + rng.normal(0, 0.18), 1.0
                nrm = np.hypot(dx, dy)
                L = rng.uniform(*stroke_len) * hgt
                p0 = np.float32([x0, y0])
                p1 = p0 + np.float32([dx, dy]) / nrm * L
                mid = (p0 + p1) / 2 + np.float32([-dy, dx]) / nrm * L * rng.uniform(-0.12, 0.12)
                tt = np.linspace(0, 1, 10, dtype=np.float32)[:, None]
                pts = (1 - tt) ** 2 * p0 + 2 * (1 - tt) * tt * mid + tt ** 2 * p1
                wmax = rng.uniform(*stroke_w)
                wd = wmax * np.sin(np.linspace(0.25, np.pi - 0.35, 10)) ** 0.6
                shade_k = 1.0 + 0.35 * np.tanh(rel * 2.0) * side_sign
                sd = rng.uniform(*stroke_d) * shade_k * np.exp(-(y0 - ridge[ix]) / (2.2 * L + 1))
                sts.append(dict(pts=pts, widths=wd, dens=float(np.clip(sd, 0.05, 0.95)), dry=0.6,
                                seed=int(rng.integers(1 << 30))))
            if sts:
                sdm, _ = render_strokes(h, w, sts)
                d = 1.0 - (1.0 - d) * (1.0 - sdm * smoothstep(0.0, 2.0, dist))
        d = np.clip(d, 0, 1)
        dens_map = dens_map * (1.0 - sil) + d * sil
        alpha = np.maximum(alpha, sil)
        if moss:
            dot = np.zeros((h, w), np.float32)
            for _ in range(moss):
                px = rng.uniform(xc - 1.2 * wid, xc + 1.2 * wid)
                ix = int(np.clip(px, 0, w - 1))
                if prof[ix] < 0.3 * hgt:
                    continue
                py = ridge[ix] + rng.uniform(0.5, 12) ** 1.1
                rx, ry = rng.uniform(2.0, 5.0), rng.uniform(1.2, 2.4)
                cv2.ellipse(dot, (int(px * 4), int(py * 4)), (int(rx * 4), int(ry * 4)),
                            rng.uniform(-20, 20), 0, 360, float(rng.uniform(0.7, 0.95)), -1, cv2.LINE_AA, 2)
            dens_map = np.maximum(dens_map, blur(dot, 0.6))
        ridges.append(ridge)
    # mist at the feet: the silhouette dissolves into paper well above the layer edge
    mist = smoothstep(base_y - mist_h, base_y - 8.0, yy + 22.0 * blotch)
    keep = 1.0 - mist
    alpha = alpha * keep
    dens_map = dens_map * keep
    if blur_s > 0:
        dens_map = blur(dens_map, blur_s)
        alpha = blur(alpha, blur_s)
    return dens_map.astype(np.float32), alpha.astype(np.float32), ridges


def draw_far_trees(dens, x, y, size, rng, n=3):
    """A cluster of tiny distant trees (远树): trunks with stacked needle dabs."""
    h, w = dens.shape
    ss = 4
    pad = int(size * 2.2)
    x0, y0 = int(x - pad), int(y - size * 1.3)
    sw, sh = 2 * pad, int(size * 1.6)
    if x0 < 0 or y0 < 0 or x0 + sw >= w or y0 + sh >= h:
        return
    spr = np.zeros((sh * ss, sw * ss), np.float32)
    for i in range(n):
        tx = (x - x0 + rng.uniform(-0.9, 0.9) * size) * ss
        th = size * rng.uniform(0.6, 1.05)
        by = (y - y0 + rng.uniform(0, 0.15) * size) * ss
        ty = by - th * ss
        lean = rng.uniform(-0.12, 0.12) * th * ss
        cv2.line(spr, (int(tx), int(by)), (int(tx + lean), int(ty)), 0.9, max(1, int(0.07 * size * ss)), cv2.LINE_AA)
        levels = rng.integers(3, 5)
        for j in range(levels):
            f = 0.35 + 0.65 * j / (levels - 1)
            cy = by - f * th * ss
            cx = tx + lean * f
            half = (1.0 - 0.6 * f) * size * 0.42 * ss
            cv2.ellipse(spr, (int(cx), int(cy)), (int(half), max(1, int(half * 0.28))),
                        rng.uniform(-8, 8), 0, 360, float(rng.uniform(0.75, 0.95)), -1, cv2.LINE_AA)
    spr = cv2.resize(spr, (sw, sh), interpolation=cv2.INTER_AREA)
    spr = blur(spr, 0.45)
    sub = dens[y0:y0 + sh, x0:x0 + sw]
    np.maximum(sub, spr, out=sub)
