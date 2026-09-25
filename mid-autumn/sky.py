"""Night sky painted with 烘云托月: ink blooms spread across the paper and the
moon is the untouched paper left in the middle."""
import numpy as np
import cv2
from ink import (W, H, GOLD, fbm, smoothstep, sstep, blur)


class Sky:
    """Global sky wash that spreads in from several ink blooms (static T-map)."""

    MARGIN = 90

    def __init__(self, seed=3):
        m = self.MARGIN
        self.h, self.w = H + 2 * m, W + 2 * m
        h, w = self.h, self.w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy -= m
        xx -= m
        self.origin = (-m, -m)            # world coords of pixel (0, 0)

        # ---- target density of the finished wash: dark zenith, pale horizon
        top = 1.0 - smoothstep(-160.0, 700.0, yy)
        F = 0.76 * top ** 1.3
        F *= 1.0 - 0.42 * smoothstep(900.0, 1900.0, xx) * smoothstep(-200.0, 500.0, yy + 0 * xx) ** 0.5
        n1 = fbm(h, w, 420, 5, seed=seed)
        n2 = fbm(h, w, 200, 5, seed=seed + 1, aspect=4.0)    # cloud streaks
        n3 = fbm(h, w, 60, 4, seed=seed + 2)
        F *= 1.0 + 0.14 * n1 + 0.05 * n3
        # moonlit cloud bands: paler streaks with a slightly darker underside
        cloud = smoothstep(0.25, 1.5, n2)
        F *= 1.0 - 0.42 * cloud
        under = np.roll(cloud, -14, axis=0) - cloud
        F += 0.05 * np.clip(under, 0, 1) * top
        self.F = np.clip(F, 0, 1).astype(np.float32)

        # ---- arrival time of the spreading ink: union of blooms
        blooms = [  # (x, y, t_start, Rmax, tau)
            (560, 170, 1.45, 950, 1.5),
            (1080, 80, 1.9, 1050, 1.8),
            (180, 60, 2.1, 820, 1.6),
            (1580, 140, 2.5, 950, 1.8),
            (1880, 420, 3.0, 700, 1.6),
            (-20, 420, 2.9, 700, 1.6),
            (900, 430, 3.4, 760, 1.9),
        ]
        nwarp = fbm(h, w, 260, 6, seed=seed + 5)
        nfine = fbm(h, w, 24, 3, seed=seed + 6)
        T = np.full((h, w), np.inf, np.float32)
        for bx, by, t0, Rm, tau in blooms:
            d = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
            deff = d * (1.0 + 0.28 * nwarp) + 14.0 * nfine
            q = np.clip(np.maximum(deff, 0) / Rm, 0, 0.9995)
            np.minimum(T, t0 + tau * (-np.log1p(-q)), out=T)
        self.T = T
        self.fn = (0.5 + 0.5 * np.clip(nfine, -1, 1)).astype(np.float32)
        self._final = None

    def density(self, t):
        """Ink density of the sky wash at time t (layer resolution)."""
        if t > 9.0:
            if self._final is None:
                self._final = self._density(1e3)
            return self._final
        return self._density(t)

    def _density(self, t):
        dt = t - self.T
        a = self.F * smoothstep(0.0, 0.7, dt)
        # wet front: pigment gathers where the spreading edge currently is
        front = np.exp(-(dt / 0.25) ** 2) * (dt > -0.06) * (0.05 + 0.08 * self.fn)
        a += front * smoothstep(0.08, 0.35, self.F)
        return np.clip(a, 0, 1)


class Moon:
    """The moon, its ink halo and the ink-drop birth, in a moon-centred ROI."""

    def __init__(self, R=132, seed=21, t_hit=0.9):
        self.R = R
        self.t_hit = t_hit
        self.half = half = 640
        n = 2 * half
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - half
        d = np.sqrt(xx * xx + yy * yy)
        self.d = d
        self.yy = yy
        nw = fbm(n, n, 150, 6, seed=seed)
        nf = fbm(n, n, 22, 3, seed=seed + 1)
        self.nw, self.nf = nw, nf
        out = np.maximum(d - R, 0)

        # halo arrival map: ink pushed out of the moon then spreading outward
        deff = out * (1.0 + 0.34 * nw) + 10 * nf
        q = np.clip(deff / 600.0, 0, 0.9995)
        self.Th = t_hit + 0.55 + 1.2 * (-np.log1p(-q))
        halo = 0.28 * np.exp(-out / 210.0)
        halo += 0.07 * np.exp(-out / 22.0)                       # crisp reserve rim
        halo *= 1.0 + 0.22 * nw + 0.06 * nf
        win = smoothstep(half - 8.0, half - 260.0, d)
        self.win = win
        halo *= win
        self.halo = np.clip(halo * smoothstep(R - 1.5, R + 1.5, d), 0, 1).astype(np.float32)

        # the splash: an ink spot that swells to the moon size
        self.spot_r = R * (1.02 + 0.07 * nw + 0.03 * nf)
        self.spot_tex = np.clip(0.80 + 0.10 * nw + 0.05 * nf, 0, 1)

        # splatter droplets flung out on impact
        rng = np.random.default_rng(seed + 7)
        drops = np.zeros((n, n), np.float32)
        for _ in range(28):
            a0 = rng.uniform(0, 2 * np.pi)
            rr = R * rng.uniform(1.18, 2.7)
            cx, cy = half + np.cos(a0) * rr, half + np.sin(a0) * rr
            s = rng.uniform(1.5, 6.5) * (1.45 - rr / (2.7 * R))
            el = rng.uniform(1.0, 2.8)
            cv2.ellipse(drops, (int(cx), int(cy)), (max(1, int(s * el)), max(1, int(s))),
                        np.degrees(a0), 0, 360, 1.0, -1, cv2.LINE_AA)
        self.drops = blur(drops, 1.0) * 0.85

        # moon surface: luminous warm paper with very faint maria
        mar = fbm(n, n, 60, 5, seed=seed + 3)
        disc = smoothstep(R + 0.9, R - 0.9, d)
        self.disc = disc.astype(np.float32)
        maria = np.clip(smoothstep(0.5, 1.8, mar) * 0.035, 0, 1) * smoothstep(R, R * 0.6, d)
        rr = np.clip(d / R, 0, 1)
        col = np.stack([
            np.full_like(d, 1.0),
            0.985 - 0.035 * rr ** 2,
            0.935 - 0.10 * rr ** 2,
        ], -1)
        col *= (1.0 - maria)[..., None]
        self.surface = np.clip(col, 0, 1).astype(np.float32)

        # glow sprite (screen blend): tight corona + broad moonlit haze
        g = 0.42 * np.exp(-out / 38.0) + 0.24 * np.exp(-out / 150.0) + 0.10 * np.exp(-out / 420.0)
        g *= smoothstep(R - 5, R - 0.5, d) * win
        self.glow = (g[..., None] * GOLD[None, None, :]).astype(np.float32)

    def render(self, t):
        """Return (ink_density, surface_alpha, glow_strength) for the ROI at time t."""
        tau = t - self.t_hit
        d, R = self.d, self.R
        ink = np.zeros_like(d)
        # 0. the falling drop
        if -0.28 < tau < 0.0:
            p = (tau + 0.28) / 0.28
            cy = -420.0 * (1.0 - p * p)
            dy = (self.yy - cy) / 20.0
            dx = np.sqrt(np.maximum(d * d - self.yy * self.yy, 0)) / 7.5
            ink = np.clip(1.2 - np.sqrt(dx * dx + dy * dy), 0, 1) * 0.9
            return ink, np.zeros_like(d), 0.0
        if tau < 0:
            return ink, np.zeros_like(d), 0.0
        # 1. splash spot swelling to the moon radius, darker at its edge
        rs = self.spot_r * (0.22 + 0.80 * (1 - np.exp(-tau / 0.12)))
        spot = smoothstep(rs + 2.0, rs - 2.0, d) * self.spot_tex
        spot += np.exp(-((d - rs + 4) / 5.0) ** 2) * 0.12
        # 2. the moon opens from the centre, pushing ink to the rim
        clr_r = R * 1.12 * sstep(0.28, 1.05, tau) * (1.0 + 0.05 * self.nw * (1 - sstep(0.85, 1.2, tau)))
        clr_r = np.maximum(clr_r, 0.0)
        opened = smoothstep(clr_r + 2.0, clr_r - 8.0, d) if tau > 0.28 else np.zeros_like(d)
        spot *= 1.0 - opened
        # ink left outside the rim dilutes into the halo
        spot *= 1.0 - sstep(0.6, 1.5, tau) * smoothstep(R - 2, R + 3, d)
        ring_r = np.minimum(clr_r, R) + 3.0
        ring = np.exp(-((d - ring_r) / 6.0) ** 2) * 0.35 * sstep(0.3, 0.55, tau) * (1 - sstep(0.75, 1.35, tau))
        # 3. halo spreading outward from the rim
        dt = t - self.Th
        halo = self.halo * smoothstep(0.0, 0.7, dt)
        front = np.exp(-(dt / 0.25) ** 2) * (dt > -0.06) * 0.08 * smoothstep(R, R + 20, d)
        drops = self.drops * sstep(0.02, 0.09, tau) * (1.0 - 0.9 * sstep(1.2, 4.0, tau))
        wet = np.clip(spot + ring * smoothstep(R - 4, R + 4, d), 0, 1)
        ink = 1.0 - (1.0 - wet) * (1.0 - np.clip(halo + front, 0, 1)) * (1.0 - drops)
        surf_a = self.disc * opened
        glow_k = sstep(0.7, 3.0, tau)
        return np.clip(ink, 0, 1), surf_a, glow_k
