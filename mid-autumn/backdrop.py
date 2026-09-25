"""Sky + moon composed into the opaque backdrop layer."""
import numpy as np
from ink import PAPER, INK_SKY, over_ink
from sky import Sky, Moon
from world import warp

SKY_DEPTH = 22.0


class Backdrop:
    def __init__(self):
        self.sky = Sky()
        self.moon = Moon()
        self._moon_cache = None

    def moon_state(self, t):
        if t - self.moon.t_hit > 5.5:
            if self._moon_cache is None:
                self._moon_cache = self.moon.render(self.moon.t_hit + 6.0)
            return self._moon_cache
        return self.moon.render(t)

    def render(self, t, cam, moon_xy, glow_boost=1.0):
        sky, moon = self.sky, self.moon
        base = np.empty((sky.h, sky.w, 3), np.float32)
        base[:] = PAPER
        over_ink(base, sky.density(t), INK_SKY)
        ink, surf_a, glow_k = self.moon_state(t)
        hf = moon.half
        # moon ROI placement in layer pixels
        mx = int(round(moon_xy[0] - sky.origin[0])) - hf
        my = int(round(moon_xy[1] - sky.origin[1])) - hf
        X0, Y0 = max(mx, 0), max(my, 0)
        X1, Y1 = min(mx + 2 * hf, sky.w), min(my + 2 * hf, sky.h)
        sl = (slice(Y0 - my, Y1 - my), slice(X0 - mx, X1 - mx))
        sub = base[Y0:Y1, X0:X1]
        over_ink(sub, ink[sl], INK_SKY)
        sa = surf_a[sl][..., None]
        sub *= (1.0 - sa)
        sub += moon.surface[sl] * sa
        g = moon.glow[sl] * (glow_k * glow_boost)
        sub[...] = 1.0 - (1.0 - sub) * (1.0 - np.clip(g, 0, 1))
        M, _ = cam.matrix(SKY_DEPTH, sky.origin)
        return warp(base, M)
