"""月满中秋 — ink-wash Mid-Autumn greeting video.

    python3 music.py soundtrack.wav            # the score (about 25 s)
    python3 render.py preview 1.0 5.0 12.0     # stills into ./preview/
    python3 render.py video                    # 1080p30 render + mux -> ./output/mid_autumn_ink.mp4
"""
import os
import sys
import time
import subprocess
import multiprocessing as mp
import numpy as np
import cv2

from ink import (W, H, PAPER, INK, VERMILION, make_paper_texture, make_vignette, smoothstep, sstep, fbm)
from world import Camera, Layer, Mist, build_landscape, build_water, rgba_from, HORIZON, warp
from backdrop import Backdrop, SKY_DEPTH
from foreground import build_near
from elements import Boat, MoonPath, Geese, Lanterns, Petals
from osmanthus import Osmanthus
from callig import Calligraphy, Seal, font_path

FPS = 30
DURATION = 43.0
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ timeline
T_POEM1, T_OUT1 = 4.8, 9.5
T_POEM2, T_OUT2 = 11.4, 18.0
T_BRANCH = 19.4
T_POEM3, T_OUT3 = 21.2, 26.9
T_LANTERN = 27.4
T_POEM4, T_OUT4 = 29.0, 35.0
T_TITLE = 36.1
T_SEAL = 38.75
T_END_FADE = 42.0

MOON_WORLD = (600.0, 332.0)

# camera keyframes: (t, x, y, dz)  -- Catmull-Rom through them
CAM_KEYS = [
    (-4.0, 0, 0, -0.02),
    (0.0, 0, 0, 0.0),
    (9.8, -40, 0, 0.05),
    (19.2, -120, 6, 0.085),
    (24.0, -130, -50, 0.13),
    (28.4, -110, -40, 0.12),
    (35.8, -20, 0, 0.02),
    (43.0, 0, 0, 0.045),
    (48.0, 0, 0, 0.05),
]


def camera_at(t):
    """Cubic Hermite through CAM_KEYS with time-scaled Catmull-Rom tangents (C1 in real time)."""
    keys = CAM_KEYS
    i = 1
    while i < len(keys) - 2 and t >= keys[i + 1][0]:
        i += 1
    k0, k1, k2, k3 = keys[i - 1], keys[i], keys[i + 1], keys[i + 2]
    dt = k2[0] - k1[0]
    u = min(max((t - k1[0]) / dt, 0.0), 1.0)
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    vals = []
    for j in (1, 2, 3):
        m1 = (k2[j] - k0[j]) / (k2[0] - k0[0])
        m2 = (k3[j] - k1[j]) / (k3[0] - k1[0])
        vals.append(h00 * k1[j] + h10 * dt * m1 + h01 * k2[j] + h11 * dt * m2)
    return Camera(*vals)


# ------------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        t = time.time()
        self.tex = make_paper_texture()
        self.vig = make_vignette()
        self.backdrop = Backdrop()
        self.far, self.midfar, self.mid = build_landscape()
        self.water = build_water()
        self.near = build_near()
        self.mists = [Mist(690, 70, 5.0, 9.0, 0.85, 501), Mist(560, 90, 3.6, -6.0, 0.5, 502),
                      Mist(772, 60, 2.2, 12.0, 0.65, 503)]
        # water reveal: the wash spreads from the horizon toward the viewer
        wl = self.water
        hh, ww = wl.rgba.shape[:2]
        yy = np.arange(hh, dtype=np.float32)[:, None] + wl.origin[1]
        n = fbm(hh, ww, 140, 4, seed=909)
        wl.ramp = 0.8
        wl.set_T((3.4 + 2.4 * np.clip((yy - HORIZON) / 380.0, 0, 1) + 0.35 * n).astype(np.float32))
        self.boat = Boat()
        self.moonpath = MoonPath()
        self.geese = Geese()
        self.lanterns = Lanterns(t0=T_LANTERN)
        self.petals = Petals(t0=T_BRANCH + 3.3)
        self.osm = Osmanthus(t0=T_BRANCH)
        self.grain = [np.random.default_rng(900 + i).standard_normal((H, W)).astype(np.float32) for i in range(6)]
        self._build_text()
        print(f"scene built in {time.time() - t:.1f}s", flush=True)

    def _build_text(self):
        MS = font_path("MaShanZheng")
        WK = font_path("LXGWWenKaiTC")
        x_r, y_t, sz = 1640, 150, 84
        self.texts = []

        def poem(cols, t0, t_out, author, seal_chars, seed, size=sz, y_top=y_t, x_right=x_r):
            p = Calligraphy(cols, MS, size, x_right, y_top, t0, t_out=t_out, seed=seed,
                            char_gap=0.22, col_pause=0.28)
            ax = x_right - size * 1.25 * len(cols) - 8
            a_len = len(author.replace(" ", ""))
            ay = y_top + size * 1.1 * 5 - a_len * 34 - 30
            a = Calligraphy([author], WK, 30, ax, ay, p.t_end_write + 0.1, dens=0.72,
                            char_gap=0.07, char_dur=0.25, t_out=t_out + 0.5, seed=seed + 50)
            s = Seal(seal_chars, 40, ax, ay + a_len * 33 + 38, p.t_end_write + 0.25 + 0.07 * a_len, seed=seed)
            s.t_out = t_out + 0.8
            self.texts.append((p, a, s))

        poem(["海上生明月", "天涯共此时"], T_POEM1, T_OUT1, "唐 张九龄", "望月", 11)
        poem(["明月几时有", "把酒问青天"], T_POEM2, T_OUT2, "宋 苏轼", "问天", 23)
        poem(["桂子月中落", "天香云外飘"], T_POEM3, T_OUT3, "唐 宋之问", "天香", 37)
        poem(["但愿人长久", "千里共婵娟"], T_POEM4, T_OUT4, "宋 苏轼", "婵娟", 53, size=90, y_top=140)
        # finale
        self.title = Calligraphy(["中秋"], MS, 232, 1575, 170, T_TITLE, char_dur=0.75, char_gap=0.62,
                                 dens=0.95, seed=71)
        self.greet = Calligraphy(["花好月圆 阖家安康"], MS, 54, 1352, 222, T_TITLE + 1.45, char_gap=0.14,
                                 char_dur=0.3, dens=0.9, seed=81)
        self.inscr = Calligraphy(["丙午仲秋"], WK, 28, 1262, 548, T_SEAL - 0.55, char_gap=0.08,
                                 char_dur=0.22, dens=0.7, seed=91)
        self.seal = Seal("人月團圓", 88, 1262, 752, T_SEAL, seed=7)

    def night_map(self, mx, my):
        if getattr(self, "_night", None) is None:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            self._yy, self._xx = yy, xx
            base = 0.12 + 0.16 * (1.0 - yy / H) ** 1.5
            # keep the inscription area on the right a little lighter
            base *= 1.0 - 0.35 * smoothstep(1150, 1500, xx) * smoothstep(900, 150, yy)
            self._night = base.astype(np.float32)
        d = np.sqrt((self._xx - mx) ** 2 + (self._yy - my) ** 2)
        return self._night * smoothstep(150.0, 330.0, d)

    # --------------------------------------------------------------- frame
    def render(self, t):
        cam = camera_at(t)
        mx, my = MOON_WORLD[0], MOON_WORLD[1] - 26.0 * sstep(0.0, 40.0, t)
        glow_boost = 1.0 + 0.45 * sstep(T_LANTERN, T_LANTERN + 6.0, t)
        f = self.backdrop.render(t, cam, (mx, my), glow_boost)

        # geese cross the moon, high in the sky
        if 10.8 < t < 19.5:
            u = (t - 10.8) / 8.7
            gx = 1500 - 1750 * u
            gy = 215 + 70 * u - 30 * np.sin(u * np.pi) * 0.5
            sx, sy, _ = cam.to_screen(9.0, gx, gy)
            self.geese.draw(f, t, sx, sy, 0.92, alpha=sstep(10.8, 11.4, t) * (1 - sstep(18.8, 19.5, t)))

        mist_k = sstep(1.8, 5.0, t)
        self.far.draw(f, t, cam)
        self.mists[0].draw(f, t, cam, mist_k)
        self.midfar.draw(f, t, cam)
        self.water.draw(f, t, cam)

        # moonlight on the water
        msx, msy, _ = cam.to_screen(SKY_DEPTH, mx, my)
        hx, hy, _ = cam.to_screen(3.2, 0, HORIZON + 6)
        self.moonpath.draw(f, t, msx, hy, hy + 300, strength=sstep(4.5, 7.5, t) * (1 + 0.3 * sstep(T_LANTERN, T_LANTERN + 6, t)))

        # the boat drifts leftward across the lake
        bx, by = 1210 - 8.5 * t, 812 + 1.6 * np.sin(t * 1.25)
        sx, sy, s = cam.to_screen(2.9, bx, by)
        self.boat.draw(f, t, sx, sy, s * 0.92, alpha=sstep(6.4, 7.6, t), ripple_t=t)

        self.mists[1].draw(f, t, cam, mist_k)
        self.mid.draw(f, t, cam)
        self.mists[2].draw(f, t, cam, mist_k * (1.0 + 0.4 * sstep(T_TITLE - 1.0, T_TITLE + 2.0, t)))
        self.near.draw(f, t, cam)
        self.osm.draw(f, t, cam)

        # night deepens as the lanterns rise: cool darkening, the moon spared
        night = sstep(T_LANTERN - 1.8, T_LANTERN + 3.5, t)
        if night > 0.001:
            k = night * self.night_map(msx, msy)
            f *= 1.0 - k[..., None] * np.array([1.0, 0.94, 0.78], np.float32)
        title_k = 0.75 * sstep(T_TITLE - 1.0, T_TITLE + 0.5, t)
        avoid = [(-100, -100, 820, 420, 1.0, 0.72),            # behind the osmanthus branch
                 (1430, 140, 1720, 700, title_k, 1.01),        # the title
                 (1180, 180, 1400, 820, title_k * 0.8, 1.01)]  # greeting and seal
        self.lanterns.draw(f, t, -cam.x * 0.25, -cam.y * 0.25, depth_range=(0.0, 0.3), avoid=avoid)
        self.lanterns.draw(f, t, -cam.x * 0.5, -cam.y * 0.5, depth_range=(0.3, 0.72), avoid=avoid)
        self.lanterns.draw(f, t, -cam.x * 0.8, -cam.y * 0.8, depth_range=(0.72, 1.01), avoid=avoid)
        self.petals.draw(f, t, near=False)
        self.petals.draw(f, t, near=True)

        # inscriptions
        for p, a, s in self.texts:
            p.draw(f, t)
            a.draw(f, t)
            if t < s.t_out + 1.0:
                s.draw(f, t, fade=1.0 - sstep(s.t_out, s.t_out + 0.9, t))
        self.title.draw(f, t)
        self.greet.draw(f, t)
        self.inscr.draw(f, t)
        self.seal.draw(f, t)

        # paper, vignette, grain, fades
        f *= self.tex[..., None]
        f *= self.vig
        g = self.grain[int(t * FPS) % len(self.grain)]
        f += (g * 0.005)[..., None]
        fade = sstep(0.0, 0.6, t) * (1.0 - sstep(T_END_FADE, DURATION, t))
        if fade < 1.0:
            f *= fade
        return (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------- execution
SCENE = None


def _init():
    global SCENE
    if SCENE is None:
        SCENE = Scene()


def _render_chunk(args):
    idx, frames, path = args
    _init()
    ff = ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-tune", "film", "-x264-params", "aq-mode=3:aq-strength=0.9", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for k, fi in enumerate(frames):
        img = SCENE.render(fi / FPS)
        p.stdin.write(img.tobytes())
        if k % 30 == 0:
            print(f"[chunk {idx}] frame {fi} ({k + 1}/{len(frames)}) {time.time() - t0:.0f}s", flush=True)
    p.stdin.close()
    p.wait()
    return path


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def render_video(out_dir, workers=4, chunk_s=3.0):
    os.makedirs(out_dir, exist_ok=True)
    seg_dir = os.path.join(out_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    n = int(round(DURATION * FPS))
    step = int(chunk_s * FPS)
    jobs = []
    for i, a in enumerate(range(0, n, step)):
        jobs.append((i, list(range(a, min(a + step, n))), os.path.join(seg_dir, f"seg_{i:03d}.mp4")))
    _init()                         # build once, workers inherit it (fork)
    with mp.get_context("fork").Pool(workers) as pool:
        paths = pool.map(_render_chunk, jobs, chunksize=1)
    lst = os.path.join(seg_dir, "list.txt")
    with open(lst, "w") as fh:
        for pth in paths:
            fh.write(f"file '{os.path.abspath(pth)}'\n")
    silent = os.path.join(out_dir, "video_silent.mp4")
    subprocess.check_call([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                           "-i", lst, "-c", "copy", silent])
    return silent


def mux(video, wav, out):
    subprocess.check_call([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", video, "-i", wav,
                           "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart",
                           "-metadata", "title=月满中秋", "-shortest", out])


def preview(times, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _init()
    for t in times:
        t0 = time.time()
        img = SCENE.render(t)
        cv2.imwrite(os.path.join(out_dir, f"t{t:05.2f}.png"), img[..., ::-1])
        print(f"t={t:.2f} rendered in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "video"
    if mode == "preview":
        out = os.environ.get("PREVIEW_DIR", os.path.join(HERE, "preview"))
        preview([float(a) for a in sys.argv[2:]], out)
    else:
        out = os.environ.get("OUT_DIR", os.path.join(HERE, "output"))
        silent = render_video(out, workers=int(os.environ.get("WORKERS", "4")))
        print(silent)
        wav = os.path.join(HERE, "soundtrack.wav")
        if os.path.exists(wav):
            final = os.path.join(out, "mid_autumn_ink.mp4")
            mux(silent, wav, final)
            print(final)
