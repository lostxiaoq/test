"""古风配乐 — synthesized score for the Mid-Autumn video.

D major pentatonic (the guzheng's native tuning).  Instruments are all
synthesized here: guzheng (Karplus-Strong string + wooden body), bamboo flute
(additive, breath noise, vibrato, grace notes), a soft string pad, 堂鼓 drum,
bronze bells, a gong, wind chimes and the ink drop.  Cues line up with the
video timeline in render.py.

    python3 music.py out.wav
"""
import sys
import wave
import numpy as np
from scipy import signal

FS = 44100
DUR = 43.0
N = int(DUR * FS)

_OFF = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi(name):
    return 12 * (int(name[-1]) + 1) + _OFF[name[:-1]]


def hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


def pan_gains(p):
    a = (p + 1.0) * np.pi / 4.0
    return np.cos(a), np.sin(a)


class Bus:
    def __init__(self):
        self.L = np.zeros(N + FS * 8)
        self.R = np.zeros(N + FS * 8)

    def add(self, t, x, gain=1.0, pan=0.0):
        s = int(round(t * FS))
        if s < 0:
            x = x[-s:]
            s = 0
        e = min(s + len(x), len(self.L))
        gl, gr = pan_gains(pan)
        self.L[s:e] += x[:e - s] * gain * gl
        self.R[s:e] += x[:e - s] * gain * gr

    def add_st(self, t, l, r, gain=1.0):
        s = int(round(t * FS))
        e = min(s + len(l), len(self.L))
        self.L[s:e] += l[:e - s] * gain
        self.R[s:e] += r[:e - s] * gain


# ------------------------------------------------------------------ guzheng
def _bend(y, cents):
    rate = 2.0 ** (cents / 1200.0)
    ph = np.cumsum(rate) - rate[0]
    ph = ph[ph < len(y) - 1]
    return np.interp(ph, np.arange(len(y)), y)


def zheng(m, dur=4.0, vel=0.7, orn=None, seed=0):
    """One plucked guzheng string via Karplus-Strong (IIR comb through lfilter)."""
    rng = np.random.default_rng(seed)
    f = float(hz(m))
    n = int(dur * FS)
    P = FS / f
    t60 = float(np.interp(np.log2(f), [np.log2(60), np.log2(1400)], [11.0, 4.0]))
    bright = 0.62 + 0.3 * vel
    p = 0.25 * (1.0 - bright)
    # loop delay = Nd + 1 (smoothing FIR) + d (first-order Thiran allpass, lossless)
    Nd = int(np.floor(P - 1.5))
    d = (P - 1.0) - Nd                      # in [0.5, 1.5)
    eta = (1.0 - d) / (1.0 + d)
    g = 10.0 ** (-3.0 * P / (t60 * FS))
    loop = g * np.convolve([p, 1.0 - 2.0 * p, p], [eta, 1.0])
    den = np.zeros(Nd + len(loop))
    den[0] = 1.0
    den[1] += eta
    den[Nd:] -= loop
    num = np.array([1.0, eta])
    L = max(int(P), 2)
    # initial displacement: a triangle peaked at the plucking point, plus a little
    # low-passed noise for the nail's grit; brighter when plucked harder
    pos = 0.13
    tri = np.interp(np.arange(L), [0, pos * (L - 1), L - 1], [0.0, 1.0, 0.0])
    grit = rng.uniform(-1, 1, L)
    fc = min(f * (5.0 + 22.0 * vel), 15000.0)
    bl, al = signal.butter(2, fc / (FS / 2))
    exc = tri - tri.mean() + 0.30 * signal.filtfilt(bl, al, grit, padlen=min(3 * max(len(al), len(bl)), L - 1))
    exc = signal.filtfilt(bl, al, np.tile(exc, 3), padlen=0)[L:2 * L]   # periodic smoothing
    exc -= exc.mean()
    exc /= np.abs(exc).max() + 1e-9
    x = np.zeros(n)
    x[:L] = exc
    y = signal.lfilter(num, den, x)
    # fingernail pick click
    nc = int(0.005 * FS)
    click = rng.standard_normal(nc) * np.exp(-np.arange(nc) / (0.0007 * FS))
    click = signal.lfilter(*signal.butter(2, 2500 / (FS / 2), "high"), click)
    y[:nc] += 0.10 * vel * click
    tt = np.arange(n) / FS
    if orn == "vib":           # 揉弦: vibrato blooms after the attack
        cents = 16.0 * np.sin(2 * np.pi * 5.2 * tt) * np.clip((tt - 0.25) / 0.4, 0, 1)
        y = _bend(y, cents)
    elif orn == "slide":       # 按滑: slide up a whole tone into the note
        cents = -200.0 * (1.0 - np.clip(tt / 0.22, 0, 1)) ** 2
        cents += 12.0 * np.sin(2 * np.pi * 5.0 * tt) * np.clip((tt - 0.45) / 0.4, 0, 1)
        y = _bend(y, cents)
    elif orn == "dip":         # 回滑: press after pluck, then release
        cents = 100.0 * np.sin(np.pi * np.clip((tt - 0.12) / 0.5, 0, 1))
        y = _bend(y, cents)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    # damp the tail softly
    fade = int(0.08 * FS)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y * vel


def zheng_body_ir():
    t = np.arange(int(0.16 * FS)) / FS
    modes = [(112, 0.060, 1.0), (185, 0.050, 0.8), (262, 0.040, 0.7), (377, 0.032, 0.55),
             (530, 0.026, 0.45), (752, 0.020, 0.35), (1090, 0.014, 0.25), (1580, 0.010, 0.18),
             (2350, 0.007, 0.12)]
    ir = np.zeros_like(t)
    for f, d, a in modes:
        ir += a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
    ir /= np.sqrt((ir ** 2).sum())
    return ir


def harmonic_tone(m, dur=3.0, vel=0.4):
    """泛音: the glassy harmonic of a zheng string (pure, bell-like)."""
    f = float(hz(m))
    t = np.arange(int(dur * FS)) / FS
    y = (np.sin(2 * np.pi * f * t) + 0.18 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.4))
    env = np.exp(-t / (dur * 0.35)) * np.clip(t / 0.004, 0, 1)
    return y * env * vel


# -------------------------------------------------------------------- flute
def flute(notes, seed=5):
    """notes: list of (t, dur, midi, vel, orn). Returns mono track of length N.

    Legato phrasing: pitch glides between touching notes, vibrato blooms on
    long notes, grace notes (倚音) and slides (滑音) are added as ornaments.
    """
    rng = np.random.default_rng(seed)
    pitch = np.full(N, np.nan)
    amp = np.zeros(N)
    cents = np.zeros(N)
    chiff = np.zeros(N)
    notes = sorted(notes)
    for i, (t, d, m, v, orn) in enumerate(notes):
        s, e = int(t * FS), int((t + d) * FS)
        e = min(e, N)
        L = e - s
        tt = np.arange(L) / FS
        pitch[s:e] = m
        nxt = notes[i + 1][0] if i + 1 < len(notes) else 1e9
        prv_end = notes[i - 1][0] + notes[i - 1][1] if i > 0 else -1e9
        legato_in = (t - prv_end) < 0.04
        legato_out = (nxt - (t + d)) < 0.04
        att = 0.035 if legato_in else 0.09
        rel = 0.02 if legato_out else 0.16
        env = np.clip(tt / att, 0, 1) ** 1.5
        if not legato_out:
            env *= np.clip((d - tt) / rel, 0, 1)
        if d > 0.7:                          # swell through long notes
            env *= 0.82 + 0.18 * np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 0.8
        if legato_in:                        # a slight tonguing dip
            env *= 1.0 - 0.12 * np.exp(-tt / 0.03)
        amp[s:e] = np.maximum(amp[s:e], env * v)
        if not legato_in:
            nch = min(int(0.05 * FS), L)
            chiff[s:s + nch] += np.exp(-np.arange(nch) / (0.012 * FS)) * v
        if d > 0.55:
            depth = 22.0 + 6.0 * rng.random()
            rate = 5.1 + 0.5 * rng.random()
            vib = depth * np.sin(2 * np.pi * rate * tt + rng.uniform(0, 6.28)) * np.clip((tt - 0.22) / 0.45, 0, 1)
            cents[s:e] += vib
        if orn == "grace":                   # upper neighbour flick
            gl = min(int(0.075 * FS), L)
            up = 2.0 if (m % 12) in (2, 4, 9) else 3.0
            pitch[s:s + gl] = m + up
        elif orn == "slide":
            sl = min(int(0.20 * FS), L)
            cents[s:s + sl] += -180.0 * (1.0 - np.linspace(0, 1, sl)) ** 2
    # fill rests with the last pitch so the oscillator never jumps while silent
    idx = np.where(~np.isnan(pitch), np.arange(N), 0)
    np.maximum.accumulate(idx, out=idx)
    pitch = pitch[idx]
    pitch[np.isnan(pitch)] = 69
    # portamento: smooth the step curve (~45 ms)
    k = int(0.045 * FS)
    win = np.hanning(2 * k + 1)
    win /= win.sum()
    pitch = np.convolve(np.pad(pitch, k, mode="edge"), win, mode="valid")
    amp = np.convolve(np.pad(amp, 200, mode="edge"), np.ones(401) / 401, mode="valid")
    drift = np.convolve(rng.standard_normal(N), np.ones(4410) / 4410, mode="same") * 40
    f0 = hz(pitch) * 2.0 ** ((cents + drift) / 1200.0)
    ph = 2 * np.pi * np.cumsum(f0) / FS
    base = [1.0, 0.36, 0.17, 0.09, 0.05, 0.028, 0.014]
    y = np.zeros(N)
    for k_, a in enumerate(base, start=1):
        bright = amp ** (0.35 * (k_ - 1))
        y += a * bright * np.sin(k_ * ph + 0.3 * k_)
    y *= amp
    # breath: band-passed noise riding the tone, plus chiff at attacks
    noise = rng.standard_normal(N)
    b1 = signal.lfilter(*signal.butter(2, [1200 / (FS / 2), 5200 / (FS / 2)], "band"), noise)
    b2 = signal.lfilter(*signal.butter(2, [300 / (FS / 2), 1200 / (FS / 2)], "band"), noise)
    y += (0.055 * b1 + 0.03 * b2) * amp ** 1.2
    y += 0.12 * b1 * chiff
    return y


# ---------------------------------------------------------------------- pad
def pad(chords):
    out = np.zeros(N + FS * 3)
    rng = np.random.default_rng(9)
    for t0, t1, notes, v in chords:
        s = int(t0 * FS)
        L = int((t1 - t0 + 1.8) * FS)
        t = np.arange(L) / FS
        att = 1.1
        env = np.sin(np.pi / 2 * np.clip(t / att, 0, 1)) ** 2
        env *= np.clip(1.0 - (t - (t1 - t0)) / 1.8, 0, 1) ** 1.5
        chord = np.zeros(L)
        for m in notes:
            m = midi(m)
            for det in (-6.0, 0.0, 6.5):
                f = float(hz(m)) * 2 ** (det / 1200.0)
                ph0 = rng.uniform(0, 6.28)
                for k in range(1, 7):
                    if f * k > 5000:
                        break
                    chord += np.sin(2 * np.pi * f * k * t + ph0 * k) / k ** 1.8
        chord *= env * v / (3.0 * len(notes)) * (1.0 + 0.08 * np.sin(2 * np.pi * 0.23 * t))
        out[s:s + L] += chord
    return out[:N]


# ------------------------------------------------------------- percussion
def drum(size=1.0, vel=1.0, seed=0):
    rng = np.random.default_rng(seed)
    L = int((1.4 + size) * FS)
    t = np.arange(L) / FS
    f = 46.0 + 80.0 * np.exp(-t / 0.045)
    ph = 2 * np.pi * np.cumsum(f) / FS
    body = np.sin(ph) * np.exp(-t / (0.30 + 0.35 * size))
    body += 0.35 * np.sin(1.58 * ph) * np.exp(-t / 0.12)
    body += 0.15 * np.sin(2.3 * ph) * np.exp(-t / 0.07)
    n = rng.standard_normal(L)
    slap = signal.lfilter(*signal.butter(2, [90 / (FS / 2), 1800 / (FS / 2)], "band"), n) * np.exp(-t / 0.025)
    return vel * (body + 0.35 * slap) * np.clip(t / 0.002, 0, 1)


def bell(freq, dur=4.0, vel=0.5, seed=0):
    """Bronze bell (编钟-like): inharmonic partials with individual decays."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    parts = [(1.0, 1.0, 1.0), (2.0, 0.55, 0.62), (2.76, 0.42, 0.5), (3.98, 0.26, 0.36),
             (5.35, 0.16, 0.25), (6.9, 0.08, 0.16)]
    y = np.zeros_like(t)
    for r, a, dk in parts:
        y += a * np.sin(2 * np.pi * freq * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (dur * 0.3 * dk))
    y *= np.clip(t / 0.002, 0, 1)
    # a little beating like a real casting
    y *= 1.0 + 0.06 * np.sin(2 * np.pi * 1.7 * t)
    return y * vel


def gong(f0=82.0, dur=5.0, vel=0.6, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    y = np.zeros_like(t)
    for r in [1.0, 1.47, 1.98, 2.52, 3.09, 3.71, 4.36, 5.12, 5.93, 6.8, 7.9]:
        a = 1.0 / r ** 0.9
        bloom = np.clip(t / (0.02 + 0.05 * r), 0, 1)
        dec = np.exp(-t / (dur * 0.45 / r ** 0.35))
        fr = f0 * r * (1.0 + 0.004 * rng.standard_normal())
        y += a * bloom * dec * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.28))
    n = rng.standard_normal(len(t))
    y += 0.25 * signal.lfilter(*signal.butter(2, 900 / (FS / 2)), n) * np.exp(-t / 0.05)
    return y * vel / 3.0


def chime(freq, vel=0.25, seed=0):
    t = np.arange(int(2.2 * FS)) / FS
    rng = np.random.default_rng(seed)
    y = np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.55)
    y += 0.35 * np.sin(2 * np.pi * freq * 2.93 * t + rng.uniform(0, 6)) * np.exp(-t / 0.2)
    y += 0.15 * np.sin(2 * np.pi * freq * 5.1 * t + rng.uniform(0, 6)) * np.exp(-t / 0.08)
    return y * np.clip(t / 0.001, 0, 1) * vel


def brush_swish(dur=0.55, seed=0):
    """The dry rasp of a big brush on rice paper."""
    rng = np.random.default_rng(seed)
    n = int(dur * FS)
    t = np.arange(n) / FS
    x = rng.standard_normal(n)
    x = signal.lfilter(*signal.butter(2, [700 / (FS / 2), 3800 / (FS / 2)], "band"), x)
    grain = 0.6 + 0.4 * np.abs(signal.lfilter(*signal.butter(1, 30 / (FS / 2)), rng.standard_normal(n))) * 6
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5 * np.exp(-t / (dur * 0.8))
    return x * env * np.clip(grain, 0, 1.5)


def ink_drop():
    t = np.arange(int(0.25 * FS)) / FS
    f = 380.0 * np.exp(np.clip(t, 0, 0.05) / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / FS
    return np.sin(ph) * np.exp(-t / 0.045) * np.clip(t / 0.001, 0, 1)


# ------------------------------------------------------------------- reverb
def reverb_ir(rt60=3.2, seconds=4.0, seed=12):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    irs = []
    for ch in range(2):
        noise = rng.standard_normal(n)
        lo = signal.lfilter(*signal.butter(2, 1800 / (FS / 2)), noise)
        hi = noise - lo
        ir = lo * np.exp(-6.9 * t / rt60) + 0.5 * hi * np.exp(-6.9 * t / (rt60 * 0.35))
        ir *= np.clip((t - 0.018) / 0.03, 0, 1)                     # pre-delay, soft onset
        for k in range(8):                                            # early reflections
            d = int(rng.uniform(0.008, 0.07) * FS)
            ir[d] += rng.uniform(0.2, 0.5) * (1 if rng.random() < 0.5 else -1)
        ir /= np.sqrt((ir ** 2).sum())
        irs.append(ir)
    return irs


# -------------------------------------------------------------------- score
BEAT = 60.0 / 72.0
T0 = 8.2


def B(bar, beat=0.0):
    return T0 + ((bar - 1) * 4 + beat) * BEAT


PENTA = ["D", "E", "F#", "A", "B"]


def penta_run(lo, hi):
    """All pentatonic notes from lo to hi inclusive (names)."""
    out = []
    for o in range(1, 8):
        for n in PENTA:
            m = midi(f"{n}{o}")
            if midi(lo) <= m <= midi(hi):
                out.append(f"{n}{o}")
    return out


def build():
    zb, fb, pb, db, bb = Bus(), Bus(), Bus(), Bus(), Bus()
    seed = [100]

    def Z(t, name, vel=0.5, dur=None, orn=None, gain=1.0):
        m = midi(name)
        d = dur or float(np.interp(m, [38, 90], [6.0, 2.4]))
        seed[0] += 1
        y = zheng(m, d, vel, orn, seed[0])
        pan = float(np.clip(-0.45 + 0.9 * (m - 38) / 50.0, -0.5, 0.5))
        zb.add(t, y, 0.21 * gain, pan)

    def gliss(t0, t1, lo, hi, v0, v1, curve=0.9, down=False):
        names = penta_run(lo, hi)
        if down:
            names = names[::-1]
        n = len(names)
        for i, nm in enumerate(names):
            u = i / (n - 1)
            tt = t0 + (t1 - t0) * (0.5 - 0.5 * np.cos(np.pi * u)) ** curve
            Z(tt, nm, v0 + (v1 - v0) * u, dur=2.5 if i < n - 1 else 4.0)

    # ---------------- intro (free time): the ink drop and the moon's birth
    bb.add(0.9, ink_drop(), 0.35, 0.0)
    Z(0.90, "D2", 0.95, dur=8.0)
    Z(0.92, "D3", 0.45, dur=6.0)
    bb.add(0.9, bell(float(hz(midi("D5"))), 5.0, 0.20, 1), 1.0, 0.25)
    gliss(1.05, 2.35, "D3", "D6", 0.30, 0.72)
    Z(2.80, "A4", 0.55, orn="vib")
    Z(3.40, "F#4", 0.5, orn="slide")
    Z(4.00, "E4", 0.45)
    Z(4.40, "D4", 0.5, orn="vib")
    bb.add(4.95, harmonic_tone(midi("A5"), 3.0, 0.22), 1.0, 0.3)
    for t, nm, v in [(5.25, "D2", 0.55), (5.65, "A2", 0.4), (6.05, "D3", 0.38), (6.45, "F#3", 0.36),
                     (6.85, "A3", 0.36), (7.3, "D4", 0.34), (7.7, "E4", 0.32)]:
        Z(t, nm, v)

    fl = []

    def F(t, d, name, v=0.6, orn=None):
        fl.append((t, d, midi(name), v, orn))

    F(4.95, 1.35, "A4", 0.50, "slide")
    F(6.30, 0.40, "B4", 0.52)
    F(6.70, 0.40, "D5", 0.55)
    F(7.10, 0.85, "E5", 0.58)
    F(7.95, 0.25, "D5", 0.52)

    # ---------------- in tempo (72 bpm)
    def Fb(bar, b0, b1, name, v=0.62, orn=None):
        F(B(bar, b0), (b1 - b0) * BEAT, name, v, orn)

    # flute melody
    Fb(1, 0, 2, "F#5", 0.64, "grace"); Fb(1, 2, 2.5, "E5", 0.58); Fb(1, 2.5, 3, "D5", 0.56); Fb(1, 3, 4, "E5", 0.6)
    Fb(2, 0, 1.5, "B4", 0.58); Fb(2, 1.5, 2, "A4", 0.52); Fb(2, 2, 3.8, "B4", 0.6, "slide")
    Fb(3, 0, 1, "D5", 0.6); Fb(3, 1, 1.5, "E5", 0.6); Fb(3, 1.5, 2, "F#5", 0.62)
    Fb(3, 2, 3.5, "A5", 0.68, "grace"); Fb(3, 3.5, 4, "F#5", 0.6)
    Fb(4, 0, 1, "E5", 0.6); Fb(4, 1, 1.5, "D5", 0.56); Fb(4, 1.5, 2, "B4", 0.54); Fb(4, 2, 3.6, "D5", 0.6)
    Fb(5, 2, 3, "D5", 0.48); Fb(5, 3, 4, "B4", 0.46)
    Fb(6, 0, 1.5, "A4", 0.5, "slide"); Fb(6, 1.5, 2, "B4", 0.52); Fb(6, 2, 3, "D5", 0.58); Fb(6, 3, 4, "E5", 0.64)
    Fb(7, 0, 1.5, "A5", 0.78, "grace"); Fb(7, 1.5, 2, "B5", 0.74); Fb(7, 2, 3, "A5", 0.74); Fb(7, 3, 4, "F#5", 0.7)
    Fb(8, 0, 1, "E5", 0.7); Fb(8, 1, 1.5, "F#5", 0.68); Fb(8, 1.5, 2, "E5", 0.64); Fb(8, 2, 3, "D5", 0.64)
    Fb(8, 3, 4, "B4", 0.6)
    F(B(9), 2.35, "D5", 0.62)
    # after the seal
    F(38.82, 0.46, "F#5", 0.6, "grace")
    F(39.28, 0.40, "E5", 0.56)
    F(39.68, 2.2, "D5", 0.58)

    # zheng accompaniment
    def arp(bar, b0, names, step=0.5, v=0.42, vbass=0.58):
        for i, nm in enumerate(names):
            Z(B(bar, b0 + i * step), nm, vbass if i == 0 else v * (0.9 + 0.2 * ((i * 7) % 3) / 2))

    arp(1, 0, ["D2", "A2", "D3", "F#3", "A3", "F#3", "D3", "A2"])
    arp(2, 0, ["B1", "F#2", "B2", "D3", "F#3", "D3", "B2", "F#2"])
    arp(3, 0, ["E2", "B2", "E3", "B3"]); arp(3, 2, ["A2", "E3", "A3", "E4"])
    arp(4, 0, ["D2", "A2", "D3", "E3", "F#3", "A3"])
    arp(4, 3, ["D5", "B4", "A4", "F#4"], step=0.25, v=0.4, vbass=0.42)
    # bar 5: the zheng sings while the flute rests
    Z(B(5, 0), "B1", 0.5); Z(B(5, 2), "F#2", 0.4)
    Z(B(5, 1), "D3", 0.3); Z(B(5, 3), "D3", 0.28)
    Z(B(5, 0), "B4", 0.6, orn="vib"); Z(B(5, 1), "A4", 0.52); Z(B(5, 1.5), "F#4", 0.5)
    Z(B(5, 2), "A4", 0.58, orn="slide")
    arp(6, 0, ["A1", "E2", "A2"], step=1.0, v=0.4, vbass=0.5)
    gliss(27.35, B(7) - 0.03, "A3", "D6", 0.28, 0.75, curve=1.2)
    # climax: shimmering sixteenths
    Z(B(7, 0), "D2", 0.82); Z(B(7, 2), "A2", 0.7)
    for rep in range(2):
        arp(7, rep * 2, ["D3", "A3", "D4", "F#4", "A4", "F#4", "D4", "A3"], step=0.25, v=0.38, vbass=0.4)
    Z(B(8, 0), "B1", 0.78)
    arp(8, 0, ["B2", "F#3", "B3", "D4", "F#4", "D4", "B3", "F#3"], step=0.25, v=0.38, vbass=0.4)
    Z(B(8, 2), "A1", 0.76)
    arp(8, 2, ["A2", "E3", "A3", "B3", "E4", "B3", "A3", "E3"], step=0.25, v=0.38, vbass=0.4)
    # finale: slow, spread chord (ritardando) while the title is written
    Z(B(9), "D2", 0.85, dur=7.0)
    for t, nm, v in [(35.3, "A2", 0.45), (35.75, "D3", 0.42), (36.25, "F#3", 0.4), (36.8, "A3", 0.38),
                     (37.4, "D4", 0.36)]:
        Z(t, nm, v, orn="vib" if nm == "D4" else None)
    gliss(38.05, 38.70, "D3", "D6", 0.25, 0.7, curve=1.3)
    # the seal: a full strum
    for i, nm in enumerate(["D2", "A2", "D3", "A3", "D4", "F#4", "A4", "D5"]):
        Z(38.75 + i * 0.014, nm, 0.85 - 0.03 * i, dur=6.0)
    for t, nm in [(40.1, "D6"), (40.6, "A5"), (41.1, "F#5"), (41.6, "D5")]:
        bb.add(t, harmonic_tone(midi(nm), 2.6, 0.2), 1.0, 0.2)

    fb.add(0.0, flute(fl), 0.25, 0.12)

    # pad
    chords = [
        (2.0, 8.2, ["D2", "A2", "D3"], 0.30),
        (8.2, B(2), ["D2", "A2", "D3", "F#3"], 0.34),
        (B(2), B(3), ["B1", "F#2", "B2", "D3"], 0.34),
        (B(3), B(3, 2), ["E2", "B2", "E3"], 0.34),
        (B(3, 2), B(4), ["A1", "E2", "A2", "E3"], 0.34),
        (B(4), B(5), ["D2", "A2", "D3", "F#3"], 0.36),
        (B(5), B(6), ["B1", "F#2", "B2", "D3"], 0.30),
        (B(6), B(7), ["A1", "E2", "A2", "E3"], 0.36),
        (B(7), B(8), ["D2", "A2", "D3", "F#3", "A3"], 0.48),
        (B(8), B(8, 2), ["B1", "F#2", "B2", "D3"], 0.48),
        (B(8, 2), B(9), ["A1", "E2", "A2", "E3"], 0.48),
        (B(9), 37.8, ["D2", "A2", "D3", "F#3"], 0.36),
        (38.75, 42.4, ["D2", "A2", "D3", "A3", "D4"], 0.42),
    ]
    padtrack = pad(chords)
    pb.add_st(0.0, padtrack * 0.9, np.roll(padtrack, int(0.011 * FS)) * 0.9, 0.5)

    # drums
    for t, sz, v in [(B(7, 0), 1.0, 0.75), (B(7, 2), 0.7, 0.42), (B(7, 3.5), 0.5, 0.3),
                     (B(8, 0), 1.0, 0.72), (B(8, 2), 0.8, 0.55), (B(8, 3), 0.5, 0.35),
                     (B(8, 3.5), 0.5, 0.4), (B(8, 3.75), 0.5, 0.45), (B(9), 1.2, 0.8)]:
        db.add(t, drum(sz, v, int(t * 100)), 0.36, 0.0)
    # roll into the seal
    k = 0
    tr = 37.65
    while tr < 38.68:
        u = (tr - 37.65) / 1.03
        db.add(tr, drum(0.3, 0.12 + 0.45 * u ** 1.6, 700 + k), 0.30, 0.08 * ((k % 2) * 2 - 1))
        tr += 0.11 - 0.05 * u
        k += 1
    db.add(38.75, drum(1.8, 1.0, 999), 0.48, 0.0)
    db.add(38.75, gong(80.0, 5.0, 0.55), 0.5, -0.1)

    # the title's two characters are brushed in
    bb.add(36.08, brush_swish(0.7, 1), 0.10, 0.25)
    bb.add(36.70, brush_swish(0.8, 2), 0.10, 0.25)

    # bells and chimes
    bb.add(19.4, bell(float(hz(midi("A5"))), 4.0, 0.2, 3), 1.0, 0.3)
    bb.add(27.4, bell(float(hz(midi("D5"))), 4.5, 0.18, 4), 1.0, -0.2)
    bb.add(38.75, bell(float(hz(midi("D4"))), 6.0, 0.22, 5), 1.0, 0.0)
    bb.add(38.76, bell(float(hz(midi("A4"))), 5.0, 0.12, 6), 1.0, 0.2)
    rng = np.random.default_rng(77)
    sparkle = [(22.0, 24.8, 9), (27.5, 30.0, 8), (39.0, 41.8, 9)]
    for a, b_, n in sparkle:
        for _ in range(n):
            nm = rng.choice(["D6", "E6", "F#6", "A6", "B6", "D7"])
            bb.add(rng.uniform(a, b_), chime(float(hz(midi(nm))), rng.uniform(0.05, 0.11), int(rng.integers(1000000))),
                   1.0, rng.uniform(-0.6, 0.6))
    return zb, fb, pb, db, bb


def mixdown():
    zb, fb, pb, db, bb = build()
    body = zheng_body_ir()
    for bus in (zb,):
        for ch in ("L", "R"):
            x = getattr(bus, ch)
            setattr(bus, ch, 0.7 * x + 0.5 * signal.fftconvolve(x, body)[:len(x)])
    buses = [(zb, 0.30), (fb, 0.50), (pb, 0.55), (db, 0.22), (bb, 0.60)]
    L = sum(b.L for b, _ in buses)
    R = sum(b.R for b, _ in buses)
    sL = sum(b.L * s for b, s in buses)
    sR = sum(b.R * s for b, s in buses)
    irL, irR = reverb_ir()
    wetL = signal.fftconvolve(sL + 0.35 * sR, irL)[:len(L)]
    wetR = signal.fftconvolve(sR + 0.35 * sL, irR)[:len(R)]
    L = L + 0.55 * wetL
    R = R + 0.55 * wetR
    # soft wind ambience
    rng = np.random.default_rng(31)
    wn = signal.lfilter(*signal.butter(2, 500 / (FS / 2)), rng.standard_normal(len(L)))
    mod = 0.6 + 0.4 * np.sin(2 * np.pi * 0.09 * np.arange(len(L)) / FS)
    L += 0.012 * wn * mod
    R += 0.012 * np.roll(wn, 5000) * mod
    out = np.stack([L, R])[:, :N]
    # tidy low end, gentle bus limiter
    out = signal.lfilter(*signal.butter(2, 32 / (FS / 2), "high"), out, axis=1)
    # lookahead peak limiter (5 ms) with 180 ms release, then a soft clip for safety
    from scipy.ndimage import maximum_filter1d
    lvl = np.percentile(np.abs(out), 99.7)
    out = out / (lvl + 1e-9) * 0.55
    pk = maximum_filter1d(np.abs(out).max(0), size=int(0.010 * FS) + 1)
    gain = np.minimum(1.0, 0.80 / (pk + 1e-9))
    rel = np.exp(-1.0 / (0.18 * FS))
    gs = signal.lfilter([1 - rel], [1, -rel], gain - 1.0) + 1.0
    gain = np.minimum(gain, gs)
    out = out * gain
    out = np.tanh(out * 1.05) / np.tanh(1.05)
    t = np.arange(N) / FS
    fade = np.clip(t / 0.3, 0, 1) * np.clip((DUR - t) / 1.3, 0, 1) ** 1.5
    out *= fade
    out *= 10 ** (-1.0 / 20) / (np.abs(out).max() + 1e-9)
    return out


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16).T.copy()
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    import time
    t = time.time()
    x = mixdown()
    path = sys.argv[1] if len(sys.argv) > 1 else "soundtrack.wav"
    write_wav(path, x)
    rms = np.sqrt((x ** 2).mean())
    print(f"wrote {path}: {x.shape[1] / FS:.2f}s peak={np.abs(x).max():.3f} rms={20 * np.log10(rms):.1f} dBFS "
          f"in {time.time() - t:.1f}s")
