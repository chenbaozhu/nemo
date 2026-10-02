#!/usr/bin/env python3
""""Inside the cup" POV effect: look up at a face through a glass of iced matcha.

Everything except the selfie is procedural: a log-polar tunnel of ice cubes
(rounded Voronoi cells) swirling in matcha, rising bubbles, caustics, the glass
rim, a fisheye on the face and a straw reaching toward the camera.

    python3 render.py            # -> out/cup_pov.mp4
    python3 render.py --preview  # key frames as PNG only
"""
import argparse
import math
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PHOTO = os.path.join(HERE, "assets", "selfie.jpg")

W, H = 1080, 1350
FPS = 24
DURATION = 10.0
SR = 48000

# Window (the open top of the cup, seen from below)
CX, CY, R = 540.0, 600.0, 400.0
RIM = 22.0
# Region of the selfie shown in the window: centre and radius in source px
SRC_C = (566.0, 1085.0)
SRC_R = 530.0
FISHEYE = 0.28          # >0 bulges the centre of the face

# Ice tunnel
N_AROUND = 13           # cubes around the circumference
RADIAL_STRETCH = 1.35   # >1 flattens cubes radially for depth
SWIRL = 0.16            # cubes per second around
SINK = 0.05             # rings per second toward the camera

# Matcha palette (linear-ish 0..1)
GAP = np.array([0.17, 0.21, 0.07], np.float32)
ICE = np.array([0.50, 0.54, 0.27], np.float32)
FOAM = np.array([0.93, 0.95, 0.80], np.float32)

yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
DX, DY = xx - CX, yy - CY
RAD = np.sqrt(DX ** 2 + DY ** 2) + 1e-6
THETA = np.arctan2(DY, DX)
U0 = THETA / (2 * np.pi) * N_AROUND
V0 = RADIAL_STRETCH * N_AROUND / (2 * np.pi) * np.log(np.maximum(RAD, R) / R)
DEPTH = np.clip((RAD - R) / (math.hypot(W / 2, H) - R), 0, 1)


# ---- helpers ---------------------------------------------------------------
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def hash2(i, j, seed):
    h = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


class ValueNoise:
    """Smooth tileable value noise sampled with bilinear + smoothstep."""

    def __init__(self, size, seed):
        self.g = np.random.default_rng(seed).random((size, size)).astype(np.float32)
        self.n = size

    def __call__(self, x, y, px=None):
        """px: optional period along x (integer <= size) for seamless wrapping."""
        n = self.n
        px = px or n
        xi, yi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = x - xi, y - yi
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        x0, y0, x1, y1 = xi % px, yi % n, (xi + 1) % px, (yi + 1) % n
        g = self.g
        a = g[y0, x0] + (g[y0, x1] - g[y0, x0]) * fx
        b = g[y1, x0] + (g[y1, x1] - g[y1, x0]) * fx
        return a + (b - a) * fy


NOISE_A, NOISE_B, NOISE_C = ValueNoise(64, 1), ValueNoise(64, 2), ValueNoise(256, 3)


def voronoi(u, v, period, seed, p=3.0):
    """Rounded-square Voronoi: returns F1, F2, cell hash, offset to nearest point."""
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    f1 = np.full(u.shape, 9.0, np.float32)
    f2 = np.full(u.shape, 9.0, np.float32)
    cid = np.zeros(u.shape, np.float32)
    ndy = np.zeros(u.shape, np.float32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ci = (iu + di).astype(np.int64) % period
            cj = (iv + dj).astype(np.int64)
            px = 0.2 + 0.6 * hash2(ci, cj, seed)
            py = 0.2 + 0.6 * hash2(ci, cj, seed + 7)
            dx = di + px - fu
            dy = dj + py - fv
            d = (np.abs(dx) ** p + np.abs(dy) ** p) ** (1 / p)
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            f1 = np.where(closer, d, f1)
            cid = np.where(closer, hash2(ci, cj, seed + 13), cid)
            ndy = np.where(closer, dy, ndy)
    return f1, f2, cid, ndy


def bilinear(img, x, y):
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    b = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return a * (1 - fy) + b * fy


# ---- layers ----------------------------------------------------------------
def liquid(t):
    """Matcha + ice tunnel outside the window, float RGB 0..1."""
    wu = 0.22 * (NOISE_A(U0 + 3, V0 * 1.6 + t * 0.15, N_AROUND) - 0.5)
    wv = 0.22 * (NOISE_B(U0 + t * 0.12, V0 * 1.6 + 5, N_AROUND) - 0.5)
    u = U0 + SWIRL * t + wu
    v = V0 - SINK * t + wv
    f1, f2, cid, ndy = voronoi(u, v, N_AROUND, 11)
    edge = f2 - f1
    cube = smoothstep(0.07, 0.24, edge)

    bright = 0.82 + 0.36 * cid
    facing = 1 + 0.28 * np.clip(ndy / (f1 + 1e-3), -1, 1)          # side facing the light
    frost = 0.8 + 0.4 * NOISE_C(u * 5, v * 5, 5 * N_AROUND) * (0.75 + 0.25 * NOISE_C(u * 17 + 3, v * 17, 17 * N_AROUND))
    ice = ICE[None, None, :] * (bright * facing * frost)[..., None]
    # cube edges catch light, centres look into the green
    edge_hl = np.exp(-((edge - 0.24) / 0.06) ** 2)[..., None]
    centre = smoothstep(0.3, 0.8, edge)[..., None]
    ice = ice * (1 - 0.18 * centre) + edge_hl * np.array([0.20, 0.22, 0.12], np.float32)

    # tiny bubbles caught in the gaps (they ride with the cubes)
    specks = np.zeros(u.shape, np.float32)
    for k, (sc, amt) in enumerate(((14, 0.35), (26, 0.25))):
        bu, bv = u * sc, v * sc
        bi, bj = np.floor(bu), np.floor(bv)
        ci = (bi % (N_AROUND * sc)).astype(np.int64)
        bh = hash2(ci, bj.astype(np.int64), 29 + k)
        bx = 0.25 + 0.5 * hash2(ci, bj.astype(np.int64), 31 + k)
        by = 0.25 + 0.5 * hash2(ci, bj.astype(np.int64), 37 + k)
        bd = np.sqrt((bu - bi - bx) ** 2 + (bv - bj - by) ** 2)
        brad = 0.06 + 0.14 * bh
        specks += amt * smoothstep(brad, brad * 0.4, bd) * (bh > 0.35)
    gap = GAP[None, None, :] * (0.7 + 0.6 * NOISE_C(u * 3 + 9, v * 3, 3 * N_AROUND))[..., None]
    gap = gap + specks[..., None] * np.array([0.55, 0.6, 0.4], np.float32)

    col = gap * (1 - cube[..., None]) + ice * cube[..., None]
    col = col + (specks * cube * 0.25)[..., None] * 0.4

    # caustics: thin bright filaments drifting through the liquid
    c1 = NOISE_A(xx / 90 + t * 0.35, yy / 90 - t * 0.2)
    c2 = NOISE_B(xx / 70 - t * 0.25, yy / 70 + t * 0.3)
    caus = np.exp(-((c1 - c2) / 0.08) ** 2)
    col = col * (0.94 + 0.12 * caus)[..., None]

    # light falls off away from the opening
    light = 1.1 - 0.7 * DEPTH ** 0.8
    col = col * light[..., None]
    # bright glow just outside the window where the drink is thin
    glow = np.exp(-np.maximum(RAD - R, 0) / 45.0)
    col = col + glow[..., None] * np.array([0.22, 0.24, 0.14], np.float32)
    return col


def window(photo, t):
    """Selfie seen through the cup opening (fisheye + handheld sway)."""
    sway_x = 9 * math.sin(t * 1.3) + 5 * math.sin(t * 2.9 + 1)
    sway_y = 7 * math.sin(t * 1.1 + 2) + 4 * math.sin(t * 3.3)
    rot = math.radians(1.6 * math.sin(t * 0.9) + 0.6 * math.sin(t * 2.3))
    zoom = 1 + 0.03 * math.sin(t * 0.7)

    rr = np.minimum(RAD / R, 1.0)
    s = rr * ((1 - FISHEYE) + FISHEYE * rr ** 2) / zoom
    # meniscus wobble near the rim
    wob = 1 + 0.012 * smoothstep(0.82, 1.0, rr) * np.sin(THETA * 7 + t * 3.0)
    s = s * wob
    ang = THETA + rot
    sx = SRC_C[0] + sway_x + np.cos(ang) * s * SRC_R
    sy = SRC_C[1] + sway_y + np.sin(ang) * s * SRC_R
    col = bilinear(photo, sx, sy)
    # slight green cast + inner shadow at the rim
    col = col * np.array([0.96, 1.0, 0.92], np.float32)
    col = col * (1 - 0.22 * smoothstep(0.86, 1.0, rr))[..., None]
    return col


def straw(col, t):
    """Translucent straw from the mouth toward the camera."""
    sway = 6 * math.sin(t * 1.3)
    y0, y1 = CY + R - 30, H + 40
    yrel = np.clip((yy - y0) / (y1 - y0), 0, 1)
    cx = (CX + 2 + sway) + (460 - CX) * yrel
    half = 16 + 95 * yrel ** 1.6
    a = (xx - cx) / half
    inside = smoothstep(1.0, 0.92, np.abs(a)) * (yy > y0) * (RAD > R - 2)
    tube = np.sqrt(np.clip(1 - a * a, 0, 1))
    base = np.array([0.52, 0.60, 0.36], np.float32)
    s = base[None, None, :] * (0.55 + 0.45 * tube)[..., None]
    hl = np.exp(-((a + 0.38) / 0.11) ** 2) * 0.55 + np.exp(-((a - 0.62) / 0.06) ** 2) * 0.18
    s = s + hl[..., None] * np.array([0.9, 0.95, 0.8], np.float32)
    s = s * (1.15 - 0.5 * yrel)[..., None]
    alpha = (inside * 0.8)[..., None]
    return col * (1 - alpha) + s * alpha


def rim(col):
    """Glass edge of the opening: a bright refractive ring."""
    d = RAD - R
    ring = np.exp(-((d - RIM * 0.25) / 2.5) ** 2) * 0.55 + np.exp(-((d - RIM) / 6.0) ** 2) * 0.15
    band = smoothstep(-1, 1, d) * smoothstep(RIM * 1.6, RIM * 0.6, d)
    col = col * (1 - 0.15 * band[..., None])
    return col + ring[..., None] * FOAM[None, None, :]


class Bubbles:
    """Bubbles rising from the bottom toward the opening (inward in the frame)."""

    def __init__(self, n=90, seed=5):
        rng = np.random.default_rng(seed)
        self.th = rng.uniform(0, 2 * np.pi, n)
        self.v0 = rng.uniform(0, 1, n)
        self.speed = rng.uniform(0.05, 0.16, n)
        self.size = rng.uniform(0.004, 0.012, n)
        self.wob = rng.uniform(0, 2 * np.pi, n)

    def draw(self, t):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)
        rmax = math.hypot(W / 2, H)
        for th, v0, sp, sz, wb in zip(self.th, self.v0, self.speed, self.size, self.wob):
            p = (v0 - sp * t) % 1.0                    # 1 = near camera, 0 = at the rim
            r = R + 8 + (rmax - R) * p ** 1.7
            a = th + SWIRL * t * 2 * np.pi / N_AROUND + 0.05 * math.sin(t * 2 + wb)
            x, y = CX + r * math.cos(a), CY + r * math.sin(a)
            rad = max(1.5, sz * r)
            fade = min(1.0, (1 - p) * 4) * min(1.0, p * 6)
            al = int(120 * fade)
            dr.ellipse((x - rad, y - rad, x + rad, y + rad), fill=(200, 210, 160, al // 4), outline=(225, 232, 195, al), width=max(1, int(rad * 0.18)))
            hr = rad * 0.3
            dr.ellipse((x - rad * 0.45 - hr, y - rad * 0.45 - hr, x - rad * 0.45 + hr, y - rad * 0.45 + hr), fill=(255, 255, 245, al))
        return layer.filter(ImageFilter.GaussianBlur(0.6))


# ---- film ------------------------------------------------------------------
class Film:
    def __init__(self):
        self.photo = np.asarray(Image.open(PHOTO).convert("RGB")).astype(np.float32) / 255.0
        self.bubbles = Bubbles()
        self.inside = np.clip(R - RAD + 0.5, 0, 1)[..., None]
        vig = np.sqrt((DX / (W * 0.75)) ** 2 + (DY / (H * 0.75)) ** 2)
        self.vignette = (1 - 0.35 * smoothstep(0.5, 1.1, vig))[..., None]
        self.rng = np.random.default_rng(9)

    def render(self, t):
        liq = liquid(t)
        # depth of field: the drink near the lens is out of focus
        li = Image.fromarray(np.clip(liq * 255, 0, 255).astype(np.uint8))
        soft = np.asarray(li.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255
        blur = np.asarray(li.filter(ImageFilter.GaussianBlur(7))).astype(np.float32) / 255
        k = smoothstep(0.25, 0.9, DEPTH)[..., None]
        liq = soft * (1 - k) + blur * k
        col = liq * (1 - self.inside) + window(self.photo, t) * self.inside
        col = straw(col, t)
        col = rim(col)
        col = col * self.vignette
        img = np.clip(col * 255 + self.rng.normal(0, 2.0, (H, W, 1)), 0, 255).astype(np.uint8)
        frame = Image.fromarray(img, "RGB").convert("RGBA")
        frame.alpha_composite(self.bubbles.draw(t))
        return frame.convert("RGB")


# ---- sound: bubble blips + a couple of ice clinks ---------------------------
def soundtrack(path):
    n = int(SR * DURATION)
    t = np.arange(n) / SR
    out = np.zeros(n)
    rng = np.random.default_rng(4)
    for _ in range(70):  # a bubble is a decaying sine whose pitch rises
        t0 = rng.uniform(0, DURATION - 0.1)
        f0 = rng.uniform(500, 1400)
        tt = t - t0
        m = (tt >= 0) & (tt < 0.08)
        out[m] += rng.uniform(0.02, 0.07) * np.sin(2 * np.pi * f0 * (tt[m] + 6 * tt[m] ** 2)) * np.exp(-tt[m] * 60)
    for t0 in (1.7, 4.9, 7.6):  # ice clink: inharmonic partials
        tt = t - t0
        m = tt >= 0
        for f, a in ((2350, .10), (3710, .06), (5230, .035)):
            out[m] += a * np.sin(2 * np.pi * f * tt[m]) * np.exp(-tt[m] * 9)
    # soft liquid hiss under everything
    noise = rng.standard_normal(n)
    hiss = np.convolve(noise, np.ones(40) / 40, mode="same") * 0.05
    out += hiss * (0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * t))
    out *= np.clip(t / 0.3, 0, 1) * np.clip((DURATION - t) / 0.4, 0, 1)
    out = np.tanh(out * 1.5) * 0.7
    st = np.stack([out, np.roll(out, 240)], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    film = Film()
    if args.preview:
        for ts in (0.0, 3.0, 6.0, 9.0):
            film.render(ts).save(os.path.join(OUT, f"frame_{ts:03.1f}s.png"))
        return

    wav = os.path.join(OUT, "sound.wav")
    soundtrack(wav)
    mp4 = os.path.join(OUT, "cup_pov.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(DURATION * FPS)
    for i in range(total):
        proc.stdin.write(film.render(i / FPS).tobytes())
        if i % 48 == 0:
            print(f"  frame {i}/{total}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", mp4)


if __name__ == "__main__":
    main()
