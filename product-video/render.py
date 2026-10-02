#!/usr/bin/env python3
"""Render a 10-second Apple-style product film for the "Halo" sunglasses.

Pipeline: every frame is composited in Python (Pillow + NumPy) and piped
straight into ffmpeg; the soundtrack is synthesised procedurally.

    python3 render.py            # -> out/halo_film.mp4
    python3 render.py --preview  # also dumps a few key frames as PNG
"""
import argparse
import math
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
FONTS = os.path.join(ASSETS, "fonts")
OUT = os.path.join(HERE, "out")

W, H = 1920, 1080
FPS = 30
DURATION = 10.0
SR = 48000

# ---- Copy (edit freely) ----------------------------------------------------
OPEN_LINE = "Lighter than light."
SPEC_LINE = "Pure titanium.  Featherweight."
PRODUCT_NAME = "Halo"
TAGLINE_ZH = "輕若無物，經典如初。"
FOOTER = "Titanium Round Sunglasses"

# Apple-ish palette
INK = (29, 29, 31)
GREY = (110, 110, 115)
WHITE = (245, 245, 247)

# Product source geometry (pixels in sunglasses.png, 1672x941)
SRC_BG = 254.0          # studio background level of the source photo
UP = 1                  # master upscale factor (source is already high-res)
FULL_CENTER = (817.0, 448.0)
SHADOW = dict(offset=70, blur=34, squash=0.35, opacity=0.11)


# ---- easing ----------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, t0, t1):
    return clamp((t - t0) / (t1 - t0))


def ease_io(x):  # cubic in-out
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):  # quint out, the "Apple settle"
    return 1 - (1 - x) ** 5


def lerp(a, b, x):
    return a + (b - a) * x


# ---- assets ----------------------------------------------------------------
def load_product():
    """Return (transmittance, body mask) masters upscaled UP times.

    The photo is a dark object on a white sweep, so dividing by the
    background level gives a multiply layer: background -> 1.0 (transparent),
    product -> < 1.0, and it can be re-lit on any light backdrop. The studio
    shot has almost no shadow, so a soft ground shadow is synthesised from the
    product silhouette to keep it from floating.
    """
    big = Image.open(os.path.join(ASSETS, "sunglasses.png")).convert("RGB")
    if UP != 1:
        big = big.resize((big.width * UP, big.height * UP), Image.LANCZOS)
    arr = np.asarray(big).astype(np.float32)
    trans = np.clip(arr / SRC_BG, 0, 1)
    lum = arr.mean(axis=2)
    body = np.clip((150.0 - lum) / 90.0, 0, 1)  # lens + frame

    # Ground shadow: silhouette squashed toward its lower edge, pushed down, blurred.
    sil = Image.fromarray((body * 255).astype(np.uint8), "L")
    ys = np.nonzero(body.max(axis=1) > 0.5)[0]
    base = int(ys.max())
    sh_h = max(1, int(sil.height * SHADOW["squash"]))
    squashed = sil.resize((sil.width, sh_h), Image.BILINEAR)
    canvas = Image.new("L", sil.size, 0)
    canvas.paste(squashed, (0, base - int((base / sil.height) * sh_h) + SHADOW["offset"] * UP))
    canvas = canvas.filter(ImageFilter.GaussianBlur(SHADOW["blur"] * UP))
    shadow = np.asarray(canvas).astype(np.float32) / 255.0
    trans = trans * (1 - SHADOW["opacity"] * shadow)[..., None]
    trans_img = Image.fromarray((trans * 255).astype(np.uint8), "RGB")
    body_img = Image.fromarray((body * 255).astype(np.uint8), "L")
    body_img = body_img.filter(ImageFilter.GaussianBlur(1.5))
    return trans_img, body_img


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


# ---- background ------------------------------------------------------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.46) / (H * 0.75)) ** 2)
BACKDROP = (np.array([251, 251, 253], np.float32)[None, None, :] * (1 - np.clip(_r, 0, 1)[..., None])
            + np.array([226, 226, 231], np.float32)[None, None, :] * np.clip(_r, 0, 1)[..., None])
SWEEP_AXIS = (xx * math.cos(math.radians(28)) + yy * math.sin(math.radians(28)))
RNG = np.random.default_rng(7)


# ---- camera ----------------------------------------------------------------
def camera(t):
    """(cx, cy, zoom, rotation_deg, screen_dy) at time t."""
    # Shot A: macro drift across the right lens.
    a = ease_io(prog(t, 1.6, 4.6))
    cx, cy = lerp(800, 700, a), lerp(500, 470, a)
    z, rot = lerp(2.1, 1.8, a), lerp(-4.0, -1.5, a)
    # Shot B: one continuous dolly out to the hero framing.
    b = ease_io(prog(t, 4.2, 6.8))
    cx, cy = lerp(cx, FULL_CENTER[0], b), lerp(cy, FULL_CENTER[1], b)
    z, rot = lerp(z, 0.88, b), lerp(rot, 0.0, b)
    # Shot C: hero breathe.
    c = prog(t, 6.8, 8.0)
    z *= 1 + 0.025 * ease_io(c)
    dy = 6 * math.sin(max(0.0, t - 6.8) * 1.6) * (1 - prog(t, 7.9, 8.4))
    # Shot D: lift product to make room for the end card.
    d = ease_out(prog(t, 7.9, 9.0))
    z = lerp(z, 0.69, d)
    dy = lerp(dy, -150, d)
    return cx, cy, z, rot, dy


def warp(img, cam, fill):
    cx, cy, z, rot, dy = cam
    s = UP / z
    co, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    a, b = s * co, s * si
    d, e = -s * si, s * co
    oy = H / 2 + dy
    c0 = UP * cx - a * W / 2 - b * oy
    f0 = UP * cy - d * W / 2 - e * oy
    return img.transform((W, H), Image.AFFINE, (a, b, c0, d, e, f0),
                         resample=Image.BICUBIC, fillcolor=fill)


# ---- typography ------------------------------------------------------------
def text_layer(text, fnt, color, tracking=0.0):
    """Render text with letter spacing into a tight RGBA layer."""
    widths = [fnt.getlength(ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    asc, desc = fnt.getmetrics()
    pad = 60
    layer = Image.new("RGBA", (int(total) + pad * 2, asc + desc + pad * 2), color + (0,))
    dr = ImageDraw.Draw(layer)
    x = pad
    for ch, w in zip(text, widths):
        dr.text((x, pad), ch, font=fnt, fill=color + (255,))
        x += w + tracking
    return layer


def place(frame, layer, cx, cy, alpha, blur=0.0):
    if alpha <= 0.003:
        return
    if blur > 0.25:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    if alpha < 1:
        r, g, b, a = layer.split()
        a = a.point(lambda v: int(v * alpha))
        layer = Image.merge("RGBA", (r, g, b, a))
    frame.alpha_composite(layer, (int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2))))


def reveal(t, t_in, t_out, dur_in=0.9, dur_out=0.5):
    """alpha, blur, y-offset for an Apple-style soft focus reveal."""
    i = ease_out(prog(t, t_in, t_in + dur_in))
    o = ease_io(prog(t, t_out, t_out + dur_out)) if t_out is not None else 0.0
    alpha = i * (1 - o)
    blur = 14 * (1 - i) + 10 * o
    dy = 28 * (1 - i) - 14 * o
    return alpha, blur, dy


# ---- frame -----------------------------------------------------------------
class Film:
    def __init__(self):
        self.trans, self.body = load_product()
        self.f_open = font("InterDisplay-SemiBold.ttf", 104)
        self.f_spec = font("InterDisplay-Medium.ttf", 40)
        self.f_name = font("InterDisplay-SemiBold.ttf", 150)
        self.f_zh = font("NotoTC-500.ttf", 50)
        self.f_foot = font("InterDisplay-Medium.ttf", 26)

    def product_plate(self, t):
        cam = camera(t)
        T = np.asarray(warp(self.trans, cam, (255, 255, 255))).astype(np.float32) / 255.0
        M = np.asarray(warp(self.body, cam, 0)).astype(np.float32) / 255.0
        img = BACKDROP * T / 255.0

        # Specular sweeps: a soft light bar gliding across the lenses.
        hl = np.zeros((H, W), np.float32)
        for t0, t1, width, strength in ((2.6, 4.4, 170.0, 0.42), (6.9, 8.1, 120.0, 0.32)):
            p = prog(t, t0, t1)
            if 0 < p < 1:
                pos = lerp(-600, W * math.cos(math.radians(28)) + H * math.sin(math.radians(28)) + 600, ease_io(p))
                hl += strength * np.exp(-((SWEEP_AXIS - pos) / width) ** 2) * math.sin(math.pi * p) ** 0.5
        # Constant soft rim so the black frame reads as metal, not a cut-out.
        hl += 0.05
        img = 1 - (1 - img) * (1 - (hl * M)[..., None])

        # Light comes on: exposure ramp out of black.
        exp_ = ease_io(prog(t, 2.0, 3.1))
        img = img * exp_
        return img

    def render(self, t):
        if t < 2.0:
            img = np.zeros((H, W, 3), np.float32)
        else:
            img = self.product_plate(t)

        # End: fade to black.
        img *= 1 - ease_io(prog(t, 9.55, 10.0))
        img = img * 255 + RNG.uniform(-0.6, 0.6, (H, W, 1)).astype(np.float32)  # dither
        frame = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

        fade_all = 1 - ease_io(prog(t, 9.55, 10.0))

        # Opening title on black.
        a, bl, dy = reveal(t, 0.25, 1.45, dur_in=1.0, dur_out=0.45)
        if a > 0:
            track = lerp(10, 0, ease_out(prog(t, 0.25, 1.9)))
            place(frame, text_layer(OPEN_LINE, self.f_open, WHITE, track), W / 2, H / 2 + dy, a, bl)

        # Spec line under the hero shot.
        a, bl, dy = reveal(t, 6.7, 7.75, dur_in=0.8, dur_out=0.4)
        if a > 0:
            place(frame, text_layer(SPEC_LINE, self.f_spec, GREY, 1.0), W / 2, H * 0.83 + dy, a, bl)

        # End card.
        a, bl, dy = reveal(t, 8.25, None, dur_in=1.0)
        if a > 0:
            track = lerp(18, -2, ease_out(prog(t, 8.25, 9.4)))
            place(frame, text_layer(PRODUCT_NAME, self.f_name, INK, track), W / 2, H * 0.665 + dy, a * fade_all, bl)
        a, bl, dy = reveal(t, 8.6, None, dur_in=0.9)
        if a > 0:
            place(frame, text_layer(TAGLINE_ZH, self.f_zh, INK, 6), W / 2, H * 0.79 + dy, a * fade_all, bl)
        a, bl, dy = reveal(t, 8.9, None, dur_in=0.8)
        if a > 0:
            place(frame, text_layer(FOOTER.upper(), self.f_foot, GREY, 7), W / 2, H * 0.88 + dy * 0.5, a * fade_all, bl)

        return frame.convert("RGB")


# ---- soundtrack ------------------------------------------------------------
def soundtrack(path):
    n = int(SR * DURATION)
    t = np.arange(n) / SR
    out = np.zeros((n, 2), np.float64)

    def env(t0, a, hold, r):
        e = np.clip((t - t0) / a, 0, 1)
        e = np.where(t > t0 + a + hold, np.exp(-(t - t0 - a - hold) / r), e)
        return np.where(t < t0, 0, e)

    def note(f, amp, e, detune=0.25, pan=0.0):
        sig = 0.5 * (np.sin(2 * np.pi * (f - detune) * t) + np.sin(2 * np.pi * (f + detune) * t + 0.7))
        sig += 0.12 * np.sin(2 * np.pi * 2 * f * t)
        s = sig * amp * e
        out[:, 0] += s * (1 - pan) / 2
        out[:, 1] += s * (1 + pan) / 2

    rng = np.random.default_rng(3)

    def lowpass(x, alpha):
        y = np.empty_like(x)
        acc = 0.0
        for i in range(len(x)):  # fine for 10 s
            acc += alpha * (x[i] - acc)
            y[i] = acc
        return y

    # Low drone under the opening title.
    note(55.0, 0.20, env(0.0, 1.5, 0.4, 0.6), detune=0.1)
    # Riser into the light-on moment.
    noise = rng.standard_normal(n)
    riser = lowpass(noise, 0.06) * np.clip((t - 1.0) / 1.0, 0, 1) ** 2 * (t < 2.05)
    out += riser[:, None] * 0.35
    # Warm pad (A add9) from the reveal on.
    pad_env = env(2.0, 1.4, 5.6, 1.2)
    for f, amp, pan in ((110.0, .16, -.2), (164.81, .11, .3), (220.0, .10, -.4), (277.18, .07, .4), (329.63, .06, -.1), (493.88, .035, .2)):
        note(f, amp, pad_env, pan=pan)
    # Soft impact as the light comes on.
    hit = np.sin(2 * np.pi * (48 + 40 * np.exp(-(t - 2.0) * 9)) * (t - 2.0)) * np.exp(-(t - 2.0) * 3.2) * (t >= 2.0)
    out += hit[:, None] * 0.45
    # Glass chimes on the specular sweeps and on the logo.
    for t0, freqs in ((3.25, (1760.0, 2637.0)), (7.35, (1318.5, 1975.5)), (8.3, (880.0, 1318.5, 2217.5))):
        for k, f in enumerate(freqs):
            e = np.exp(-(t - t0) * 2.2) * (t >= t0)
            out[:, k % 2] += 0.05 * np.sin(2 * np.pi * f * (t - t0)) * e
            out[:, 1 - k % 2] += 0.025 * np.sin(2 * np.pi * f * (t - t0)) * e
    # Logo bass bloom.
    note(55.0, 0.25, env(8.25, 0.05, 0.2, 0.9), detune=0.0)

    # Simple stereo slapback for space, master fade.
    d = int(0.11 * SR)
    out[d:, 0] += 0.25 * out[:-d, 1]
    out[d:, 1] += 0.25 * out[:-d, 0]
    out *= np.clip((DURATION - t) / 0.5, 0, 1)[:, None]
    out = np.tanh(out * 1.2) / np.tanh(1.2) * 0.8
    pcm = (out * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="dump key frames as PNG")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    film = Film()
    if args.preview:
        for ts in (1.0, 2.6, 3.6, 5.2, 7.2, 9.4):
            film.render(ts).save(os.path.join(OUT, f"frame_{ts:04.1f}s.png"))

    wav = os.path.join(OUT, "soundtrack.wav")
    soundtrack(wav)

    mp4 = os.path.join(OUT, "halo_film.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(DURATION * FPS)
    for i in range(total):
        proc.stdin.write(film.render(i / FPS).tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{total}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", mp4)


if __name__ == "__main__":
    main()
