"""Original instrumental for the invite: tanpura drone + santoor-like plucks
in a major pentatonic (Bhupali / Pahadi-folk flavour), soft pad, light frame
drum and manjira. Royalty-free by construction (synthesised here).
Output: a seamless ~63 s loop as WAV, then MP3 via ffmpeg."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
import subprocess, sys

SR = 44100
BPM = 76
BEAT = 60 / BPM
TOTAL_BEATS = 80
LOOP = int(TOTAL_BEATS * BEAT * SR)
TAIL = int(6 * SR)
L = np.zeros(LOOP + TAIL)
R = np.zeros(LOOP + TAIL)
rng = np.random.default_rng(7)

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def add(sig, t0, gain=1.0, pan=0.0):
    i = int(t0 * SR)
    n = min(len(sig), len(L) - i)
    if n <= 0:
        return
    gl = gain * np.cos((pan + 1) * np.pi / 4)
    gr = gain * np.sin((pan + 1) * np.pi / 4)
    L[i:i + n] += sig[:n] * gl
    R[i:i + n] += sig[:n] * gr

def santoor(m, vel=1.0, dur=3.2):
    f = mtof(m)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for detune in (-1.6, 0.0, 1.9):          # three strings per course -> shimmer
        fd = f * 2 ** (detune / 1200)
        for k in range(1, 9):
            fk = fd * k * np.sqrt(1 + 0.00035 * k * k)   # slight inharmonicity
            if fk > 12000:
                break
            amp = (1 / k ** 1.15) * (1.0 if k > 1 else 1.2)
            dec = 2.6 / (1 + 0.55 * (k - 1))
            out += amp * np.exp(-t / dec) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28))
    atk = np.minimum(1, t / 0.003)
    hammer = sosfilt(butter(2, 3500, btype='low', fs=SR, output='sos'), rng.normal(0, 1, len(t))) * np.exp(-t / 0.008) * 0.12
    return (out / 3 * atk + hammer) * vel

def tanpura(m, dur=4.6):
    f = mtof(m)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    centre = 2 + 13 * (1 - np.exp(-t / 1.1))           # jawari bloom moving up the harmonics
    for k in range(1, 26):
        fk = f * k
        if fk > 9000:
            break
        w = 0.35 + np.exp(-((k - centre) ** 2) / (2 * 3.2 ** 2))
        out += (w / k ** 0.7) * np.sin(2 * np.pi * fk * t + 0.3 * np.sin(2 * np.pi * 0.3 * t) + k)
    env = np.minimum(1, t / 0.02) * np.exp(-t / 3.2)
    return out * env

def pad(ms, dur, attack=2.5):
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for m in ms:
        f = mtof(m)
        for d in (-4, 4):
            fd = f * 2 ** (d / 1200)
            out += np.sin(2 * np.pi * fd * t) + 0.25 * np.sin(2 * np.pi * 2 * fd * t) + 0.08 * np.sin(2 * np.pi * 3 * fd * t)
    env = np.minimum(1, t / attack) * np.minimum(1, (dur - t) / 2.0).clip(0, 1)
    return out * env / len(ms)

def drum(vel=1.0, low=True):
    dur = 0.6
    t = np.arange(int(dur * SR)) / SR
    f0 = 95 if low else 180
    freq = f0 * (0.62 + 0.38 * np.exp(-t / 0.05))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / (0.22 if low else 0.09))
    sos = butter(2, [700, 3000], btype='band', fs=SR, output='sos')
    slap = sosfilt(sos, rng.normal(0, 1, len(t))) * np.exp(-t / 0.02) * (0.15 if low else 0.35)
    return (body + slap) * vel

def manjira(vel=1.0):
    dur = 1.6
    t = np.arange(int(dur * SR)) / SR
    out = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / d)
              for f, a, d in ((2637, 1, 1.1), (3951, 0.6, 0.8), (5274, 0.35, 0.5), (6644, 0.2, 0.3)))
    return out * np.minimum(1, t / 0.001) * vel

# ---------- tanpura cycle (Pa, Sa', Sa', Sa) over the whole loop ----------
D2, A2, D3 = 38, 45, 50
cycle = [A2, D3, D3, D2]
step = 1.2 * BEAT
tt, i = 0.0, 0
while tt < TOTAL_BEATS * BEAT + 2:
    add(tanpura(cycle[i % 4]), tt, 0.055, pan=[-0.2, 0.15, 0.25, -0.1][i % 4])
    tt += step if i % 4 != 3 else step * 1.6
    i += 1

# ---------- pad ----------
add(pad([50, 57, 62], 26 * BEAT), 0, 0.05)
add(pad([47, 54, 59, 62], 16 * BEAT), 24 * BEAT, 0.045)
add(pad([50, 57, 62, 66], 18 * BEAT), 38 * BEAT, 0.05)
add(pad([45, 52, 57, 64], 16 * BEAT), 54 * BEAT, 0.045)
add(pad([50, 57, 62], 14 * BEAT), 68 * BEAT, 0.05)

# ---------- melody (Bhupali: D E F# A B) ----------
D4, E4, Fs4, A4, B4, D5, E5, Fs5, A3, B3 = 62, 64, 66, 69, 71, 74, 76, 78, 57, 59
phrases = [
    (0, [(D5, 2), (A4, 2), (None, 4)]),
    (8, [(Fs4, 1), (E4, .5), (D4, .5), (E4, 1), (Fs4, 1), (A4, 2), (Fs4, 1), (E4, 1),
         (D4, 1), (E4, 1), (Fs4, 1), (A4, 1), (B4, 1.5), (A4, .5), (Fs4, 2)]),
    (24, [(A4, 1), (B4, 1), (D5, 2), (B4, .5), (A4, .5), (Fs4, 1), (A4, 2),
          (Fs4, 1), (E4, 1), (D4, 1), (E4, 1), (Fs4, 2), (D4, 2)]),
    (40, [(D5, 1), (E5, 1), (Fs5, 2), (E5, .5), (D5, .5), (B4, 1), (D5, 2),
          (B4, 1), (A4, 1), (Fs4, 1), (A4, 1), (B4, 1), (A4, .5), (Fs4, .5), (E4, 2)]),
    (56, [(Fs4, 1), (A4, 1), (B4, 1), (D5, 1), (B4, 1), (A4, 1), (Fs4, 2),
          (E4, 1), (Fs4, .5), (E4, .5), (D4, 2), (A3, 1), (B3, 1), (D4, 2)]),
    (72, [(D5, 2), (A4, 1), (Fs4, 1), (D4, 4)]),
]
for start, notes in phrases:
    b = start
    for idx, (m, d) in enumerate(notes):
        if m is not None:
            on_beat = abs(b - round(b)) < 1e-6
            vel = 0.95 if on_beat else 0.75
            pan = 0.18 * np.sin(b * 1.3)
            add(santoor(m, vel), b * BEAT, 0.16, pan)
            if d >= 2:                                # santoor-style tremolo on long notes
                k = 0.5
                while k < d - 0.01:
                    add(santoor(m, 0.42 - 0.08 * k, 1.8), (b + k) * BEAT, 0.16, pan)
                    k += 0.5
            if on_beat and int(round(b)) % 4 == 0 and start >= 8:   # low octave support
                add(santoor(m - 12, 0.5, 2.5), b * BEAT, 0.11, -pan)
        b += d

# ---------- percussion: soft frame drum from bar 24, manjira from 40 ----------
for bar in range(24, 72, 4):
    for off, low, v in ((0, True, 1.0), (1.5, False, 0.5), (2, True, 0.7), (3, False, 0.55), (3.5, False, 0.35)):
        add(drum(v, low), (bar + off) * BEAT, 0.12, 0.1 if low else -0.15)
for bar in range(40, 72, 4):
    add(manjira(0.8), bar * BEAT, 0.035, 0.4)
    add(manjira(0.5), (bar + 2) * BEAT, 0.03, -0.4)
add(manjira(0.9), 72 * BEAT, 0.04, 0.0)

# ---------- reverb ----------
def reverb(x, seconds=2.4, seed=1):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.random.default_rng(seed).normal(0, 1, n) * np.exp(-t / (seconds / 4.5))
    sos = butter(1, 5000, btype='low', fs=SR, output='sos')
    ir = sosfilt(sos, ir)
    ir[:int(0.012 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    return fftconvolve(x, ir)[:len(x)]

wetL, wetR = reverb(L, seed=1), reverb(R, seed=2)
mixL, mixR = L * 0.78 + wetL * 0.42, R * 0.78 + wetR * 0.42

# ---------- seamless loop: fold the tail onto the start ----------
outL, outR = mixL[:LOOP].copy(), mixR[:LOOP].copy()
outL[:TAIL] += mixL[LOOP:LOOP + TAIL]
outR[:TAIL] += mixR[LOOP:LOOP + TAIL]
st = np.stack([outL, outR], axis=1)
st = np.tanh(st / np.max(np.abs(st)) * 1.15) * 0.89
wav = (st * 32767).astype(np.int16)

import wave
with wave.open(sys.argv[1], 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(wav.tobytes())
print('seconds', LOOP / SR)
