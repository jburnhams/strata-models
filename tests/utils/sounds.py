"""Synthetic speech sounds for tests of the phonics cutter (workers/phonics.py) and the TTS worker: a hiss, a hum or a click, then a vowel made
of harmonics with formant peaks, at 24 kHz (Kokoro's rate)."""

import numpy as np

SR = 24000


def vowel(secs, f0=200.0, level=0.5):
    """A vowel-like tone: harmonics of f0 shaped by formants at 700 and 1800 Hz (most of the energy in 500 Hz to 3 kHz)."""
    t = np.arange(int(secs * SR)) / SR
    amp = [np.exp(-(((k * f0 - 700) / 250) ** 2)) + 0.7 * np.exp(-(((k * f0 - 1800) / 300) ** 2)) for k in range(1, 20)]
    x = sum(a * np.sin(2 * np.pi * k * f0 * t) for k, a in enumerate(amp, 1))
    return (level * x / np.abs(x).max()).astype(np.float32)


def hiss(secs, level=0.1, seed=0):
    """Noise above 3 kHz (a fricative's shape)."""
    n = int(secs * SR)
    f = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    f[np.fft.rfftfreq(n, 1 / SR) < 3000] = 0
    x = np.fft.irfft(f, n)
    return (level * x / np.abs(x).max()).astype(np.float32)


def hum(secs, f0=200.0, level=0.15):
    """A nasal murmur: the fundamental and a weak second harmonic, nearly all of it under 500 Hz."""
    t = np.arange(int(secs * SR)) / SR
    return (level * (np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(4 * np.pi * f0 * t))).astype(np.float32)


def silence(secs):
    return np.zeros(int(secs * SR), np.float32)


def carrier(*parts):
    return np.concatenate([silence(0.1), *parts, silence(0.1)])
