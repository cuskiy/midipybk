import numpy as np
import soundfile as sf
import mido
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
from scipy.signal import fftconvolve, butter, sosfilt, lfilter

A4 = 440.0
SAMPLE_RATE = 44100
HEAD_RADIUS = 0.0875
SPEED_OF_SOUND = 343.0
MAX_PEDAL_SUSTAIN = 5.0
ANTICLICK_SAMPLES = 64
VEL_CURVE_EXP = 0.65
MAX_DETUNE_HARMONICS = 5
_NOISE_BP_SOS = butter(2, [800, 8000], btype='band', fs=SAMPLE_RATE, output='sos')


def butter_lpf(signal, cutoff, order=2):
    return sosfilt(butter(order, np.clip(cutoff, 30, SAMPLE_RATE * 0.45),
                          btype='low', fs=SAMPLE_RATE, output='sos'), signal)


def pitch_register(freq):
    return np.clip((np.log2(freq / 261.63) + 2) / 4, 0.0, 1.0)


def pitch_low_factor(freq):
    return np.clip(1.0 - pitch_register(freq), 0.0, 1.0)


def vel_curve(velocity):
    return velocity ** VEL_CURVE_EXP


def note_to_frequency(midi_note):
    return A4 * 2 ** ((midi_note - 69) / 12.0)


@dataclass
class Timbre:
    harmonics: List[Tuple[float, float]]
    decay_sigma0: float = 0.0
    decay_sigma1: float = 0.0
    decay_sigma2: float = 0.0
    inharmonicity: float = 0.0
    inharmonicity_stretch: float = 0.0
    attack: float = 0.005
    decay1: float = 0.4
    decay1_level: float = 0.35
    decay2: float = 2.0
    decay2_slow: float = 0.0
    prompt_ratio: float = 1.0
    release: float = 0.3
    vel_attack_scale: float = 0.5
    vel_harmonic_boost: float = 0.4
    vel_decay_scale: float = 0.3
    vel_noise_boost: float = 0.5
    vel_brightness: float = 0.0
    pitch_brightness: float = -0.5
    pitch_attack_add: float = 0.0
    pitch_sustain_add: float = 0.0
    pitch_decay2_mult: float = 1.0
    pitch_low_warmth: float = 0.0
    filter_cutoff: float = 18000.0
    detune_cents: float = 0.0
    detune_mix: float = 0.0
    vibrato_rate: float = 5.0
    vibrato_depth: float = 0.0
    vibrato_delay: float = 0.3
    tremolo_rate: float = 0.0
    tremolo_depth: float = 0.0
    sub_osc: float = 0.0
    noise_level: float = 0.0
    noise_decay: float = 30.0
    soundboard: float = 0.0
    liveliness: float = 0.0


TIMBRES = {
    "piano": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.48), (3, 0.24), (4, 0.12), (5, 0.07),
            (6, 0.04), (7, 0.022), (8, 0.013), (9, 0.008), (10, 0.005),
        ],
        decay_sigma0=1.0, decay_sigma1=3.5e-3, decay_sigma2=2e-7,
        inharmonicity=0.0001, inharmonicity_stretch=5.0,
        attack=0.003,
        decay1=0.40, decay1_level=0.32,
        decay2=2.8, decay2_slow=8.0, prompt_ratio=0.65,
        release=0.50,
        vel_attack_scale=0.6, vel_harmonic_boost=0.45,
        vel_decay_scale=0.3, vel_noise_boost=0.8, vel_brightness=0.18,
        pitch_brightness=-0.6,
        pitch_attack_add=0.012, pitch_sustain_add=0.25, pitch_decay2_mult=2.0,
        pitch_low_warmth=0.10,
        filter_cutoff=16000,
        detune_cents=1.0, detune_mix=0.04,
        noise_level=0.008, noise_decay=55.0,
        soundboard=0.15, liveliness=0.006,
    ),
    "electric_piano": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.52), (3, 0.08), (4, 0.20), (5, 0.03),
            (7, 0.07), (9, 0.02),
        ],
        decay_sigma0=1.2, decay_sigma1=4e-3,
        attack=0.002,
        decay1=0.22, decay1_level=0.25,
        decay2=1.6, decay2_slow=4.0, prompt_ratio=0.70,
        release=0.30,
        vel_attack_scale=0.4, vel_harmonic_boost=0.35,
        vel_decay_scale=0.2, vel_noise_boost=0.3, vel_brightness=0.15,
        pitch_brightness=-0.4,
        pitch_attack_add=0.006, pitch_sustain_add=0.15, pitch_decay2_mult=1.5,
        filter_cutoff=10000,
        detune_cents=2.0, detune_mix=0.06,
        tremolo_rate=4.5, tremolo_depth=0.05,
        noise_level=0.004, noise_decay=70.0,
    ),
    "organ": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.6), (3, 0.35), (4, 0.18), (5, 0.05),
            (6, 0.08), (8, 0.04),
        ],
        attack=0.008, decay1=0.02, decay1_level=0.92,
        decay2=0.02, release=0.08,
        vel_attack_scale=0.1, vel_harmonic_boost=0.15,
        pitch_brightness=-0.2, filter_cutoff=6000,
        vibrato_rate=6.0, vibrato_depth=8, vibrato_delay=0.1,
    ),
    "harpsichord": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.65), (3, 0.40), (4, 0.25), (5, 0.15),
            (6, 0.10), (7, 0.06), (8, 0.04),
        ],
        decay_sigma0=1.5, decay_sigma1=6e-3,
        attack=0.001, decay1=0.08, decay1_level=0.12,
        decay2=2.0, decay2_slow=3.0, prompt_ratio=0.75, release=0.12,
        vel_attack_scale=0.2, vel_harmonic_boost=0.15, vel_noise_boost=0.6,
        pitch_brightness=-0.3,
        pitch_attack_add=0.004, pitch_sustain_add=0.08, pitch_decay2_mult=1.3,
        filter_cutoff=14000,
        noise_level=0.020, noise_decay=90.0, soundboard=0.06,
    ),
    "guitar": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.42), (3, 0.24), (4, 0.12), (5, 0.07),
            (6, 0.035), (7, 0.018),
        ],
        decay_sigma0=1.2, decay_sigma1=4e-3,
        attack=0.001, decay1=0.12, decay1_level=0.18,
        decay2=3.0, decay2_slow=5.0, prompt_ratio=0.70, release=0.20,
        vel_attack_scale=0.5, vel_harmonic_boost=0.40,
        vel_noise_boost=0.7, vel_brightness=0.20,
        pitch_brightness=-0.5,
        pitch_attack_add=0.005, pitch_sustain_add=0.12, pitch_decay2_mult=1.5,
        filter_cutoff=9000,
        detune_cents=0.6, detune_mix=0.03,
        noise_level=0.014, noise_decay=80.0,
        soundboard=0.08, liveliness=0.005,
    ),
    "nylon_guitar": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08), (5, 0.04), (6, 0.02),
        ],
        decay_sigma0=1.0, decay_sigma1=3e-3,
        attack=0.002, decay1=0.15, decay1_level=0.15,
        decay2=2.5, decay2_slow=4.0, prompt_ratio=0.72, release=0.18,
        vel_attack_scale=0.4, vel_harmonic_boost=0.30,
        vel_noise_boost=0.5, vel_brightness=0.15,
        pitch_brightness=-0.4,
        pitch_attack_add=0.005, pitch_sustain_add=0.10, pitch_decay2_mult=1.4,
        filter_cutoff=7000,
        noise_level=0.010, noise_decay=60.0,
        soundboard=0.06, liveliness=0.004,
    ),
    "bass": Timbre(
        harmonics=[(1, 1.0), (2, 0.55), (3, 0.20), (4, 0.08), (5, 0.03)],
        decay_sigma0=0.8, decay_sigma1=3e-3, sub_osc=0.28,
        attack=0.003, decay1=0.10, decay1_level=0.40,
        decay2=1.0, release=0.08,
        vel_attack_scale=0.3, vel_harmonic_boost=0.30,
        vel_noise_boost=0.4, vel_brightness=0.12,
        pitch_brightness=-0.3,
        pitch_attack_add=0.008, pitch_sustain_add=0.15, pitch_decay2_mult=1.3,
        filter_cutoff=3500, noise_level=0.005, noise_decay=50.0,
    ),
    "strings": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.30), (3, 0.18), (4, 0.09), (5, 0.05), (6, 0.025),
        ],
        decay_sigma0=0.15, decay_sigma1=3e-4,
        attack=0.08, decay1=0.10, decay1_level=0.75,
        decay2=0.08, release=0.30,
        vel_attack_scale=0.3, vel_harmonic_boost=0.25,
        pitch_brightness=-0.3, filter_cutoff=8000,
        detune_cents=3.5, detune_mix=0.12,
        vibrato_rate=5.5, vibrato_depth=9, vibrato_delay=0.45,
        noise_level=0.003, noise_decay=5.0, liveliness=0.012,
    ),
    "cello": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.38), (3, 0.22), (4, 0.14), (5, 0.08),
            (6, 0.04), (7, 0.02),
        ],
        decay_sigma0=0.12, decay_sigma1=2.5e-4,
        attack=0.06, decay1=0.08, decay1_level=0.78,
        decay2=0.06, release=0.25,
        vel_attack_scale=0.35, vel_harmonic_boost=0.30,
        pitch_brightness=-0.25, filter_cutoff=6000,
        detune_cents=2.5, detune_mix=0.08,
        vibrato_rate=5.0, vibrato_depth=10, vibrato_delay=0.35,
        noise_level=0.005, noise_decay=4.0, liveliness=0.010,
    ),
    "brass": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.44), (3, 0.30), (4, 0.20), (5, 0.12),
            (6, 0.07), (7, 0.04), (8, 0.02),
        ],
        decay_sigma0=0.18, decay_sigma1=5e-4,
        attack=0.030, decay1=0.10, decay1_level=0.65,
        decay2=0.15, release=0.12,
        vel_attack_scale=0.6, vel_harmonic_boost=0.50, vel_brightness=0.25,
        pitch_brightness=-0.3, filter_cutoff=5000,
        detune_cents=0.8, detune_mix=0.03,
        vibrato_rate=5.0, vibrato_depth=7, vibrato_delay=0.3,
        noise_level=0.006, noise_decay=15.0, liveliness=0.008,
    ),
    "woodwind": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.12), (3, 0.24), (4, 0.03), (5, 0.08), (7, 0.02),
        ],
        decay_sigma0=0.12, decay_sigma1=2e-4,
        attack=0.020, decay1=0.06, decay1_level=0.62,
        decay2=0.10, release=0.15,
        vel_attack_scale=0.3, vel_harmonic_boost=0.25,
        vel_noise_boost=0.5, vel_brightness=0.12,
        pitch_brightness=-0.2, filter_cutoff=7000,
        detune_cents=1.2, detune_mix=0.04,
        vibrato_rate=5.5, vibrato_depth=11, vibrato_delay=0.20,
        noise_level=0.014, noise_decay=4.0, liveliness=0.010,
    ),
    "flute": Timbre(
        harmonics=[(1, 1.0), (2, 0.08), (3, 0.04), (4, 0.01)],
        decay_sigma0=0.06, decay_sigma1=8e-5,
        attack=0.030, decay1=0.05, decay1_level=0.70,
        decay2=0.06, release=0.12,
        vel_attack_scale=0.25, vel_harmonic_boost=0.20,
        vel_noise_boost=0.6, vel_brightness=0.10,
        pitch_brightness=-0.15, filter_cutoff=8000,
        vibrato_rate=5.0, vibrato_depth=10, vibrato_delay=0.25,
        noise_level=0.018, noise_decay=3.0, liveliness=0.008,
    ),
    "choir": Timbre(
        harmonics=[(1, 1.0), (2, 0.25), (3, 0.15), (4, 0.06), (5, 0.03)],
        decay_sigma0=0.05, decay_sigma1=5e-5,
        attack=0.12, decay1=0.10, decay1_level=0.78,
        decay2=0.05, release=0.35,
        vel_attack_scale=0.2, vel_harmonic_boost=0.15,
        pitch_brightness=-0.3, filter_cutoff=5500,
        detune_cents=5.0, detune_mix=0.15,
        vibrato_rate=5.5, vibrato_depth=8, vibrato_delay=0.4,
        noise_level=0.006, noise_decay=3.0, liveliness=0.015,
    ),
    "celesta": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.10), (3, 0.55), (4, 0.04), (5, 0.20),
            (6, 0.02), (8, 0.06),
        ],
        decay_sigma0=2.0, decay_sigma1=7e-3,
        attack=0.001, decay1=0.10, decay1_level=0.10,
        decay2=1.8, decay2_slow=3.0, prompt_ratio=0.60, release=0.35,
        vel_attack_scale=0.3, vel_harmonic_boost=0.20, vel_brightness=0.12,
        pitch_brightness=-0.2, filter_cutoff=14000,
        detune_cents=1.0, detune_mix=0.03,
        noise_level=0.005, noise_decay=80.0,
    ),
    "vibraphone": Timbre(
        harmonics=[(1, 1.0), (4, 0.35), (10, 0.12), (3, 0.05)],
        decay_sigma0=0.6, decay_sigma1=2.5e-3,
        attack=0.001, decay1=0.15, decay1_level=0.20,
        decay2=3.5, decay2_slow=5.0, prompt_ratio=0.55, release=0.40,
        vel_attack_scale=0.4, vel_harmonic_boost=0.25,
        pitch_brightness=-0.3, filter_cutoff=12000,
        tremolo_rate=5.5, tremolo_depth=0.10,
        noise_level=0.008, noise_decay=70.0,
    ),
    "marimba": Timbre(
        harmonics=[(1, 1.0), (4, 0.22), (10, 0.06), (3, 0.03)],
        decay_sigma0=1.5, decay_sigma1=6e-3,
        attack=0.001, decay1=0.08, decay1_level=0.06,
        decay2=1.5, release=0.15,
        vel_attack_scale=0.5, vel_harmonic_boost=0.30,
        vel_noise_boost=0.5, vel_brightness=0.15,
        pitch_brightness=-0.4, filter_cutoff=10000,
        noise_level=0.012, noise_decay=80.0,
    ),
    "harp": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.35), (3, 0.18), (4, 0.08), (5, 0.04),
            (6, 0.02), (7, 0.01),
        ],
        decay_sigma0=1.0, decay_sigma1=4e-3, decay_sigma2=1.5e-7,
        attack=0.001, decay1=0.10, decay1_level=0.15,
        decay2=3.5, decay2_slow=5.0, prompt_ratio=0.65, release=0.30,
        vel_attack_scale=0.4, vel_harmonic_boost=0.30,
        vel_noise_boost=0.5, vel_brightness=0.15,
        pitch_brightness=-0.5,
        pitch_attack_add=0.004, pitch_sustain_add=0.10, pitch_decay2_mult=1.5,
        filter_cutoff=11000,
        detune_cents=0.5, detune_mix=0.02,
        noise_level=0.010, noise_decay=65.0,
        soundboard=0.05, liveliness=0.004,
    ),
    "synth_pad": Timbre(
        harmonics=[(1, 1.0), (2, 0.15), (3, 0.08), (4, 0.04), (5, 0.02)],
        decay_sigma0=0.05, decay_sigma1=3e-5,
        attack=0.15, decay1=0.12, decay1_level=0.72,
        decay2=0.03, release=0.45,
        vel_attack_scale=0.15, vel_harmonic_boost=0.15,
        pitch_brightness=-0.3, filter_cutoff=6000,
        detune_cents=5.5, detune_mix=0.16,
        vibrato_rate=4.0, vibrato_depth=5, vibrato_delay=0.6,
    ),
    "synth_lead": Timbre(
        harmonics=[(1, 1.0), (2, 0.25), (3, 0.12), (4, 0.06), (5, 0.03)],
        decay_sigma0=0.25, decay_sigma1=5e-4,
        attack=0.004, decay1=0.08, decay1_level=0.58,
        decay2=0.25, release=0.15,
        vel_attack_scale=0.4, vel_harmonic_boost=0.35, vel_brightness=0.15,
        pitch_brightness=-0.3, filter_cutoff=8000,
        detune_cents=2.5, detune_mix=0.08,
        vibrato_rate=5.5, vibrato_depth=9, vibrato_delay=0.15,
    ),
    "pluck": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.32), (3, 0.44), (4, 0.20), (5, 0.12),
            (6, 0.06), (7, 0.03),
        ],
        decay_sigma0=1.5, decay_sigma1=5e-3,
        attack=0.001, decay1=0.15, decay1_level=0.08,
        decay2=4.0, decay2_slow=6.0, prompt_ratio=0.70, release=0.25,
        vel_attack_scale=0.4, vel_harmonic_boost=0.35,
        vel_noise_boost=0.5, vel_brightness=0.15,
        pitch_brightness=-0.5,
        pitch_attack_add=0.003, pitch_sustain_add=0.08, pitch_decay2_mult=1.3,
        filter_cutoff=13000,
        detune_cents=1.2, detune_mix=0.04,
        noise_level=0.006, noise_decay=75.0,
    ),
    "default": Timbre(
        harmonics=[(1, 1.0), (2, 0.22), (3, 0.12), (4, 0.06), (5, 0.03)],
        decay_sigma0=0.8, decay_sigma1=3e-3,
        attack=0.005, decay1=0.20, decay1_level=0.30,
        decay2=1.5, release=0.20,
        vel_attack_scale=0.3, vel_harmonic_boost=0.25,
        vel_noise_boost=0.3, vel_brightness=0.10,
        pitch_brightness=-0.4,
        pitch_attack_add=0.005, pitch_sustain_add=0.10, pitch_decay2_mult=1.3,
        filter_cutoff=10000,
        detune_cents=1.2, detune_mix=0.05,
        noise_level=0.004, noise_decay=40.0,
    ),
}

PROGRAM_MAP = {}
for _r, _name in [
    (range(0, 6), "piano"), (range(6, 8), "harpsichord"),
    (range(8, 16), "electric_piano"), (range(16, 24), "organ"),
    (range(24, 26), "nylon_guitar"), (range(26, 32), "guitar"),
    (range(32, 40), "bass"), (range(40, 46), "strings"),
    (range(46, 48), "harp"), (range(48, 52), "strings"),
    (range(52, 56), "choir"), (range(56, 64), "brass"),
    (range(64, 72), "woodwind"), (range(72, 76), "flute"),
    (range(76, 80), "woodwind"), (range(80, 88), "synth_lead"),
    (range(88, 100), "synth_pad"), (range(100, 104), "synth_lead"),
    (range(104, 108), "guitar"), (range(108, 112), "celesta"),
    (range(112, 116), "marimba"), (range(116, 120), "pluck"),
    (range(120, 128), "default"),
]:
    for _p in _r:
        PROGRAM_MAP[_p] = _name

TRACK_PANNING = {
    "piano": 0, "electric_piano": -10, "organ": 5, "harpsichord": -8,
    "guitar": -22, "nylon_guitar": -18, "bass": 0,
    "strings": 18, "cello": 12, "brass": 25, "woodwind": -18, "flute": -15,
    "choir": 0, "celesta": 10, "vibraphone": -14, "marimba": -10, "harp": 15,
    "synth_pad": 0, "synth_lead": 8, "pluck": -12, "default": 0, "drums": 0,
}

_REF_AMPS = {name: sum(a for _, a in t.harmonics) for name, t in TIMBRES.items()}


def harmonic_envelope(sigma, duration, n_samples, freq, actual_mult):
    delta_m = max(actual_mult - 1.0, 0.0)
    f_n = freq * actual_mult
    inv_tau = sigma[0] * delta_m + sigma[1] * (f_n - freq) + sigma[2] * (f_n ** 2 - freq ** 2)
    if inv_tau < 0.005:
        return np.ones(n_samples)
    return np.exp(-inv_tau * np.linspace(0, duration, n_samples, endpoint=False))


def harmonic_amplitude(base_amp, idx, vc, timbre, freq):
    pf = pitch_register(freq)
    lf = pitch_low_factor(freq)
    ps = max(1.0 + timbre.pitch_brightness * pf * idx * 0.15, 0.0)
    vs = min(1.0 + timbre.vel_harmonic_boost * vc * idx * 0.1, 2.0)
    if timbre.pitch_low_warmth > 0 and lf > 0.5 and 1 <= idx <= 3:
        ps *= 1.0 + timbre.pitch_low_warmth * (lf - 0.5) * 0.8
    return base_amp * ps * vs


def build_envelope(timbre, duration, n_samples, velocity, freq):
    lf = pitch_low_factor(freq)
    vc = vel_curve(velocity)
    at = min(timbre.attack * (1.0 - timbre.vel_attack_scale * vc)
             + timbre.pitch_attack_add * lf, duration * 0.25)
    d1t = min(timbre.decay1, duration * 0.4)
    rt = min(timbre.release, duration * 0.25)
    a_n = int(SAMPLE_RATE * at)
    d1_n = int(SAMPLE_RATE * d1t)
    r_n = int(SAMPLE_RATE * rt)
    r_start = max(0, n_samples - r_n)
    env = np.zeros(n_samples)

    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n) ** (0.5 + 0.3 * (1 - vc) + 0.2 * lf)

    if d1_n > 0:
        d1_level = min(timbre.decay1_level + (1 - timbre.decay1_level) * vc * 0.15
                       + timbre.pitch_sustain_add * lf, 0.95)
        env[a_n:a_n + d1_n] = d1_level + (1.0 - d1_level) * np.exp(
            -3.0 * (1.0 - 0.4 * lf) * np.linspace(0, 1, d1_n))

    d2_start = a_n + d1_n
    d2_n = max(0, r_start - d2_start)
    if d2_n > 0:
        sl = env[d2_start - 1] if d2_start > 0 else timbre.decay1_level
        tau = max(timbre.decay2 * (1.0 + timbre.vel_decay_scale * vc), 0.05)
        tau *= 1.0 + (timbre.pitch_decay2_mult - 1.0) * lf
        d2_t = np.linspace(0, d2_n / SAMPLE_RATE, d2_n)
        if timbre.decay2_slow > 0 and timbre.prompt_ratio < 1.0:
            tau_s = timbre.decay2_slow * (1.0 + 0.5 * lf)
            pr = timbre.prompt_ratio
            env[d2_start:r_start] = sl * (pr * np.exp(-d2_t / tau)
                                          + (1 - pr) * np.exp(-d2_t / tau_s))
        else:
            env[d2_start:r_start] = sl * np.exp(-d2_t / tau)

    if r_n > 0 and r_start < n_samples:
        sv = env[r_start - 1] if r_start > 0 else 0
        env[r_start:] = sv * np.exp(-4.0 * np.linspace(0, 1, n_samples - r_start))

    if timbre.liveliness > 0 and n_samples > SAMPLE_RATE * 0.1:
        ta = np.linspace(0, duration, n_samples, endpoint=False)
        rs = np.random.RandomState(int(freq * 100 + velocity * 1000) % (2 ** 31))
        r1, r2 = 2.5 + rs.random() * 1.5, 4.0 + rs.random() * 2.0
        onset = np.clip(ta / 0.15, 0, 1)
        env *= 1.0 + timbre.liveliness * onset * (
            0.6 * np.sin(2 * np.pi * r1 * ta + rs.random() * 6.28) +
            0.4 * np.sin(2 * np.pi * r2 * ta + rs.random() * 6.28))

    ac = min(ANTICLICK_SAMPLES, n_samples)
    if ac > 1:
        env[-ac:] *= np.linspace(1, 0, ac) ** 0.5
    return env


def synthesize(freq, duration, velocity, timbre, timbre_name="default"):
    n_samples = int(SAMPLE_RATE * duration)
    if n_samples == 0:
        return np.zeros(0)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    vc = vel_curve(velocity)
    rng = np.random.RandomState(int(freq * 100 + velocity * 1000) % (2 ** 31))
    sigma = (timbre.decay_sigma0, timbre.decay_sigma1, timbre.decay_sigma2)

    if timbre.vibrato_depth > 0:
        vib_env = np.clip((t - timbre.vibrato_delay) / 0.15, 0, 1)
        vib = vib_env * timbre.vibrato_depth * np.sin(2 * np.pi * timbre.vibrato_rate * t)
        phase = 2 * np.pi * np.cumsum(freq * 2 ** (vib / 1200.0)) / SAMPLE_RATE
    else:
        phase = 2 * np.pi * freq * t

    B = 0.0
    if timbre.inharmonicity > 0:
        B = (timbre.inharmonicity * np.exp(timbre.inharmonicity_stretch * pitch_register(freq))
             if timbre.inharmonicity_stretch > 0
             else timbre.inharmonicity * (1.0 + 3.0 * pitch_low_factor(freq)))

    waveform = np.zeros(n_samples)
    for idx, (mult, amp) in enumerate(timbre.harmonics):
        am = mult * np.sqrt(1 + B * mult * mult) if B > 0 else mult
        if freq * am >= SAMPLE_RATE / 2:
            break
        ha = harmonic_amplitude(amp, idx, vc, timbre, freq)
        ha *= 1.0 + 0.03 * (rng.random() - 0.5)
        hp = rng.random() * 0.4 if B > 0 else 0.0
        he = harmonic_envelope(sigma, duration, n_samples, freq, am)
        waveform += ha * np.sin(phase * am + hp) * he

    if timbre.detune_cents > 0 and timbre.detune_mix > 0:
        lf_d = pitch_low_factor(freq)
        eff_c = timbre.detune_cents * (1.0 + lf_d)
        eff_m = timbre.detune_mix * (1.0 + 0.6 * lf_d)
        ratio = 2 ** (eff_c / 1200)
        n_det = min(len(timbre.harmonics), MAX_DETUNE_HARMONICS)
        for r in (ratio, 1.0 / ratio):
            dp = 2 * np.pi * freq * r * t
            for idx, (mult, amp) in enumerate(timbre.harmonics[:n_det]):
                am = mult * np.sqrt(1 + B * mult * mult) if B > 0 else mult
                if freq * r * am >= SAMPLE_RATE / 2:
                    break
                he = harmonic_envelope(sigma, duration, n_samples, freq, am)
                waveform += eff_m * 0.5 * amp * np.sin(dp * am + rng.random() * 6.28) * he

    if timbre.sub_osc > 0 and freq / 2 > 20:
        waveform += timbre.sub_osc * np.sin(2 * np.pi * (freq / 2) * t)

    if timbre.noise_level > 0:
        lf = pitch_low_factor(freq)
        nl = timbre.noise_level * (1.0 + timbre.vel_noise_boost * vc) * (1.0 - 0.4 * lf)
        nd = timbre.noise_decay * (1.0 + 0.5 * lf)
        noise = rng.randn(n_samples)
        if n_samples > 512:
            noise = sosfilt(_NOISE_BP_SOS, noise)
        waveform += nl * noise * np.exp(-nd * t)

    ref = _REF_AMPS.get(timbre_name, 1.0)
    if ref > 0:
        waveform /= ref

    fc = timbre.filter_cutoff * (1.0 - 0.3 * pitch_register(freq))
    if timbre.vel_brightness > 0:
        fc *= 1.0 + timbre.vel_brightness * vc
    if fc < SAMPLE_RATE * 0.4:
        waveform = butter_lpf(waveform, fc)

    if timbre.tremolo_depth > 0:
        waveform *= 1.0 - timbre.tremolo_depth * 0.5 * (
            1 + np.sin(2 * np.pi * timbre.tremolo_rate * t))

    return waveform * build_envelope(timbre, duration, n_samples, velocity, freq) * (0.4 + 0.6 * vc)


def synthesize_drum(note, duration, velocity):
    n = int(SAMPLE_RATE * min(duration, 0.6))
    if n == 0:
        return np.zeros(0)
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    vc = vel_curve(velocity)

    if note in (35, 36):
        sweep = 160 * np.exp(-35 * t) + 42
        body = np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE) * np.exp(-8 * t)
        wave = (body + np.random.randn(n) * np.exp(-80 * t) * 0.25
                + np.sin(2 * np.pi * 42 * t) * np.exp(-6 * t) * 0.45)
    elif note in (38, 40):
        wave = (np.sin(2 * np.pi * 185 * t) * np.exp(-20 * t) * 0.65
                + np.sin(2 * np.pi * 330 * t) * np.exp(-25 * t) * 0.25
                + np.random.randn(n) * np.exp(-14 * t) * 0.5)
    elif note in (42, 44):
        wave = (np.random.randn(n) * np.exp(-40 * t) * 0.4
                + np.sin(2 * np.pi * 6000 * t + np.sin(2 * np.pi * 8300 * t))
                * np.exp(-35 * t) * 0.12)
    elif note == 46:
        wave = (np.random.randn(n) * np.exp(-10 * t) * 0.35
                + np.sin(2 * np.pi * 6000 * t + np.sin(2 * np.pi * 8300 * t))
                * np.exp(-8 * t) * 0.18)
    elif note in (49, 57):
        wave = (np.random.randn(n) * np.exp(-4 * t) * 0.30
                + np.sin(2 * np.pi * 3500 * t + 2 * np.sin(2 * np.pi * 5100 * t))
                * np.exp(-5 * t) * 0.18)
    elif note in (51, 53, 59):
        wave = (np.sin(2 * np.pi * 4500 * t + 1.5 * np.sin(2 * np.pi * 6800 * t))
                * np.exp(-6 * t) * 0.25
                + np.random.randn(n) * np.exp(-18 * t) * 0.12)
    elif note in (47, 48, 50, 45, 43, 41):
        bf = 80 + (note - 41) * 18
        sweep = bf * 1.5 * np.exp(-15 * t) + bf
        wave = (np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE) * np.exp(-10 * t)
                + np.random.randn(n) * np.exp(-50 * t) * 0.12)
    elif note in (39, 37):
        noise = np.random.randn(n)
        wave = noise * (np.exp(-60 * t) * 0.3
                        + np.exp(-40 * np.maximum(t - 0.01, 0)) * 0.4
                        + np.exp(-20 * np.maximum(t - 0.02, 0)) * 0.5) * 0.25
    else:
        wave = np.sin(2 * np.pi * (200 + (note - 35) * 10) * t) * np.exp(-12 * t)

    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak
    ac = min(ANTICLICK_SAMPLES, n)
    if ac > 1:
        wave[-ac:] *= np.linspace(1, 0, ac) ** 0.5
    return wave * (0.4 + 0.6 * vc)


def apply_soundboard(signal, amount=0.1):
    if amount <= 0:
        return signal
    board = np.zeros_like(signal)
    for f, r, g in [(110.0, 0.997, 0.25), (220.0, 0.995, 0.18),
                     (175.0, 0.996, 0.15), (330.0, 0.993, 0.10)]:
        w0 = 2 * np.pi * f / SAMPLE_RATE
        board += lfilter([g * (1 - r * r), 0, 0], [1.0, -2 * r * np.cos(w0), r * r], signal)
    bp = np.max(np.abs(board))
    if bp > 1e-8:
        board *= np.max(np.abs(signal)) / bp
    return signal * (1.0 - amount) + board * amount


def generate_hrtf_pair(azimuth_deg, ir_length=128):
    az = np.radians(azimuth_deg)
    sin_az = abs(np.sin(az))
    itd = abs(HEAD_RADIUS / SPEED_OF_SOUND * (az + np.sin(az))) if sin_az > 1e-6 else 0.0
    ild_db = 1.5 * np.sin(az)

    near_ir = np.zeros(ir_length)
    far_ir = np.zeros(ir_length)
    near_ir[0] = 10 ** (abs(ild_db) / 20)
    itd_sample = min(int(itd * SAMPLE_RATE), ir_length - 1)
    far_ir[itd_sample] = 10 ** (-abs(ild_db) / 20)

    if sin_az > 0.05:
        shadow_fc = np.clip(12000.0 / (1.0 + 2.5 * sin_az ** 2), 30, SAMPLE_RATE * 0.45)
        far_ir = sosfilt(
            butter(1, shadow_fc, btype='low', fs=SAMPLE_RATE, output='sos'), far_ir)

    return (far_ir, near_ir) if azimuth_deg >= 0 else (near_ir, far_ir)


def apply_hrtf(mono, azimuth_deg):
    l_ir, r_ir = generate_hrtf_pair(azimuth_deg)
    n = len(mono)
    return fftconvolve(mono, l_ir, mode='full')[:n], fftconvolve(mono, r_ir, mode='full')[:n]


def generate_room_ir(duration=0.9, room_size=0.40, damping=0.65):
    n = int(SAMPLE_RATE * duration)
    ir_l, ir_r = np.zeros(n), np.zeros(n)
    for i, (dms, g) in enumerate(zip(
            [1.5, 4.2, 8.0, 13.5, 20.0, 28.0, 38.0, 52.0],
            [0.55, 0.40, 0.28, 0.18, 0.11, 0.07, 0.04, 0.02])):
        d = int(SAMPLE_RATE * dms * room_size / 1000)
        if d < n:
            ir_l[d] += g * (1.0 + 0.06 * np.sin(i * 1.7))
            ir_r[min(d + int(SAMPLE_RATE * 0.00025 * (i % 3)), n - 1)] += g * (
                1.0 - 0.06 * np.sin(i * 1.7))
    ls = int(SAMPLE_RATE * 0.06 * room_size)
    ln = n - ls
    if ln > 0:
        env = np.exp(-4.5 / (duration * room_size + 0.01) * np.linspace(0, duration, ln))
        env *= 1 - np.exp(-np.linspace(0, 8, ln))
        sos = butter(2, 1200 + 5000 * (1 - damping), btype='low', fs=SAMPLE_RATE, output='sos')
        np.random.seed(42)
        ir_l[ls:] += sosfilt(sos, np.random.randn(ln) * env * 0.06)
        np.random.seed(137)
        ir_r[ls:] += sosfilt(sos, np.random.randn(ln) * env * 0.06)
    peak = max(np.max(np.abs(ir_l)), np.max(np.abs(ir_r)), 1e-10)
    return ir_l / peak, ir_r / peak


def stereo_compressor(left, right, threshold=0.55, ratio=2.5, attack_ms=5, release_ms=60):
    n = len(left)
    linked = np.maximum(np.abs(left), np.abs(right))
    ca = np.exp(-1.0 / max(int(SAMPLE_RATE * attack_ms / 1000), 1))
    cr = np.exp(-1.0 / max(int(SAMPLE_RATE * release_ms / 1000), 1))
    env = np.empty(n)
    env[0] = linked[0]
    for i in range(1, n):
        c = ca if linked[i] > env[i - 1] else cr
        env[i] = c * env[i - 1] + (1 - c) * linked[i]
    gain = np.ones(n)
    mask = env > threshold
    gain[mask] = (threshold + (env[mask] - threshold) / ratio) / env[mask]
    return left * gain, right * gain


def collect_tempo_map(mid):
    events = []
    for track in mid.tracks:
        ticks = 0
        for msg in track:
            ticks += msg.time
            if msg.type == "set_tempo":
                events.append((ticks, msg.tempo))
    events.sort(key=lambda x: x[0])
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    return events


def ticks_to_seconds(abs_ticks, tempo_map, tpb):
    sec = 0.0
    prev_tick, prev_tempo = 0, tempo_map[0][1]
    for tick, tempo in tempo_map[1:]:
        if tick >= abs_ticks:
            break
        sec += (tick - prev_tick) / tpb * (prev_tempo / 1_000_000)
        prev_tick, prev_tempo = tick, tempo
    return sec + (abs_ticks - prev_tick) / tpb * (prev_tempo / 1_000_000)


def parse_midi(midi_file):
    mid = mido.MidiFile(midi_file)
    tempo_map = collect_tempo_map(mid)
    tracks, track_pans = [], []
    for track in mid.tracks:
        notes, active, active_vel = [], {}, {}
        pedal_on, pedal_held = {}, {}
        abs_ticks = 0
        channel_program, channel_pan = {}, {}
        for msg in track:
            abs_ticks += msg.time
            t = ticks_to_seconds(abs_ticks, tempo_map, mid.ticks_per_beat)
            if msg.type == "program_change":
                channel_program[msg.channel] = msg.program
            elif msg.type == "control_change":
                if msg.control == 64:
                    ch = msg.channel
                    if msg.value >= 64:
                        pedal_on[ch] = True
                    else:
                        pedal_on[ch] = False
                        for nn, st, vl in pedal_held.pop(ch, []):
                            notes.append((st, nn, min(max(t - st, 0.01), MAX_PEDAL_SUSTAIN),
                                          vl, ch, channel_program.get(ch, 0)))
                elif msg.control == 10:
                    channel_pan[msg.channel] = msg.value
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = t
                active_vel[(msg.channel, msg.note)] = msg.velocity / 127
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key), active_vel.pop(key)
                    if pedal_on.get(msg.channel, False):
                        pedal_held.setdefault(msg.channel, []).append((msg.note, start, vel))
                    else:
                        notes.append((start, msg.note, max(t - start, 0.01), vel,
                                      msg.channel, channel_program.get(msg.channel, 0)))
        end_t = ticks_to_seconds(abs_ticks, tempo_map, mid.ticks_per_beat)
        for ch in pedal_held:
            for nn, st, vl in pedal_held[ch]:
                notes.append((st, nn, min(max(end_t - st, 0.01), MAX_PEDAL_SUSTAIN),
                              vl, ch, channel_program.get(ch, 0)))
        for (ch_id, nn), st in list(active.items()):
            notes.append((st, nn, max(end_t - st, 0.01), active_vel[(ch_id, nn)],
                          ch_id, channel_program.get(ch_id, 0)))
        if notes:
            tracks.append(notes)
            track_pans.append(channel_pan.get(notes[0][4]))
    return tracks, track_pans


def render_tracks(tracks, track_pans, output_file):
    max_end = 0
    for notes in tracks:
        for start, note, dur, vel, ch, prog in notes:
            end = int((start + dur) * SAMPLE_RATE) + SAMPLE_RATE
            if end > max_end:
                max_end = end
    max_end += SAMPLE_RATE
    print(f"[INFO] {len(tracks)} tracks, {max_end / SAMPLE_RATE:.1f}s")

    ir_l, ir_r = generate_room_ir()
    master_l, master_r = np.zeros(max_end), np.zeros(max_end)

    for ti, notes in enumerate(tracks):
        buf = np.zeros(max_end)
        timbre_name, is_drum, sb_amount = "default", False, 0.0
        if notes:
            _, _, _, _, ch0, prog0 = notes[0]
            if ch0 == 9:
                timbre_name, is_drum = "drums", True
            else:
                timbre_name = PROGRAM_MAP.get(prog0, "default")
                sb_amount = TIMBRES.get(timbre_name, TIMBRES["default"]).soundboard
        print(f"  Track {ti}: {len(notes)} notes [{timbre_name}]")

        for start, note, dur, vel, ch, prog in notes:
            if ch == 9:
                tone = synthesize_drum(note, dur, vel)
            else:
                tn = PROGRAM_MAP.get(prog, "default")
                tone = synthesize(note_to_frequency(note), dur, vel, TIMBRES[tn], tn)
            s = int(start * SAMPLE_RATE)
            e = min(s + len(tone), max_end)
            buf[s:e] += tone[:e - s]

        if sb_amount > 0:
            buf = apply_soundboard(buf, sb_amount)

        rms = np.sqrt(np.mean(buf ** 2))
        if rms > 1e-6:
            buf *= 0.12 / rms
        peak = np.max(np.abs(buf))
        if peak > 1.0:
            buf /= peak

        pan_cc = track_pans[ti] if ti < len(track_pans) else None
        if pan_cc is not None:
            azimuth = (pan_cc - 64) / 64.0 * 55.0
        else:
            azimuth = TRACK_PANNING.get(timbre_name, 0)

        left, right = apply_hrtf(buf, azimuth)
        master_l += left
        master_r += right

    if tracks:
        master_l, master_r = stereo_compressor(master_l, master_r)
        master_l = master_l * 0.95 + fftconvolve(master_l, ir_l, mode='full')[:max_end] * 0.05
        master_r = master_r * 0.95 + fftconvolve(master_r, ir_r, mode='full')[:max_end] * 0.05
        peak = max(np.max(np.abs(master_l)), np.max(np.abs(master_r)), 1e-10)
        master_l *= 0.92 / peak
        master_r *= 0.92 / peak

    stereo = np.clip(np.column_stack((master_l, master_r)), -1.0, 1.0)
    sf.write(output_file, stereo, SAMPLE_RATE, format="FLAC", subtype="PCM_24")
    print(f"[OK] {output_file}")


def main():
    parser = argparse.ArgumentParser(description="MIDI Synthesizer")
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    tracks, pans = parse_midi(args.input)
    render_tracks(tracks, pans, args.output or os.path.splitext(args.input)[0] + ".flac")


if __name__ == "__main__":
    main()