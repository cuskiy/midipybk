import numpy as np
import soundfile as sf
import mido
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
from scipy.signal import fftconvolve, butter, sosfilt, lfilter

A4 = 440.0
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SAMPLE_RATE = 44100
HEAD_RADIUS = 0.0875
SPEED_OF_SOUND = 343.0
MAX_PEDAL_SUSTAIN = 5.0
ANTICLICK_SAMPLES = 128
VEL_CURVE_EXP = 0.65


def generate_note_frequencies():
    nf = {}
    for octave in range(-1, 10):
        for i, note in enumerate(NOTE_NAMES):
            n = (octave * 12) + i
            nf[f"{note}{octave}"] = A4 * 2 ** ((n - 69) / 12.0)
    return nf


NOTE_FREQUENCIES = generate_note_frequencies()


def butter_lpf(signal, cutoff, order=2):
    cutoff = np.clip(cutoff, 30, SAMPLE_RATE * 0.45)
    sos = butter(order, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, signal)


def pitch_register(frequency):
    return np.clip((np.log2(frequency / 261.63) + 2) / 4, 0.0, 1.0)


def pitch_low_factor(frequency):
    return np.clip(1.0 - pitch_register(frequency), 0.0, 1.0)


def vel_curve(velocity):
    return velocity ** VEL_CURVE_EXP


@dataclass
class Timbre:
    harmonics: List[Tuple[float, float]]
    harmonic_decay_rate: float = 1.5
    inharmonicity: float = 0.0
    attack_brightness: float = 0.0
    attack_brightness_tau: float = 0.012

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

    pitch_brightness: float = -0.5
    pitch_decay_scale: float = 0.6
    pitch_attack_add: float = 0.0
    pitch_sustain_add: float = 0.0
    pitch_decay2_mult: float = 1.0
    pitch_low_warmth: float = 0.0

    filter_cutoff: float = 18000.0
    vel_brightness: float = 0.0

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
    release_damping: float = 0.0


TIMBRES = {
    "piano": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.15), (3, 0.07), (4, 0.03), (5, 0.015),
            (6, 0.008), (7, 0.004), (8, 0.002),
        ],
        harmonic_decay_rate=2.4,
        inharmonicity=0.00004,
        attack_brightness=1.8,
        attack_brightness_tau=0.012,
        attack=0.003,
        decay1=0.35, decay1_level=0.30,
        decay2=0.6, decay2_slow=2.8, prompt_ratio=0.72,
        release=0.50,
        vel_attack_scale=0.6,
        vel_harmonic_boost=0.45,
        vel_decay_scale=0.3,
        vel_noise_boost=0.8,
        pitch_brightness=-0.6,
        pitch_decay_scale=0.7,
        pitch_attack_add=0.014,
        pitch_sustain_add=0.28,
        pitch_decay2_mult=1.8,
        pitch_low_warmth=0.20,
        filter_cutoff=16000,
        vel_brightness=0.15,
        detune_cents=1.2, detune_mix=0.12,
        noise_level=0.015, noise_decay=100.0,
        soundboard=0.08,
        liveliness=0.008,
        release_damping=6.0,
    ),

    "electric_piano": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.52), (3, 0.08), (4, 0.20), (5, 0.03),
            (7, 0.07), (9, 0.02),
        ],
        harmonic_decay_rate=2.0,
        attack_brightness=1.5,
        attack_brightness_tau=0.010,
        attack=0.002,
        decay1=0.22, decay1_level=0.25,
        decay2=1.2, decay2_slow=4.0, prompt_ratio=0.65,
        release=0.30,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.35,
        vel_decay_scale=0.2,
        vel_noise_boost=0.3,
        pitch_brightness=-0.4,
        pitch_decay_scale=0.5,
        pitch_attack_add=0.006,
        pitch_sustain_add=0.15,
        pitch_decay2_mult=1.5,
        filter_cutoff=10000,
        detune_cents=2.0, detune_mix=0.06,
        tremolo_rate=4.5, tremolo_depth=0.05,
        noise_level=0.006, noise_decay=200.0,
        release_damping=4.0,
    ),

    "organ": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.6), (3, 0.35), (4, 0.18), (5, 0.05),
            (6, 0.08), (8, 0.04),
        ],
        harmonic_decay_rate=0.0,
        attack=0.008,
        decay1=0.02, decay1_level=0.92,
        decay2=0.02,
        release=0.08,
        vel_attack_scale=0.1,
        vel_harmonic_boost=0.15,
        pitch_brightness=-0.2,
        filter_cutoff=6000,
        vibrato_rate=6.0, vibrato_depth=8, vibrato_delay=0.1,
    ),

    "harpsichord": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.55), (3, 0.32), (4, 0.18), (5, 0.10),
            (6, 0.06), (7, 0.03), (8, 0.02),
        ],
        harmonic_decay_rate=3.2,
        attack_brightness=2.0,
        attack_brightness_tau=0.008,
        attack=0.001,
        decay1=0.08, decay1_level=0.12,
        decay2=1.5, decay2_slow=3.5, prompt_ratio=0.7,
        release=0.12,
        vel_attack_scale=0.2,
        vel_harmonic_boost=0.15,
        vel_noise_boost=0.6,
        pitch_brightness=-0.3,
        pitch_decay_scale=0.5,
        pitch_attack_add=0.004,
        pitch_sustain_add=0.08,
        pitch_decay2_mult=1.3,
        filter_cutoff=14000,
        noise_level=0.025, noise_decay=250.0,
        soundboard=0.04,
        release_damping=5.0,
    ),

    "guitar": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.30), (3, 0.15), (4, 0.07), (5, 0.04),
            (6, 0.02), (7, 0.01),
        ],
        harmonic_decay_rate=2.5,
        attack_brightness=1.8,
        attack_brightness_tau=0.010,
        attack=0.001,
        decay1=0.12, decay1_level=0.18,
        decay2=2.0, decay2_slow=5.0, prompt_ratio=0.65,
        release=0.20,
        vel_attack_scale=0.5,
        vel_harmonic_boost=0.40,
        vel_noise_boost=0.7,
        pitch_brightness=-0.5,
        pitch_decay_scale=0.6,
        pitch_attack_add=0.005,
        pitch_sustain_add=0.12,
        pitch_decay2_mult=1.5,
        filter_cutoff=9000,
        detune_cents=0.6, detune_mix=0.03,
        noise_level=0.018, noise_decay=200.0,
        soundboard=0.05,
        liveliness=0.006,
        release_damping=4.0,
    ),

    "nylon_guitar": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.25), (3, 0.12), (4, 0.05), (5, 0.02),
            (6, 0.01),
        ],
        harmonic_decay_rate=2.0,
        attack_brightness=1.2,
        attack_brightness_tau=0.012,
        attack=0.002,
        decay1=0.15, decay1_level=0.15,
        decay2=1.8, decay2_slow=4.0, prompt_ratio=0.68,
        release=0.18,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.30,
        vel_noise_boost=0.5,
        pitch_brightness=-0.4,
        pitch_decay_scale=0.5,
        pitch_attack_add=0.005,
        pitch_sustain_add=0.10,
        pitch_decay2_mult=1.4,
        filter_cutoff=7000,
        noise_level=0.012, noise_decay=180.0,
        soundboard=0.04,
        liveliness=0.005,
        release_damping=3.5,
    ),

    "bass": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.45), (3, 0.18), (4, 0.07), (5, 0.03),
            (6, 0.015),
        ],
        harmonic_decay_rate=1.5,
        sub_osc=0.25,
        attack_brightness=0.8,
        attack_brightness_tau=0.015,
        attack=0.004,
        decay1=0.10, decay1_level=0.40,
        decay2=1.1,
        release=0.08,
        vel_attack_scale=0.3,
        vel_harmonic_boost=0.28,
        vel_noise_boost=0.3,
        pitch_brightness=-0.3,
        pitch_decay_scale=0.3,
        pitch_attack_add=0.006,
        pitch_sustain_add=0.12,
        pitch_decay2_mult=1.3,
        filter_cutoff=3200,
        noise_level=0.005, noise_decay=120.0,
        liveliness=0.004,
    ),

    "strings": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.35), (3, 0.25), (4, 0.14), (5, 0.09),
            (6, 0.06), (7, 0.035), (8, 0.02),
        ],
        harmonic_decay_rate=0.2,
        attack_brightness=0.5,
        attack_brightness_tau=0.025,
        attack=0.07,
        decay1=0.10, decay1_level=0.78,
        decay2=0.06,
        release=0.28,
        vel_attack_scale=0.35,
        vel_harmonic_boost=0.30,
        pitch_brightness=-0.25,
        pitch_decay_scale=0.08,
        filter_cutoff=9000,
        detune_cents=3.5, detune_mix=0.12,
        vibrato_rate=5.5, vibrato_depth=12, vibrato_delay=0.35,
        noise_level=0.004, noise_decay=4.0,
        liveliness=0.025,
    ),

    "cello": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.42), (3, 0.28), (4, 0.18), (5, 0.11),
            (6, 0.06), (7, 0.035), (8, 0.02),
        ],
        harmonic_decay_rate=0.18,
        attack_brightness=0.6,
        attack_brightness_tau=0.020,
        attack=0.055,
        decay1=0.08, decay1_level=0.80,
        decay2=0.05,
        release=0.22,
        vel_attack_scale=0.40,
        vel_harmonic_boost=0.35,
        pitch_brightness=-0.20,
        pitch_decay_scale=0.08,
        filter_cutoff=6500,
        detune_cents=2.5, detune_mix=0.08,
        vibrato_rate=5.0, vibrato_depth=14, vibrato_delay=0.30,
        noise_level=0.006, noise_decay=3.5,
        liveliness=0.022,
    ),

    "brass": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.44), (3, 0.30), (4, 0.20), (5, 0.12),
            (6, 0.07), (7, 0.04), (8, 0.02),
        ],
        harmonic_decay_rate=0.35,
        attack_brightness=1.0,
        attack_brightness_tau=0.018,
        attack=0.030,
        decay1=0.10, decay1_level=0.65,
        decay2=0.15,
        release=0.12,
        vel_attack_scale=0.6,
        vel_harmonic_boost=0.50,
        pitch_brightness=-0.3,
        pitch_decay_scale=0.15,
        filter_cutoff=5000,
        detune_cents=0.8, detune_mix=0.03,
        vibrato_rate=5.0, vibrato_depth=7, vibrato_delay=0.3,
        noise_level=0.006, noise_decay=15.0,
        liveliness=0.012,
    ),

    "woodwind": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.12), (3, 0.24), (4, 0.03), (5, 0.08),
            (7, 0.02),
        ],
        harmonic_decay_rate=0.25,
        attack_brightness=0.8,
        attack_brightness_tau=0.015,
        attack=0.020,
        decay1=0.06, decay1_level=0.62,
        decay2=0.10,
        release=0.15,
        vel_attack_scale=0.3,
        vel_harmonic_boost=0.25,
        vel_noise_boost=0.5,
        pitch_brightness=-0.2,
        pitch_decay_scale=0.1,
        filter_cutoff=7000,
        detune_cents=1.2, detune_mix=0.04,
        vibrato_rate=5.5, vibrato_depth=11, vibrato_delay=0.20,
        noise_level=0.014, noise_decay=4.0,
        liveliness=0.015,
    ),

    "flute": Timbre(
        harmonics=[(1, 1.0), (2, 0.08), (3, 0.04), (4, 0.01)],
        harmonic_decay_rate=0.15,
        attack_brightness=0.6,
        attack_brightness_tau=0.020,
        attack=0.030,
        decay1=0.05, decay1_level=0.70,
        decay2=0.06,
        release=0.12,
        vel_attack_scale=0.25,
        vel_harmonic_boost=0.20,
        vel_noise_boost=0.6,
        pitch_brightness=-0.15,
        filter_cutoff=8000,
        vibrato_rate=5.0, vibrato_depth=10, vibrato_delay=0.25,
        noise_level=0.018, noise_decay=3.0,
        liveliness=0.012,
    ),

    "choir": Timbre(
        harmonics=[(1, 1.0), (2, 0.25), (3, 0.15), (4, 0.06), (5, 0.03)],
        harmonic_decay_rate=0.1,
        attack=0.12,
        decay1=0.10, decay1_level=0.78,
        decay2=0.05,
        release=0.35,
        vel_attack_scale=0.2,
        vel_harmonic_boost=0.15,
        pitch_brightness=-0.3,
        filter_cutoff=5500,
        detune_cents=5.0, detune_mix=0.15,
        vibrato_rate=5.5, vibrato_depth=8, vibrato_delay=0.4,
        noise_level=0.006, noise_decay=3.0,
        liveliness=0.020,
    ),

    "celesta": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.10), (3, 0.55), (4, 0.04), (5, 0.20),
            (6, 0.02), (8, 0.06),
        ],
        harmonic_decay_rate=3.5,
        attack_brightness=1.5,
        attack_brightness_tau=0.008,
        attack=0.001,
        decay1=0.10, decay1_level=0.10,
        decay2=1.5, decay2_slow=3.0, prompt_ratio=0.6,
        release=0.35,
        vel_attack_scale=0.3,
        vel_harmonic_boost=0.20,
        pitch_brightness=-0.2,
        pitch_decay_scale=0.4,
        filter_cutoff=14000,
        detune_cents=1.0, detune_mix=0.03,
        noise_level=0.005, noise_decay=150.0,
        release_damping=3.0,
    ),

    "vibraphone": Timbre(
        harmonics=[(1, 1.0), (4, 0.35), (10, 0.12), (3, 0.05)],
        harmonic_decay_rate=1.2,
        attack_brightness=1.2,
        attack_brightness_tau=0.010,
        attack=0.001,
        decay1=0.15, decay1_level=0.20,
        decay2=2.5, decay2_slow=5.0, prompt_ratio=0.55,
        release=0.40,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.25,
        pitch_brightness=-0.3,
        pitch_decay_scale=0.4,
        filter_cutoff=12000,
        tremolo_rate=5.5, tremolo_depth=0.10,
        noise_level=0.008, noise_decay=120.0,
    ),

    "marimba": Timbre(
        harmonics=[(1, 1.0), (4, 0.22), (10, 0.06), (3, 0.03)],
        harmonic_decay_rate=3.0,
        attack_brightness=1.5,
        attack_brightness_tau=0.008,
        attack=0.001,
        decay1=0.08, decay1_level=0.06,
        decay2=1.5,
        release=0.15,
        vel_attack_scale=0.5,
        vel_harmonic_boost=0.30,
        vel_noise_boost=0.5,
        pitch_brightness=-0.4,
        pitch_decay_scale=0.5,
        filter_cutoff=10000,
        noise_level=0.015, noise_decay=200.0,
    ),

    "harp": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.22), (3, 0.10), (4, 0.05), (5, 0.02),
            (6, 0.01),
        ],
        harmonic_decay_rate=2.5,
        inharmonicity=0.00002,
        attack_brightness=2.0,
        attack_brightness_tau=0.010,
        attack=0.001,
        decay1=0.10, decay1_level=0.15,
        decay2=2.0, decay2_slow=5.0, prompt_ratio=0.6,
        release=0.30,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.30,
        vel_noise_boost=0.5,
        pitch_brightness=-0.5,
        pitch_decay_scale=0.6,
        pitch_attack_add=0.004,
        pitch_sustain_add=0.10,
        pitch_decay2_mult=1.5,
        filter_cutoff=11000,
        detune_cents=0.5, detune_mix=0.02,
        noise_level=0.012, noise_decay=180.0,
        soundboard=0.03,
        liveliness=0.005,
        release_damping=4.0,
    ),

    "synth_pad": Timbre(
        harmonics=[(1, 1.0), (2, 0.15), (3, 0.08), (4, 0.04), (5, 0.02)],
        harmonic_decay_rate=0.1,
        attack=0.15,
        decay1=0.12, decay1_level=0.72,
        decay2=0.03,
        release=0.45,
        vel_attack_scale=0.15,
        vel_harmonic_boost=0.15,
        pitch_brightness=-0.3,
        filter_cutoff=6000,
        detune_cents=5.5, detune_mix=0.16,
        vibrato_rate=4.0, vibrato_depth=5, vibrato_delay=0.6,
    ),

    "synth_lead": Timbre(
        harmonics=[(1, 1.0), (2, 0.25), (3, 0.12), (4, 0.06), (5, 0.03)],
        harmonic_decay_rate=0.3,
        attack=0.004,
        decay1=0.08, decay1_level=0.58,
        decay2=0.25,
        release=0.15,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.35,
        pitch_brightness=-0.3,
        filter_cutoff=8000,
        detune_cents=2.5, detune_mix=0.08,
        vibrato_rate=5.5, vibrato_depth=9, vibrato_delay=0.15,
    ),

    "pluck": Timbre(
        harmonics=[
            (1, 1.0), (2, 0.25), (3, 0.35), (4, 0.14), (5, 0.08),
            (6, 0.04), (7, 0.02),
        ],
        harmonic_decay_rate=2.8,
        attack_brightness=2.2,
        attack_brightness_tau=0.008,
        attack=0.001,
        decay1=0.15, decay1_level=0.08,
        decay2=2.5, decay2_slow=5.5, prompt_ratio=0.65,
        release=0.25,
        vel_attack_scale=0.4,
        vel_harmonic_boost=0.35,
        vel_noise_boost=0.5,
        pitch_brightness=-0.5,
        pitch_attack_add=0.003,
        pitch_sustain_add=0.08,
        pitch_decay2_mult=1.3,
        filter_cutoff=13000,
        detune_cents=1.2, detune_mix=0.04,
        noise_level=0.008, noise_decay=200.0,
        release_damping=3.5,
    ),

    "default": Timbre(
        harmonics=[(1, 1.0), (2, 0.18), (3, 0.09), (4, 0.04), (5, 0.02)],
        harmonic_decay_rate=1.8,
        attack_brightness=1.0,
        attack_brightness_tau=0.012,
        attack=0.005,
        decay1=0.20, decay1_level=0.30,
        decay2=1.5,
        release=0.20,
        vel_attack_scale=0.3,
        vel_harmonic_boost=0.25,
        vel_noise_boost=0.3,
        pitch_brightness=-0.4,
        pitch_decay_scale=0.4,
        pitch_attack_add=0.005,
        pitch_sustain_add=0.10,
        pitch_decay2_mult=1.3,
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
    (range(32, 40), "bass"), (range(40, 44), "strings"),
    (range(44, 46), "strings"), (range(46, 48), "harp"),
    (range(48, 52), "strings"), (range(52, 56), "choir"),
    (range(56, 64), "brass"), (range(64, 68), "woodwind"),
    (range(68, 72), "woodwind"), (range(72, 76), "flute"),
    (range(76, 80), "woodwind"), (range(80, 88), "synth_lead"),
    (range(88, 96), "synth_pad"), (range(96, 100), "synth_pad"),
    (range(100, 104), "synth_lead"), (range(104, 108), "guitar"),
    (range(108, 112), "celesta"), (range(112, 116), "marimba"),
    (range(116, 120), "pluck"), (range(120, 128), "default"),
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


def build_envelope(timbre, duration, n_samples, velocity, frequency, note_off_sample=None):
    lf = pitch_low_factor(frequency)
    vc = vel_curve(velocity)

    at = timbre.attack * (1.0 - timbre.vel_attack_scale * vc) + timbre.pitch_attack_add * lf
    at = min(at, duration * 0.25)
    d1t = min(timbre.decay1, duration * 0.4)

    a_n = int(SAMPLE_RATE * at)
    d1_n = int(SAMPLE_RATE * d1t)
    d2_start = a_n + d1_n

    if note_off_sample is None:
        note_off_sample = n_samples
    r_start = min(note_off_sample, n_samples)

    env = np.zeros(n_samples)

    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n) ** (0.5 + 0.3 * (1 - vc) + 0.2 * lf)

    if d1_n > 0:
        d1_level = min(
            timbre.decay1_level + (1 - timbre.decay1_level) * vc * 0.15 + timbre.pitch_sustain_add * lf,
            0.95)
        env[a_n:a_n + d1_n] = d1_level + (1.0 - d1_level) * np.exp(
            -3.0 * (1.0 - 0.4 * lf) * np.linspace(0, 1, d1_n))

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

    if r_start < n_samples:
        sv = env[r_start - 1] if r_start > 0 else 0
        r_len = n_samples - r_start
        rel_t = np.arange(r_len, dtype=np.float64) / SAMPLE_RATE
        env[r_start:] = sv * np.exp(-15.0 * rel_t)

    if timbre.liveliness > 0 and n_samples > SAMPLE_RATE * 0.08:
        total_dur = n_samples / SAMPLE_RATE
        t = np.linspace(0, total_dur, n_samples, endpoint=False)
        seed = (int(frequency * 100 + velocity * 1000)) % (2 ** 31)
        rs = np.random.RandomState(seed)
        r1 = 2.0 + rs.random() * 2.0
        r2 = 4.5 + rs.random() * 3.0
        onset = np.clip(t / 0.12, 0, 1)
        wobble = timbre.liveliness * onset * (
            0.6 * np.sin(2 * np.pi * r1 * t + rs.random() * 6.28) +
            0.4 * np.sin(2 * np.pi * r2 * t + rs.random() * 6.28))
        env *= 1.0 + wobble

    ac = min(ANTICLICK_SAMPLES, n_samples)
    if ac > 1:
        env[-ac:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, ac)))

    return env


def build_harmonic_envelope(timbre, total_dur, n_samples, harmonic_index, frequency, note_off_sample=None):
    if timbre.harmonic_decay_rate <= 0 and timbre.attack_brightness <= 0:
        env = np.ones(n_samples)
    else:
        t = np.linspace(0, total_dur, n_samples, endpoint=False)
        env = np.ones(n_samples)

        if timbre.harmonic_decay_rate > 0:
            rate = timbre.harmonic_decay_rate * (
                1.0 + timbre.pitch_decay_scale * pitch_register(frequency))
            extra = rate * max(harmonic_index, 0.12)
            if extra >= 0.01:
                env = np.exp(-extra * t)

        if timbre.attack_brightness > 0 and harmonic_index > 0:
            brightness = timbre.attack_brightness * ((harmonic_index + 1) ** 0.5 - 1)
            tau = timbre.attack_brightness_tau
            env *= 1.0 + brightness * np.exp(-t / tau)

    if timbre.release_damping > 0 and note_off_sample is not None and note_off_sample < n_samples and harmonic_index > 0:
        r_len = n_samples - note_off_sample
        rel_t = np.arange(r_len, dtype=np.float64) / SAMPLE_RATE
        env[note_off_sample:] *= np.exp(-harmonic_index * timbre.release_damping * rel_t)

    return env


def compute_harmonic_amplitude(base_amp, mult, idx, velocity, timbre, frequency):
    pf = pitch_register(frequency)
    lf = pitch_low_factor(frequency)
    vc = vel_curve(velocity)

    pitch_scale = max(1.0 + timbre.pitch_brightness * pf * idx * 0.15, 0.0)
    vel_scale = min(1.0 + timbre.vel_harmonic_boost * vc * idx * 0.1, 2.0)

    if timbre.pitch_low_warmth > 0 and lf > 0.5 and 1 <= idx <= 3:
        low_boost = (lf - 0.5) / 0.5
        pitch_scale *= 1.0 + timbre.pitch_low_warmth * low_boost * 0.4

    return base_amp * pitch_scale * vel_scale


def synthesize(frequency, duration, velocity, timbre, timbre_name="default", note_id=0):
    release_tail = min(timbre.release * 0.6, 0.25)
    total_dur = duration + release_tail
    n_samples = int(SAMPLE_RATE * total_dur)
    if n_samples == 0:
        return np.zeros(0)
    t = np.linspace(0, total_dur, n_samples, endpoint=False)

    vc = vel_curve(velocity)
    rng = np.random.RandomState((int(frequency * 100 + velocity * 1000) + note_id * 7919) % (2 ** 31))
    note_off = int(SAMPLE_RATE * duration)

    if timbre.vibrato_depth > 0:
        vib_env = np.clip((t - timbre.vibrato_delay) / 0.2, 0, 1)
        vib = vib_env * timbre.vibrato_depth * np.sin(2 * np.pi * timbre.vibrato_rate * t)
        phase = 2 * np.pi * np.cumsum(frequency * (2 ** (vib / 1200.0))) / SAMPLE_RATE
    else:
        phase = 2 * np.pi * frequency * t

    waveform = np.zeros(n_samples)
    B = timbre.inharmonicity * (1.0 + 3.0 * pitch_low_factor(frequency)) if timbre.inharmonicity > 0 else 0.0
    det_ratio = 2 ** (timbre.detune_cents / 1200) if timbre.detune_cents > 0 and timbre.detune_mix > 0 else 0

    for idx, (mult, amp) in enumerate(timbre.harmonics):
        actual_mult = mult * np.sqrt(1 + B * mult * mult) if B > 0 else mult
        if frequency * actual_mult >= SAMPLE_RATE / 2:
            break
        h_amp = compute_harmonic_amplitude(amp, mult, idx, velocity, timbre, frequency)
        h_amp *= 1.0 + 0.03 * (rng.random() - 0.5)
        h_env = build_harmonic_envelope(timbre, total_dur, n_samples, idx, frequency, note_off)
        waveform += h_amp * np.sin(phase * actual_mult) * h_env
        if det_ratio > 0:
            waveform += h_amp * timbre.detune_mix * (
                np.sin(phase * actual_mult * det_ratio) +
                np.sin(phase * actual_mult / det_ratio)) * h_env

    if timbre.sub_osc > 0 and frequency / 2 > 20:
        waveform += timbre.sub_osc * np.sin(2 * np.pi * (frequency / 2) * t)

    if timbre.noise_level > 0:
        lf = pitch_low_factor(frequency)
        nl = timbre.noise_level * (1.0 + timbre.vel_noise_boost * vc) * (1.0 - 0.4 * lf)
        waveform += nl * rng.randn(n_samples) * np.exp(
            -timbre.noise_decay * (1.0 + 0.5 * lf) * t)

    ref = _REF_AMPS.get(timbre_name, 1.0)
    if ref > 0:
        waveform /= ref

    fc = timbre.filter_cutoff * (1.0 - 0.3 * pitch_register(frequency))
    if timbre.vel_brightness > 0:
        fc *= 1.0 + timbre.vel_brightness * vc
    if fc < SAMPLE_RATE * 0.4:
        waveform = butter_lpf(waveform, fc)

    if timbre.tremolo_depth > 0:
        waveform *= 1.0 - timbre.tremolo_depth * 0.5 * (
            1 + np.sin(2 * np.pi * timbre.tremolo_rate * t))

    env = build_envelope(timbre, duration, n_samples, velocity, frequency, note_off)
    return waveform * env * (0.4 + 0.6 * vc)


def synthesize_drum(note, duration, velocity, rng):
    n = int(SAMPLE_RATE * min(duration, 0.6))
    if n == 0:
        return np.zeros(0)
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    vc = vel_curve(velocity)

    if note in (35, 36):
        punch = 150 + 30 * vc
        sweep = punch * np.exp(-35 * t) + 42
        body = np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE) * np.exp(-7.5 * t)
        click = rng.randn(n) * np.exp(-85 * t) * 0.15
        sub = np.sin(2 * np.pi * 42 * t) * np.exp(-5.0 * t) * 0.40
        wave = body + click + sub
    elif note in (38, 40):
        tone = (np.sin(2 * np.pi * 180 * t) * np.exp(-20 * t) * 0.60
                + np.sin(2 * np.pi * 330 * t) * np.exp(-28 * t) * 0.18)
        noise = rng.randn(n) * np.exp(-15 * t) * 0.42
        wave = tone + noise
    elif note in (42, 44):
        metal = np.sin(2 * np.pi * 6100 * t + np.sin(2 * np.pi * 8300 * t))
        wave = (metal * 0.20 + rng.randn(n) * 0.28) * np.exp(-42 * t)
    elif note == 46:
        metal = np.sin(2 * np.pi * 6100 * t + np.sin(2 * np.pi * 8300 * t))
        wave = (metal * 0.22 + rng.randn(n) * 0.25) * np.exp(-9 * t)
    elif note in (49, 57):
        ring = np.sin(2 * np.pi * 3200 * t + 2 * np.sin(2 * np.pi * 4800 * t))
        wave = (ring * 0.18 + rng.randn(n) * 0.22) * np.exp(-3.5 * t)
    elif note in (51, 53, 59):
        ring = np.sin(2 * np.pi * 4500 * t + 1.2 * np.sin(2 * np.pi * 6800 * t))
        wave = (ring * 0.25 + rng.randn(n) * np.exp(-15 * t) * 0.10) * np.exp(-5 * t)
    elif note in (47, 48, 50, 45, 43, 41):
        bf = 80 + (note - 41) * 18
        sweep = bf * 1.5 * np.exp(-15 * t) + bf
        wave = (np.sin(2 * np.pi * np.cumsum(sweep) / SAMPLE_RATE) * np.exp(-9 * t)
                + rng.randn(n) * np.exp(-50 * t) * 0.10)
    elif note in (39, 37):
        noise = rng.randn(n)
        wave = noise * (np.exp(-60 * t) * 0.25 + np.exp(-40 * np.maximum(t - 0.01, 0)) * 0.35
                        + np.exp(-20 * np.maximum(t - 0.02, 0)) * 0.45) * 0.22
    else:
        wave = np.sin(2 * np.pi * (200 + (note - 35) * 10) * t) * np.exp(-12 * t)

    peak = np.max(np.abs(wave))
    if peak > 0:
        wave /= peak
    ac = min(ANTICLICK_SAMPLES, n)
    if ac > 1:
        wave[-ac:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, ac)))
    return wave * (0.4 + 0.6 * vc)


def apply_soundboard(signal, amount=0.1):
    if amount <= 0:
        return signal
    board = np.zeros_like(signal)
    for freq, r, gain in [
        (88.0, 0.982, 0.14), (110.0, 0.980, 0.18), (147.0, 0.978, 0.11),
        (175.0, 0.976, 0.09), (220.0, 0.973, 0.13), (277.0, 0.970, 0.07),
        (330.0, 0.968, 0.05), (440.0, 0.965, 0.03),
    ]:
        w0 = 2 * np.pi * freq / SAMPLE_RATE
        board += lfilter([gain * (1 - r * r), 0, 0], [1.0, -2 * r * np.cos(w0), r * r], signal)
    bp = np.max(np.abs(board))
    if bp > 1e-8:
        board *= np.max(np.abs(signal)) / bp
    return signal * (1.0 - amount) + board * amount


def generate_hrtf_pair(azimuth_deg, ir_length=64):
    az = np.radians(azimuth_deg)
    itd = abs((HEAD_RADIUS / SPEED_OF_SOUND) * (az + np.sin(az)) if abs(az) > 1e-6 else 0)
    g_near = 10 ** (abs(1.5 * np.sin(az)) / 20)
    g_far = 10 ** (-abs(1.5 * np.sin(az)) / 20)
    near_ir = np.zeros(ir_length)
    far_ir = np.zeros(ir_length)
    near_ir[0] = 1.0
    far_ir[min(int(itd * SAMPLE_RATE), ir_length - 1)] = 1.0
    sfc = 8000 - 4000 * abs(np.sin(az))
    if sfc < SAMPLE_RATE * 0.45:
        far_ir = sosfilt(butter(1, sfc, btype='low', fs=SAMPLE_RATE, output='sos'), far_ir)
    near_ir *= g_near
    far_ir *= g_far
    return (far_ir, near_ir) if azimuth_deg >= 0 else (near_ir, far_ir)


def apply_hrtf(mono, azimuth_deg):
    l_ir, r_ir = generate_hrtf_pair(azimuth_deg)
    return (fftconvolve(mono, l_ir, mode='full')[:len(mono)],
            fftconvolve(mono, r_ir, mode='full')[:len(mono)])


def generate_room_ir(duration=0.9, room_size=0.40, damping=0.65):
    n = int(SAMPLE_RATE * duration)
    ir_l, ir_r = np.zeros(n), np.zeros(n)
    for i, (dms, g) in enumerate(zip(
            [1.5, 4.2, 8.0, 13.5, 20.0, 28.0, 38.0, 52.0],
            [0.55, 0.40, 0.28, 0.18, 0.11, 0.07, 0.04, 0.02])):
        d = int(SAMPLE_RATE * dms * room_size / 1000)
        if d < n:
            ir_l[d] += g * (1.0 + 0.06 * np.sin(i * 1.7))
            ir_r[min(d + int(SAMPLE_RATE * 0.00025 * (i % 3)), n - 1)] += g * (1.0 - 0.06 * np.sin(i * 1.7))
    ls = int(SAMPLE_RATE * 0.06 * room_size)
    ln = n - ls
    if ln > 0:
        env = np.exp(-4.5 / (duration * room_size + 0.01) * np.linspace(0, duration, ln))
        env *= 1 - np.exp(-np.linspace(0, 8, ln))
        late_cutoff = 1200 + 8000 * (1 - damping)
        sos = butter(2, late_cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
        rng_l = np.random.RandomState(42)
        ir_l[ls:] += sosfilt(sos, rng_l.randn(ln) * env * 0.06)
        rng_r = np.random.RandomState(137)
        ir_r[ls:] += sosfilt(sos, rng_r.randn(ln) * env * 0.06)
    pk = max(np.max(np.abs(ir_l)), np.max(np.abs(ir_r)), 1e-10)
    return ir_l / pk, ir_r / pk


def apply_conv_reverb(l, r, irl, irr, wet=0.05):
    return (l * (1 - wet) + fftconvolve(l, irl, mode='full')[:len(l)] * wet,
            r * (1 - wet) + fftconvolve(r, irr, mode='full')[:len(r)] * wet)


def stereo_compressor(left, right, threshold=0.55, ratio=2.0, attack_ms=25, release_ms=60):
    linked = np.maximum(np.abs(left), np.abs(right))
    ca = np.exp(-1.0 / max(int(SAMPLE_RATE * attack_ms / 1000), 1))
    cr = np.exp(-1.0 / max(int(SAMPLE_RATE * release_ms / 1000), 1))
    env = np.empty(len(linked))
    env[0] = linked[0]
    for i in range(1, len(linked)):
        c = ca if linked[i] > env[i - 1] else cr
        env[i] = c * env[i - 1] + (1 - c) * linked[i]
    gain = np.ones_like(env)
    m = env > threshold
    gain[m] = (threshold + (env[m] - threshold) / ratio) / env[m]
    return left * gain, right * gain


def midi_to_note_name(mn):
    return f"{NOTE_NAMES[mn % 12]}{(mn // 12) - 1}"


def note_to_frequency(mn):
    return NOTE_FREQUENCIES[midi_to_note_name(mn)]


def collect_tempo_map(mid):
    ev = []
    for tr in mid.tracks:
        tk = 0
        for msg in tr:
            tk += msg.time
            if msg.type == "set_tempo":
                ev.append((tk, msg.tempo))
    ev.sort(key=lambda x: x[0])
    if not ev or ev[0][0] != 0:
        ev.insert(0, (0, 500000))
    return ev


def ticks_to_seconds(abs_ticks, tempo_map, tpb):
    sec = 0.0
    pt = 0
    pp = tempo_map[0][1]
    for tk, tp in tempo_map[1:]:
        if tk >= abs_ticks:
            break
        sec += (tk - pt) / tpb * (pp / 1e6)
        pt = tk
        pp = tp
    sec += (abs_ticks - pt) / tpb * (pp / 1e6)
    return sec


def parse_midi(midi_file):
    mid = mido.MidiFile(midi_file)
    tempo_map = collect_tempo_map(mid)
    tracks, track_pans = [], []
    for track in mid.tracks:
        notes, active, active_vel = [], {}, {}
        pedal_on, pedal_held = {}, {}
        abs_ticks, channel_program, channel_pan = 0, {}, {}
        for msg in track:
            abs_ticks += msg.time
            t = ticks_to_seconds(abs_ticks, tempo_map, mid.ticks_per_beat)
            if msg.type == "program_change":
                channel_program[msg.channel] = msg.program
            elif msg.type == "control_change" and msg.control == 64:
                ch = msg.channel
                if msg.value >= 64:
                    pedal_on[ch] = True
                else:
                    pedal_on[ch] = False
                    for nn, st, vl in pedal_held.get(ch, []):
                        notes.append((st, nn, min(max(t - st, 0.01), MAX_PEDAL_SUSTAIN),
                                      vl, ch, channel_program.get(ch, 0)))
                    pedal_held[ch] = []
            elif msg.type == "control_change" and msg.control == 10:
                channel_pan[msg.channel] = msg.value
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                if key in active:
                    old_st = active[key]
                    old_vl = active_vel[key]
                    notes.append((old_st, msg.note, max(t - old_st, 0.01), old_vl,
                                  msg.channel, channel_program.get(msg.channel, 0)))
                active[key] = t
                active_vel[key] = msg.velocity / 127
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    st = active.pop(key)
                    vl = active_vel.pop(key)
                    if pedal_on.get(msg.channel, False):
                        pedal_held.setdefault(msg.channel, []).append((msg.note, st, vl))
                    else:
                        notes.append((st, msg.note, max(t - st, 0.01), vl,
                                      msg.channel, channel_program.get(msg.channel, 0)))
        end_t = ticks_to_seconds(abs_ticks, tempo_map, mid.ticks_per_beat)
        for ch in pedal_held:
            for nn, st, vl in pedal_held[ch]:
                notes.append((st, nn, min(max(end_t - st, 0.01), MAX_PEDAL_SUSTAIN),
                              vl, ch, channel_program.get(ch, 0)))
        for (ci, nn), st in list(active.items()):
            vl = active_vel[(ci, nn)]
            notes.append((st, nn, max(end_t - st, 0.01), vl, ci, channel_program.get(ci, 0)))
        if notes:
            tracks.append(notes)
            track_pans.append(channel_pan.get(notes[0][4]))
    return tracks, track_pans


def render_tracks(tracks, track_pans, output_file):
    max_end = 0
    for notes in tracks:
        for start, note, dur, vel, ch, prog in notes:
            tn = PROGRAM_MAP.get(prog, "default") if ch != 9 else "default"
            tb = TIMBRES.get(tn, TIMBRES["default"])
            extra = min(tb.release * 0.6, 0.25) if ch != 9 else 0
            e = int((start + dur + extra) * SAMPLE_RATE) + SAMPLE_RATE
            if e > max_end:
                max_end = e
    max_end += SAMPLE_RATE

    print(f"[INFO] {len(tracks)} tracks, {max_end / SAMPLE_RATE:.1f}s")
    ir_l, ir_r = generate_room_ir()
    master_l, master_r = np.zeros(max_end), np.zeros(max_end)
    used_pans = set()

    for ti, notes in enumerate(tracks):
        buf = np.zeros(max_end)
        timbre_name, is_drum, sb = "default", False, 0.0
        if notes:
            _, _, _, _, ch0, prog0 = notes[0]
            if ch0 == 9:
                timbre_name, is_drum = "drums", True
            else:
                timbre_name = PROGRAM_MAP.get(prog0, "default")
                sb = TIMBRES.get(timbre_name, TIMBRES["default"]).soundboard
        print(f"  Track {ti}: {len(notes)} notes [{timbre_name}]")

        drum_rng = np.random.RandomState(42 + ti)

        for ni, (start, note, dur, vel, ch, prog) in enumerate(notes):
            if ch == 9:
                tone = synthesize_drum(note, dur, vel, drum_rng)
            else:
                tn = PROGRAM_MAP.get(prog, "default")
                tone = synthesize(note_to_frequency(note), dur, vel,
                                  TIMBRES[tn], tn, note_id=ni)
            s = int(start * SAMPLE_RATE)
            e = s + len(tone)
            if e > max_end:
                tone = tone[:max_end - s]
                e = max_end
            buf[s:e] += tone

        if sb > 0:
            buf = apply_soundboard(buf, sb)

        rms = np.sqrt(np.mean(buf ** 2))
        if rms > 1e-6:
            buf *= 0.12 / rms
        pk = np.max(np.abs(buf))
        if pk > 1.0:
            buf /= pk

        pan_cc = track_pans[ti] if ti < len(track_pans) else None
        if pan_cc is not None:
            az = (pan_cc - 64) / 64.0 * 55.0
        else:
            az = TRACK_PANNING.get(timbre_name, 0)
            if not is_drum:
                while az in used_pans and az != 0:
                    az += 5 * (1 if az >= 0 else -1)
                    if abs(az) > 60:
                        break
                used_pans.add(az)

        l, r = apply_hrtf(buf, az)
        master_l += l
        master_r += r

    if len(tracks) > 0:
        master_l, master_r = stereo_compressor(master_l, master_r)
        master_l, master_r = apply_conv_reverb(master_l, master_r, ir_l, ir_r, wet=0.05)
        pk = max(np.max(np.abs(master_l)), np.max(np.abs(master_r)), 1e-10)
        master_l *= 0.92 / pk
        master_r *= 0.92 / pk

    stereo = np.clip(np.column_stack((master_l, master_r)), -1.0, 1.0)
    sf.write(output_file, stereo, SAMPLE_RATE, format="FLAC", subtype="PCM_24")
    print(f"[OK] {output_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output")
    a = p.parse_args()
    tracks, pans = parse_midi(a.input)
    render_tracks(tracks, pans, a.output or os.path.splitext(a.input)[0] + ".flac")


if __name__ == "__main__":
    main()
