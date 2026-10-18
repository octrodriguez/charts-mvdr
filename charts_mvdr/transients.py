"""Dispersed transients: arrival times and on-pulse masks.

Cold-plasma dispersion delays a pulse at frequency f relative to a reference
frequency f_ref by

    dt(f) = K_DM * DM * (f^-2 - f_ref^-2),

with K_DM = 4148.741601 MHz^2 pc^-1 cm^3 s, f in MHz and DM in pc cm^-3. The
simulator (test_charts) references arrival times to the top channel of the band.
"""
from __future__ import annotations

import numpy as np

K_DM = 4148.741601  # MHz^2 pc^-1 cm^3 s


def dispersion_delay_s(freq_mhz, dm: float, f_ref_mhz: float | None = None) -> np.ndarray:
    """Delay [s] at freq_mhz relative to f_ref_mhz (default: the highest frequency)."""
    f = np.asarray(freq_mhz, dtype=float)
    ref = float(np.max(f)) if f_ref_mhz is None else float(f_ref_mhz)
    return K_DM * dm * (1.0 / f ** 2 - 1.0 / ref ** 2)


def arrival_times_s(freq_mhz, dm: float, t_ref_s: float, f_ref_mhz: float | None = None):
    """Arrival time [s] of the pulse peak per frequency, t_ref_s at f_ref_mhz."""
    return t_ref_s + dispersion_delay_s(freq_mhz, dm, f_ref_mhz)


def on_pulse_mask(times_s, freq_mhz, dm: float, t_ref_s: float, width_s: float,
                  n_sigma: float = 3.0, frame_s: float = 0.0,
                  f_ref_mhz: float | None = None) -> np.ndarray:
    """Boolean mask (time, freq): True where the pulse is within n_sigma widths.

    times_s are the start times of integrations of length frame_s (0 for
    instantaneous samples); an integration counts as on-pulse when it overlaps
    [t_arr - n_sigma * width_s, t_arr + n_sigma * width_s] in that channel.
    width_s is the standard deviation of the pulse envelope.
    """
    t = np.asarray(times_s, dtype=float)[:, None]
    ta = arrival_times_s(freq_mhz, dm, t_ref_s, f_ref_mhz)[None, :]
    half = n_sigma * width_s
    return (t <= ta + half) & (t + frame_s >= ta - half)
