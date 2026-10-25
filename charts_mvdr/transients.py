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


def burst_from_events(events, event_id=None) -> dict:
    """Parameters of a dispersed burst from an events list (simulator format).

    Returns {"id", "dm", "width_s", "t_top_s", "l", "m"} of the first event with
    event_type "frb" (or the one with event_id). width_s is the envelope sigma
    (pulse_width_ms / 1e3); t_top_s is the arrival time in the top channel.
    Raises ValueError if there is none.
    """
    for ev in events:
        if ev.get("event_type") == "frb" and event_id in (None, ev.get("event_id")):
            return {"id": ev.get("event_id", ""), "dm": float(ev["dm"]),
                    "width_s": float(ev["pulse_width_ms"]) * 1e-3,
                    "t_top_s": float(ev["t_start_s"]), "l": float(ev["l0"]),
                    "m": float(ev["m0"])}
    raise ValueError("no frb event" + (f" with id {event_id!r}" if event_id else ""))


def stage_frame_masks(meta: dict, events: list, n_frames: int, freqs_mhz, n_sigma: float = 3.0,
                      local_s: float = 0.05, guard_frames: int = 1,
                      frame_s: float = 1536 * 10e-6 / 3):
    """On-pulse and off-local frame masks of a stream of frames (stage outputs).

    Frame k of the stream is window frame k (frame_out_idx k in meta). Returns
    (times (n_frames,), on (n_frames, n_chan), off (n_frames, n_chan), truncated (n_chan,),
    t_arr (n_chan,), burst). on follows the dispersion delay (on_pulse_mask); off holds the
    frames within local_s of the arrival in each channel, outside the on frames and
    guard_frames on each side; truncated flags channels whose leading tail precedes the
    first frame of the burst (cut by the simulator).
    """
    burst = burst_from_events(events)
    t_of = dict(zip(np.asarray(meta["frame_out_idx"]).astype(int), meta["frame_time_s"]))
    missing = [k for k in range(n_frames) if k not in t_of]
    if missing:
        raise ValueError(f"{len(missing)} frames have no time in the metadata")
    times = np.array([t_of[k] for k in range(n_frames)])
    freqs_mhz = np.asarray(freqs_mhz, dtype=float)
    f_top = float(np.max(meta["frequencies_mhz"]))
    on = on_pulse_mask(times, freqs_mhz, burst["dm"], burst["t_top_s"], burst["width_s"],
                       n_sigma, frame_s, f_top)
    t_arr = arrival_times_s(freqs_mhz, burst["dm"], burst["t_top_s"], f_top)
    guard_m = on.copy()
    for k in range(1, guard_frames + 1):
        guard_m[k:] |= on[:-k]
        guard_m[:-k] |= on[k:]
    off = (np.abs(times[:, None] + frame_s / 2 - t_arr[None, :]) <= local_s) & ~guard_m
    frame0 = np.floor(burst["t_top_s"] / frame_s) * frame_s
    truncated = t_arr - n_sigma * burst["width_s"] < frame0
    return times, on, off, truncated, t_arr, burst
