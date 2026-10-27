"""MVDR with a moving look direction on real covariance cubes, and gains helpers.

`run_tracking` applies the rules of the Kotekan stage (stage_reference.run_stage:
blind local window guard < |k - t| <= half_window, loading relative to the median
eigenvalue, gains corrected as G^-1 R G^-H) to a cube of measured covariances,
with one look direction per frame (a source that moves across the sky, such as
the Sun). For each frame and channel it returns the DFT and MVDR beam power and
the white-noise gain WNG = N ||w||^2 of the MVDR weights (0 dB for DFT).

`interpolate_gains` fills channels without a gain solution (flagged during
calibration) by linear interpolation of amplitude and unwrapped phase across
frequency, so that RFI channels, which are usually flagged in the calibration,
can still be beamformed with calibrated steering. Interpolated channels are
returned in a mask: results there depend on the gains being smooth.

`write_sidecar` writes the JSON metadata next to a gains file
(charts_gains_format_spec.md, Sec. 1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .beamforming import steering_vectors
from .stage_reference import window_indices

SIDECAR_KEYS = ("created_utc", "valid_from_utc", "valid_to_utc", "calibrator", "ref_ant",
                "antennas", "positions_file", "fengine_gains", "charts_utils_commit")


def interpolate_gains(g, freqs):
    """Fill NaN gains per antenna: linear in amplitude and in unwrapped phase.

    g: (n_freq, n_ant) complex with NaN where there is no solution; freqs
    increasing. Channels outside the solved range take the nearest solved value.
    Returns (filled gains, mask of filled channels (n_freq, n_ant)).
    """
    g = np.asarray(g, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=float)
    out = g.copy()
    filled = ~np.isfinite(g)
    for a in range(g.shape[1]):
        ok = np.isfinite(g[:, a]) & (g[:, a] != 0)
        if ok.sum() < 2:
            raise ValueError(f"antenna index {a}: fewer than 2 solved channels")
        amp = np.interp(freqs, freqs[ok], np.abs(g[ok, a]))
        ph = np.interp(freqs, freqs[ok], np.unwrap(np.angle(g[ok, a])))
        out[~ok, a] = (amp * np.exp(1j * ph))[~ok]
        filled[:, a] = ~ok
    return out, filled


def batched_mvdr(F, a, loading: float):
    """MVDR weights for many channels: F (n_chan, n, n), a (n_chan, n).

    delta = loading x median eigenvalue of F per channel (loading_ref "noise").
    Returns w (n_chan, n) with w^H a = 1.
    """
    n = F.shape[-1]
    ev = np.linalg.eigvalsh(F)
    delta = loading * np.median(ev, axis=-1)
    Rl = F + delta[:, None, None] * np.eye(n)[None]
    X = np.linalg.solve(Rl, a[..., None])[..., 0]
    denom = np.einsum("ci,ci->c", a.conj(), X)
    return X / denom[:, None]


def run_tracking(vis, freqs_hz, pos_x, pos_y, look_lm, gains=None, half_window: int = 10,
                 guard: int = 1, loading: float = 10.0, min_frames=None, sign: int = 1):
    """DFT and MVDR beam power with one look direction per frame.

    vis: (n_ant, n_ant, n_chan, n_frames) covariances (V_ab = x_a x_b^*, as
    charts-utils readers); look_lm: (n_frames, 2) direction cosines (east,
    north); gains: (n_chan, n_ant) or None; channels with a non-finite gain in
    any antenna are skipped (NaN). sign = -1 uses conjugated steering vectors
    (the other phase convention). Returns dict with power (2, n_frames, n_chan)
    [dft, mvdr], wng (n_frames, n_chan) in linear units, n_window (n_frames,).
    """
    vis = np.asarray(vis)
    n_ant, _, n_chan, n_frames = vis.shape
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    look_lm = np.asarray(look_lm, dtype=float).reshape(n_frames, 2)
    if min_frames is None:
        min_frames = half_window - guard
    R = np.moveaxis(vis, (3, 2), (0, 1)).astype(np.complex128)       # (frame, chan, a, b)
    chans = np.arange(n_chan)
    if gains is not None:
        gains = np.asarray(gains, dtype=np.complex128)
        if gains.shape != (n_chan, n_ant):
            raise ValueError(f"gains must be ({n_chan}, {n_ant}); got {gains.shape}")
        ok = np.all(np.isfinite(gains) & (gains != 0), axis=1)
        chans = np.flatnonzero(ok)
        inv = np.zeros_like(gains)
        inv[ok] = 1.0 / gains[ok]
        R = inv[None, :, :, None] * R * inv.conj()[None, :, None, :]
    power = np.full((2, n_frames, n_chan), np.nan)
    wng = np.full((n_frames, n_chan), np.nan)
    n_window = np.zeros(n_frames, dtype=int)
    x, y = np.asarray(pos_x, float), np.asarray(pos_y, float)
    for t in range(n_frames):
        k = window_indices(t, n_frames, half_window, guard)
        n_window[t] = k.size
        A = np.stack([steering_vectors(look_lm[t, 0], look_lm[t, 1], f, x, y)[0] for f in freqs_hz[chans]])
        if sign < 0:
            A = A.conj()
        Rt = R[t, chans]
        wd = A / n_ant
        power[0, t, chans] = np.real(np.einsum("ci,cij,cj->c", wd.conj(), Rt, wd))
        if k.size < min_frames:
            continue
        F = R[k][:, chans].mean(axis=0)
        w = batched_mvdr(F, A, loading)
        power[1, t, chans] = np.real(np.einsum("ci,cij,cj->c", w.conj(), Rt, w))
        wng[t, chans] = n_ant * np.sum(np.abs(w) ** 2, axis=1)
    return {"power": power, "wng": wng, "n_window": n_window}


def write_sidecar(gains_path, **meta) -> Path:
    """Write <gains stem>.json with the spec keys (missing keys stored as null)."""
    path = Path(gains_path).with_suffix(".json")
    unknown = sorted(set(meta) - set(SIDECAR_KEYS) - {"notes"})
    if unknown:
        raise ValueError(f"unknown sidecar keys: {unknown}")
    doc = {k: meta.get(k) for k in SIDECAR_KEYS}
    if "notes" in meta:
        doc["notes"] = meta["notes"]
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
