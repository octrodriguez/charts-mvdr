"""Interference nulling on real visibility cubes and Sun fringe recovery.

A covariance cube saved by 16_rfi_census.py extract holds the charts-utils visibilities
V_ab = x_a conj(x_b) (x = conj(v), so V = conj(R) with R = v v^H in the raw-voltage
convention). The functions here are written for any Hermitian covariance; the callers keep
track of the domain.

- `load_cube` / `calibrate`: read a cube for a set of antennas and divide by the gains of a
  charts-utils gains file (V_ab / (g_a conj(g_b))).
- `sun_lm` / `sun_model`: direction of the Sun per frame and the fringe model
  M = s s^H, s = conjugated steering (charts-utils cube convention).
- `whitened_projector`: remove the k strongest eigenmodes (or those standing above the
  noise floor) of the noise-whitened covariance D R D, mapped back to the input scale,
  P_c = D^-1 (I - U U^H) D, with D^-2 the per-antenna noise of a low-rank plus diagonal fit.
  Whitening keeps unequal noise between antennas from being taken for interference, and
  using the noise (not the total power) keeps a strong emitter from hiding a weaker one.
- `mvdr_filter`: Q = (F + delta I)^-1 with the loading rule of the stage; for any steering d,
  d^H (Q R Q) d / |d^H Q d|^2 is the MVDR beam power, so Q R Q is what the MVDR beamformer
  sees, pair by pair.
- `local_mean`: running mean over +-W frames.
- `fringe_correlation`: |complex correlation| between data and model over time and
  baselines after removing each baseline's time mean (1 = data follow the model).
- `voltage_nulling_matrix`: M = P G^-1 for raw voltages, G = diag(conj(g)): instrumental
  gains with interference modes suppressed (a full N x N matrix per channel).
"""
from __future__ import annotations

import numpy as np

from charts_mvdr import tracking
from charts_mvdr.beamforming import steering_vectors

LOADING_REFS = ("median", "min")


def load_cube(path, antennas):
    """(V (n_frames, n_chan, n, n) complex128, freqs MHz, times, look or None) for antennas.

    look is the cube's own (n_frames, 2) direction array when present (synthetic tests);
    otherwise None and the caller computes the Sun's direction.
    """
    z = np.load(path)
    cube_ants = [int(a) for a in z["ants"]]
    missing = [a for a in antennas if a not in cube_ants]
    if missing:
        raise ValueError(f"antennas {missing} not in the cube {cube_ants}")
    idx = [cube_ants.index(a) for a in antennas]
    V = np.moveaxis(z["vis"][np.ix_(idx, idx)], (3, 2), (0, 1)).astype(np.complex128)
    look = np.asarray(z["look"], float) if "look" in z.files else None
    return V, np.asarray(z["freqs"], float), np.asarray(z["times"], float), look


def calibrate(V, gains, freqs_mhz, interp=True, edge="linear"):
    """V_ab / (g_a conj(g_b)) with charts-utils gains (n_chan, n); returns (Vc, usable mask)."""
    g = np.asarray(gains)
    if interp:
        g, _ = tracking.interpolate_gains(g, np.asarray(freqs_mhz, float), edge=edge)
    ok = np.all(np.isfinite(g) & (g != 0), axis=1)
    inv = np.zeros_like(g, dtype=np.complex128)
    inv[ok] = 1.0 / g[ok]
    Vc = inv[None, :, :, None] * V * np.conj(inv)[None, :, None, :]
    Vc[:, ~ok] = np.nan
    return Vc, ok


def sun_lm(times, positions):
    """(n_frames, 2) direction cosines (east, north) of the Sun (charts-utils coordinates)."""
    from charts_constants import CHARTS_ANTENNA_REF_LOCATION
    from charts_utils.cal.coordinates import CHARTSArray

    arr = CHARTSArray(*CHARTS_ANTENNA_REF_LOCATION, positions)
    alt, az = arr.get_source_altaz(np.asarray(times, dtype=float), "sun")
    alt, az = np.radians(alt), np.radians(az)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az)], axis=1)


def sun_model(look, freq_hz, pos_x, pos_y):
    """Fringe model of a unit point source along look (n_frames, 2) at one frequency:
    (n_frames, n, n), s s^H with s = conjugated steering (charts-utils cube convention)."""
    s = steering_vectors(look[:, 0], look[:, 1], freq_hz, pos_x, pos_y).conj()
    return s[:, :, None] * s.conj()[:, None, :]


def local_mean(D, half_window):
    """Running mean of D (n_frames, n, n) over |k - t| <= half_window."""
    n_t = D.shape[0]
    cs = np.concatenate([np.zeros((1,) + D.shape[1:], D.dtype), np.cumsum(D, axis=0)])
    t = np.arange(n_t)
    lo, hi = np.clip(t - half_window, 0, n_t), np.clip(t + half_window + 1, 0, n_t)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None, None]


def _noise_diag(R, k, n_iter=200, tol=1e-10):
    """Per-antenna noise power of R ~ L L^H + diag(psi) with L of rank k (factor analysis)."""
    dR = np.real(np.diag(R))
    psi = dR.copy()
    for _ in range(n_iter):
        s = 1.0 / np.sqrt(psi)
        ev, U = np.linalg.eigh(s[:, None] * R * s[None, :])
        lam = np.clip(ev[-k:] - 1.0, 0.0, None)
        L = np.sqrt(psi)[:, None] * U[:, -k:] * np.sqrt(lam)[None, :]
        new = np.clip(dR - np.sum(np.abs(L) ** 2, axis=1), 1e-6 * dR, None)
        if np.max(np.abs(new - psi) / dR) < tol:
            psi = new
            break
        psi = new
    return psi


def whitened_projector(R, k=None, thresh=None, whiten="noise", kmax=None):
    """Projector removing interference eigenmodes of a Hermitian covariance R (n, n) or (..., n, n).

    Exactly one of k (fixed number of modes) or thresh must be given. With thresh, the number
    of modes is the count of eigenvalues of diag(R)^-1/2 R diag(R)^-1/2 above thresh times
    the smallest one. At most kmax modes (default n - 2) are removed.
    whiten: "noise" (default) whitens by the per-antenna noise power of a rank-k plus diagonal
    fit (so the noise floor is white and the removed modes are coherent signals, even when the
    interference dominates the autocorrelations), "total" by diag(R), or "none".
    Returns (P_c, k_used) with P_c = D^-1 (I - U U^H) D, so that P_c R P_c^H is R with those
    modes removed.
    """
    if (k is None) == (thresh is None):
        raise ValueError("give exactly one of k or thresh")
    if whiten not in ("noise", "total", "none"):
        raise ValueError("whiten must be 'noise', 'total' or 'none'")
    R = np.asarray(R)
    if R.ndim > 2:
        flat = R.reshape((-1,) + R.shape[-2:])
        out = [whitened_projector(r, k=k, thresh=thresh, whiten=whiten, kmax=kmax) for r in flat]
        P = np.stack([o[0] for o in out]).reshape(R.shape)
        ku = np.array([o[1] for o in out]).reshape(R.shape[:-2])
        return P, ku
    n = R.shape[-1]
    kmax = n - 2 if kmax is None else kmax
    dR = np.real(np.diag(R))
    if thresh is not None:
        s = 1.0 / np.sqrt(dR)
        ev0 = np.linalg.eigvalsh(s[:, None] * R * s[None, :])
        ku = int(min(kmax, np.sum(ev0 > thresh * max(ev0[0], 1e-30 * ev0[-1]))))
    else:
        ku = int(min(int(k), kmax))
    if ku == 0:
        return np.eye(n, dtype=complex), 0
    if whiten == "noise":
        d = 1.0 / np.sqrt(_noise_diag(R, ku))
    elif whiten == "total":
        d = 1.0 / np.sqrt(dR)
    else:
        d = np.ones(n)
    ev, U = np.linalg.eigh(d[:, None] * R * d[None, :])
    Uk = U[:, n - ku:]
    Pw = np.eye(n) - Uk @ Uk.conj().T
    return (1.0 / d)[:, None] * Pw * d[None, :], ku


def mvdr_filter(F, loading=0.1, loading_ref="median"):
    """Q = (F + delta I)^-1, delta = loading x (median or smallest eigenvalue of F)."""
    if loading_ref not in LOADING_REFS:
        raise ValueError(f"loading_ref must be one of {LOADING_REFS}")
    ev, U = np.linalg.eigh(F)
    ref = np.median(ev, axis=-1) if loading_ref == "median" else ev[..., 0]
    inv = 1.0 / np.clip(ev + loading * ref[..., None], 1e-300, None)
    return (U * inv[..., None, :]) @ np.conj(np.swapaxes(U, -1, -2))


def fringe_correlation(D, M):
    """|complex correlation| over time and baselines (upper triangle) after removing each
    baseline's time mean; D, M (n_frames, n, n)."""
    n = D.shape[-1]
    iu = np.triu_indices(n, 1)
    d = D[:, iu[0], iu[1]]
    m = M[:, iu[0], iu[1]]
    d = d - d.mean(axis=0)
    m = m - m.mean(axis=0)
    den = np.sqrt(np.sum(np.abs(d) ** 2) * np.sum(np.abs(m) ** 2))
    return float(np.abs(np.sum(d * m.conj())) / den) if den > 0 else np.nan


def voltage_nulling_matrix(P_voltage, gains_row):
    """M = P G^-1 for raw voltages (v' = M v), G = diag(conj(g)) for charts-utils gains g of
    one channel; P_voltage is a projector built in the calibrated voltage domain."""
    return P_voltage * (1.0 / np.conj(np.asarray(gains_row)))[None, :]
