"""How long MVDR weights stay valid: excess output power of stale weights.

Weights trained at time s on the covariance of the frames before s,

    w(s) = (F_s + delta I)^-1 a / (a^H (F_s + delta I)^-1 a),   F_s = mean(R[s - n_avg : s]),

are applied k frames later to the covariance R_t of frame t = s + k. Fresh weights w(t),
trained the same way, satisfy the same constraint w^H a = 1, so both pass a source in the
look direction with unit gain and differ only in the noise and interference they let
through. The excess output power of weights k frames old is

    E_k(t) = w(t - k)^H R_t w(t - k) / w(t)^H R_t w(t).

E_0 = 1 by construction. E_k > 1 means the weights went stale; E_k can fall slightly below 1
when the training window of w(t) happens to describe R_t worse than an older one.

The look direction must be fixed in time (a beam fixed on the sky frame of the instrument,
e.g. the zenith beam of a transit array): with a moving look direction the stale and fresh
weights would constrain different directions and the ratio would not compare like with like.
Frames are the correlator integrations, so changes faster than one frame are not seen.
"""
from __future__ import annotations

import warnings

import numpy as np

from charts_mvdr.tracking import batched_mvdr


def training_weights(R, a, n_avg: int, loading: float, loading_ref: str = "median"):
    """MVDR weights for every frame s >= n_avg from the mean of frames [s - n_avg, s).

    R: (n_frames, n_chan, n, n) covariances; a: (n_chan, n) steering vectors.
    Returns W (n_frames, n_chan, n), NaN for s < n_avg and for channels where R is not finite.
    """
    R = np.asarray(R)
    n_frames, n_chan, n, _ = R.shape
    if n_avg < 1:
        raise ValueError("n_avg must be >= 1")
    W = np.full((n_frames, n_chan, n), np.nan, dtype=np.complex128)
    ok = np.all(np.isfinite(R.reshape(n_frames, n_chan, -1)), axis=(0, 2))
    if not ok.any():
        return W
    cs = np.concatenate([np.zeros((1, ok.sum(), n, n), np.complex128),
                         np.cumsum(R[:, ok].astype(np.complex128), axis=0)])
    for s in range(n_avg, n_frames):
        F = (cs[s] - cs[s - n_avg]) / n_avg
        W[s, ok] = batched_mvdr(F, a[ok], loading, loading_ref)
    return W


def output_power(w, R):
    """w^H R w for w (..., n) and R (..., n, n), real."""
    return np.real(np.einsum("...i,...ij,...j->...", w.conj(), R, w))


def excess_power(R, W, lags):
    """E[k_index, t, chan] = P(w(t - k), R_t) / P(w(t), R_t); NaN where undefined."""
    R = np.asarray(R)
    n_frames, n_chan = R.shape[:2]
    fresh = output_power(W, R)                                     # (n_frames, n_chan)
    E = np.full((len(lags), n_frames, n_chan), np.nan)
    for i, k in enumerate(lags):
        if k < 0 or k >= n_frames:
            continue
        stale = output_power(W[:n_frames - k], R[k:])              # w(t - k) on R_t, t = k ...
        with np.errstate(divide="ignore", invalid="ignore"):
            E[i, k:] = stale / fresh[k:]
    return E


def band_excess_db(E, freqs_mhz, bands):
    """Median over the channels of each band, then (median, 90th percentile) over time, in dB.

    bands: list of (lo, hi) MHz. Returns (med (n_lags, n_bands), p90 (n_lags, n_bands),
    n_chan per band).
    """
    f = np.asarray(freqs_mhz, float)
    med = np.full((E.shape[0], len(bands)), np.nan)
    p90 = np.full_like(med, np.nan)
    nch = []
    for j, (lo, hi) in enumerate(bands):
        sel = (f >= lo) & (f < hi) & np.isfinite(E).any(axis=(0, 1))
        nch.append(int(sel.sum()))
        if not sel.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_t = 10 * np.log10(np.nanmedian(E[:, :, sel], axis=2))   # (n_lags, n_frames)
            med[:, j] = np.nanmedian(per_t, axis=1)
            p90[:, j] = np.nanpercentile(per_t, 90, axis=1)
    return med, p90, nch


def max_lag_below(lag_s, excess_db, threshold_db):
    """Largest lag whose excess and every shorter lag's excess stay below threshold (NaN if none)."""
    best = np.nan
    for s, e in zip(lag_s, excess_db):
        if not np.isfinite(e) or e > threshold_db:
            break
        best = s
    return best
