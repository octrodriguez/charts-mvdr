"""Coherence of visibilities with a geometric fringe model.

For a source tracked by the model, the raw phase of a visibility on baseline B
(east, north, ... components in metres) is phi_B(t) = sum_k B_k phi_k(t), where
phi_k(t) is the model phase of a unit baseline along axis k (for instance from
charts_utils.cal.coordinates.geometric_visibility_phase with unit positions).
The per-channel coherence

    c(B) = |sum_t V(t) exp(-i phi_B(t))| / sum_t |V(t)|

is 1 when V follows the model up to a constant (instrumental) phase and
amplitude, and close to 0 for noise or for a signal with another fringe rate.
Because the constant phase of each channel drops out, it needs no prior gain
calibration. Three cases are useful in practice:

- c(B_layout) high: the channel is dominated by the modelled source and the
  baseline is right;
- c(-B_layout) high: the channel is conjugated (sign convention or spectral
  inversion);
- c(0) high: the channel is dominated by a stationary signal (fixed-direction
  RFI such as a geostationary satellite, or correlated instrumental pickup).

On short baselines the model barely changes over the window and c(B) cannot
tell B from 0; `model_overlap` measures that and `classify_channels` reports
such channels as ambiguous.
"""
from __future__ import annotations

import numpy as np

LABELS = ("source", "conjugate", "stationary", "ambiguous", "none")


def _check(vis, phase_unit):
    vis = np.asarray(vis)
    phase_unit = np.asarray(phase_unit, dtype=float)
    if vis.ndim != 2:
        raise ValueError(f"vis must have shape (n_time, n_chan); got {vis.shape}")
    if phase_unit.ndim != 3:
        raise ValueError(f"phase_unit must have shape (n_chan, n_dim, n_time); got {phase_unit.shape}")
    if phase_unit.shape[0] != vis.shape[1] or phase_unit.shape[2] != vis.shape[0]:
        raise ValueError(f"vis {vis.shape} and phase_unit {phase_unit.shape} do not match")
    return vis, phase_unit


def coherence(vis, phase_unit, baselines, chunk: int = 256) -> np.ndarray:
    """Per-channel coherence c(B) for each baseline in `baselines`.

    vis: (n_time, n_chan) complex visibilities; phase_unit: (n_chan, n_dim, n_time)
    model phase of a unit baseline along each axis; baselines: (n_b, n_dim).
    Returns (n_b, n_chan). Channels with no signal (sum |V| = 0) give NaN.
    """
    vis, phase_unit = _check(vis, phase_unit)
    b = np.atleast_2d(np.asarray(baselines, dtype=float))
    if b.shape[1] != phase_unit.shape[1]:
        raise ValueError(f"baselines need {phase_unit.shape[1]} components; got {b.shape[1]}")
    norm = np.abs(vis).sum(axis=0)
    out = np.empty((b.shape[0], vis.shape[1]))
    for s in range(0, b.shape[0], chunk):
        phase = np.einsum("bk,ckt->bct", b[s:s + chunk], phase_unit)
        out[s:s + chunk] = np.abs(np.einsum("tc,bct->bc", vis, np.exp(-1j * phase)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / norm[None, :]


def model_overlap(phase_unit, baseline) -> np.ndarray:
    """|<exp(-i phi_B(t))>_t| per channel: 1 when the model is constant over the window."""
    phase_unit = np.asarray(phase_unit, dtype=float)
    phase = np.einsum("k,ckt->ct", np.asarray(baseline, dtype=float), phase_unit)
    return np.abs(np.exp(-1j * phase).mean(axis=1))


def fit_baseline(vis, phase_unit, extent, step: float = 0.1, refine_step: float = 0.01):
    """Baseline that maximizes the channel-averaged coherence.

    extent: half-width of the search box per axis (metres), centred on 0, e.g.
    (6, 3). A grid of `step` is searched first, then a grid of `refine_step`
    within one coarse step of the best point. Returns (baseline, coherence).
    """
    vis, phase_unit = _check(vis, phase_unit)
    extent = np.atleast_1d(np.asarray(extent, dtype=float))
    if extent.size != phase_unit.shape[1]:
        raise ValueError(f"extent needs {phase_unit.shape[1]} values; got {extent.size}")

    def best(axes):
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))
        c = np.nanmean(coherence(vis, phase_unit, grid), axis=1)
        i = int(np.nanargmax(c))
        return grid[i], float(c[i])

    coarse = [np.arange(-e, e + step / 2, step) for e in extent]
    b0, _ = best(coarse)
    fine = [np.arange(x - step, x + step + refine_step / 2, refine_step) for x in b0]
    return best(fine)


def classify_channels(vis, phase_unit, baseline, min_coherence: float = 0.3,
                      max_overlap: float = 0.5):
    """Label each channel as source, conjugate, stationary, ambiguous or none.

    The label is the largest of c(B), c(-B) and c(0) if it reaches
    min_coherence ("none" otherwise). Channels where model_overlap(B) exceeds
    max_overlap cannot separate the three cases and are labelled "ambiguous"
    whenever they would not be "none". Returns (labels, c) with labels an
    array of strings of length n_chan and c of shape (3, n_chan) for
    (B, -B, 0).
    """
    b = np.asarray(baseline, dtype=float)
    c = coherence(vis, phase_unit, np.stack([b, -b, np.zeros_like(b)]))
    labels = np.array(LABELS[:3], dtype=object)[np.nanargmax(np.nan_to_num(c, nan=-1.0), axis=0)]
    labels[np.nanmax(np.nan_to_num(c, nan=0.0), axis=0) < min_coherence] = "none"
    ambiguous = model_overlap(phase_unit, b) > max_overlap
    labels[ambiguous & (labels != "none")] = "ambiguous"
    return labels.astype(str), c
