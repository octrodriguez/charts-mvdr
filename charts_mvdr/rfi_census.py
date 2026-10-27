"""RFI census of real autocorrelations (complement of Fig. 3 of Andrew et al. 2026).

For each antenna and time sample the autocorrelation spectrum A(f, t) is divided
by its median over frequency (removes packet loss, which zeroes samples and
scales the whole frame, and slow gain drift) and compared with a running median
over frequency (the smooth bandpass). The relative excess

    r(f, t) = A(f, t) / median_{|f' - f| <= w/2} A(f', t) - 1

is flagged when r > k * sigma(t), with sigma(t) = 1.4826 MAD_f r(., t) a robust
estimate of the spectral scatter of that frame. The fraction of valid frames in
which a channel is flagged (occupancy) separates persistent emitters from
intermittent ones:

- persistent:   occupancy >= persistent (default 0.5)
- intermittent: intermittent <= occupancy < persistent (default 0.05)
- variable:     not a spectral peak, but flagged by a temporal-variability
                criterion supplied by the caller (for instance
                charts_utils.cal.algorithms.mad_rfi_zapper)
- clean:        none of the above.

A channel class for the array takes the most severe class shared by at least
half of the antennas. `cross_counts` crosses it with the per-channel fringe
classes of charts_mvdr.fringes (majority over antenna pairs).
"""
from __future__ import annotations

import warnings

import numpy as np

CLASSES = ("clean", "variable", "intermittent", "persistent")   # increasing severity


def robust_sigma(x, axis=None):
    """1.4826 x median absolute deviation, ignoring NaN."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def normalize_frames(auto, min_level: float = 0.5):
    """Divide each frame by its median over frequency and mark invalid frames.

    auto: (n_freq, n_time) real autocorrelations. A frame is invalid when it is
    empty or its band median is below `min_level` times the median over frames
    (heavy packet loss). Returns (normalized auto with NaN in invalid frames,
    valid mask (n_time,)).
    """
    a = np.asarray(auto, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"auto must have shape (n_freq, n_time); got {a.shape}")
    a = np.where(a > 0, a, np.nan)
    level = np.nanmedian(a, axis=0) if a.size else np.array([])
    ref = np.nanmedian(level)
    valid = np.isfinite(level) & (level >= min_level * ref)
    out = np.full_like(a, np.nan)
    out[:, valid] = a[:, valid] / level[valid]
    return out, valid


def running_median(x, window: int, chunk: int = 256):
    """Running median along axis 0 (frequency) with reflected edges, NaN ignored."""
    x = np.asarray(x, dtype=float)
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be odd and >= 3")
    h = window // 2
    pad = np.pad(x, ((h, h),) + ((0, 0),) * (x.ndim - 1), mode="reflect")
    out = np.empty_like(x)
    view = np.lib.stride_tricks.sliding_window_view(pad, window, axis=0)   # (n_freq, n_time, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN (invalid) frames
        for s in range(0, x.shape[1], chunk):
            out[:, s:s + chunk] = np.nanmedian(view[:, s:s + chunk], axis=-1)
    return out


def spectral_peaks(auto_norm, window: int = 31, k: float = 5.0):
    """Relative excess r(f, t) over the running median and the flags r > k sigma(t)."""
    base = running_median(auto_norm, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = auto_norm / base - 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = robust_sigma(r, axis=0)                              # (n_time,), NaN if invalid
    with np.errstate(invalid="ignore"):
        flags = r > k * sigma[None, :]
    return r, flags


def occupancy(flags, valid):
    """Fraction of valid frames in which each channel is flagged."""
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return np.full(flags.shape[0], np.nan)
    return flags[:, valid].mean(axis=1)


def classify(occ, variable=None, persistent: float = 0.5, intermittent: float = 0.05):
    """Per-channel class index into CLASSES from occupancy and a variability mask."""
    occ = np.asarray(occ, dtype=float)
    cls = np.zeros(occ.shape, dtype=int)
    if variable is not None:
        cls[np.asarray(variable, dtype=bool)] = 1
    cls[occ >= intermittent] = 2
    cls[occ >= persistent] = 3
    return cls


def array_class(cls):
    """Most severe class shared by at least half of the antennas: cls (n_ant, n_freq)."""
    cls = np.asarray(cls)
    n = cls.shape[0]
    srt = np.sort(cls, axis=0)[::-1]                                 # descending per channel
    return srt[(n + 1) // 2 - 1]


def fringe_majority(labels):
    """Most common fringe label over antenna pairs: labels (n_pair, n_freq) of str."""
    labels = np.asarray(labels)
    out = np.empty(labels.shape[1], dtype=labels.dtype)
    for f in range(labels.shape[1]):
        vals, counts = np.unique(labels[:, f], return_counts=True)
        out[f] = vals[np.argmax(counts)]
    return out


def cross_counts(cls, fringe_labels, fringe_names):
    """Channel counts: rows CLASSES, columns fringe_names. cls (n_freq,), labels (n_freq,)."""
    table = np.zeros((len(CLASSES), len(fringe_names)), dtype=int)
    for i in range(len(CLASSES)):
        for j, name in enumerate(fringe_names):
            table[i, j] = int(np.sum((cls == i) & (fringe_labels == name)))
    return table
