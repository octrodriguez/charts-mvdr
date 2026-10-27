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


FDM_FLIPPED_CHAINS = (1, 2, 3)   # LO - IF sub-bands: sky frequency decreases with FFT bin


def split_fengine_coeffs(coeffs, n_chan: int = 672, flipped=FDM_FLIPPED_CHAINS):
    """Per-chain digital gains from a flat F-engine coefficient array.

    The coefficient file of the requantizer (e.g. final_coeffs_adc0.npy, int64)
    covers the full FFT of one ADC, 8192 bins of 0.3 MHz from 0 to 2457.6 MHz. The
    frequency-division multiplexer (Lau et al., CHARTS front-end paper, Table I)
    places the 300-501.6 MHz band of chain i in sub-band i: chain 0 directly at
    300-501.6 MHz (bin 1000), chains 1-7 translated by their LO. Each chain is one
    contiguous run of `n_chan` non-zero coefficients, zeros elsewhere. Chains in
    `flipped` use the lower sideband (LO - IF, chains 1-3), so their run is in
    decreasing sky frequency and is reversed here. Returns (n_chains, n_chan) float,
    channel k at 300 + 0.3 k MHz for every chain. Raises ValueError if a run does not
    have exactly `n_chan` coefficients (a zero inside the band, or another layout).
    """
    c = np.asarray(coeffs).ravel()
    nz = np.flatnonzero(c)
    if nz.size == 0:
        raise ValueError("no non-zero coefficients")
    br = np.flatnonzero(np.diff(nz) > 1)
    starts = np.r_[nz[0], nz[br + 1]]
    ends = np.r_[nz[br], nz[-1]] + 1
    lengths = ends - starts
    if np.any(lengths != n_chan):
        raise ValueError(f"expected runs of {n_chan} non-zero coefficients; got lengths {lengths.tolist()}")
    out = np.stack([c[s:e] for s, e in zip(starts, ends)]).astype(float)
    for i in flipped:
        if i < out.shape[0]:
            out[i] = out[i, ::-1]
    return out


def robust_sigma(x, axis=None):
    """1.4826 x median absolute deviation, ignoring NaN."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis=axis)


def normalize_frames(auto, min_level: float = 0.5, n_blocks: int = 1):
    """Divide each frame by its median over frequency and mark invalid frames.

    auto: (n_freq, n_time) real autocorrelations. With n_blocks > 1 the band is
    split into that many equal frequency blocks (GPU frame blocks, one per packet
    stream), each normalized by its own median: packet loss zeroes samples per
    block, and heavy loss in one block is missed by a single band median.
    A frame is invalid when any block is empty or below `min_level` times its
    median over frames (heavy packet loss). Returns (normalized auto with NaN in
    invalid frames, valid mask (n_time,)).
    """
    a = np.asarray(auto, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"auto must have shape (n_freq, n_time); got {a.shape}")
    if n_blocks < 1 or a.shape[0] % n_blocks:
        raise ValueError(f"{a.shape[0]} channels cannot be split into {n_blocks} blocks")
    a = np.where(a > 0, a, np.nan)
    out = np.full_like(a, np.nan)
    valid = np.ones(a.shape[1], dtype=bool)
    edges = np.linspace(0, a.shape[0], n_blocks + 1).astype(int)
    levels = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # empty frames
        for lo, hi in zip(edges[:-1], edges[1:]):
            level = np.nanmedian(a[lo:hi], axis=0)
            valid &= np.isfinite(level) & (level >= min_level * np.nanmedian(level))
            levels.append((lo, hi, level))
    for lo, hi, level in levels:
        out[lo:hi, valid] = a[lo:hi, valid] / level[valid]
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


def fringe_majority(labels, tie: str = "ambiguous"):
    """Most common fringe label over antenna pairs: labels (n_pair, n_freq) of str.

    A tie between the most common labels gives `tie` (no alphabetical preference).
    """
    labels = np.asarray(labels)
    out = np.empty(labels.shape[1], dtype=f"<U{max(labels.dtype.itemsize // 4, len(tie))}")
    for f in range(labels.shape[1]):
        vals, counts = np.unique(labels[:, f], return_counts=True)
        top = np.flatnonzero(counts == counts.max())
        out[f] = vals[top[0]] if top.size == 1 else tie
    return out


def cross_counts(cls, fringe_labels, fringe_names):
    """Channel counts: rows CLASSES, columns fringe_names. cls (n_freq,), labels (n_freq,)."""
    table = np.zeros((len(CLASSES), len(fringe_names)), dtype=int)
    for i in range(len(CLASSES)):
        for j, name in enumerate(fringe_names):
            table[i, j] = int(np.sum((cls == i) & (fringe_labels == name)))
    return table
