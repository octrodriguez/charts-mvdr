"""NumPy reference of the Kotekan correlation-domain MVDR stage (chartsMVDRBeams).

The stage forms beams from a stream of correlation frames. For frame t it
estimates a covariance from the neighbouring frames only (blind local window),

    F(t) = sum_{k in W_t} S_k / (n_t K_s),   W_t = {k : guard < |k - t| <= half_window},

computes DFT weights w = d / N and MVDR weights
w = (F + delta I)^-1 d / (d^H (F + delta I)^-1 d) per channel and beam, and
outputs the beam power P = w^H R(t) w with R(t) = S_t / K_s. S_k are the
correlation sums of frame k (R_ij = sum v_i v_j^*, as data_io and np_correlator)
and K_s the samples per frame. The current frame and its neighbours never enter
their own F, so a dispersed burst cannot null itself; RFI that starts with the
burst is only partly in F.

Weights are recomputed every ``update_every`` frames (and at the first frame
with enough frames in its window) and held in between; frames before the first
update are flagged (NaN power). An update is skipped when n_t < min_frames.

Optional complex antenna gains follow the charts-utils convention
(v_obs = g v, charts_utils.cal.algorithms.apply_gains): covariances are
corrected as G^-1 R G^-H. Antennas whose gain is NaN in a channel are left out
of that channel (zero weight).

Every rule here is mirrored by the Kotekan stage; scripts compare the two with a
floating-point tolerance, not an exact match.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .beamforming import dft_weights, mvdr_weights, steering_vectors
from .data_io import numbered_files, read_rawfile_payload


def window_indices(t: int, n_frames: int, half_window: int, guard: int) -> np.ndarray:
    """Frame indices k with guard < |k - t| <= half_window inside [0, n_frames)."""
    if half_window < 1 or guard < 0 or guard >= half_window:
        raise ValueError("need half_window >= 1 and 0 <= guard < half_window")
    k = np.arange(max(0, t - half_window), min(n_frames, t + half_window + 1))
    return k[np.abs(k - t) > guard]


def load_gains(path, channels=None, n_ant: int | None = None) -> np.ndarray:
    """Read a gains file (n_freq, n_ant) complex .npy, optionally for some channels.

    Accepts complex64 or complex128 (returned as complex128). Time-resolved
    solutions (n_freq, n_ant, n_time) are rejected: average them first
    (cal_pipeline_gpu --save-avg-gains).
    """
    g = np.load(Path(path))
    if not np.iscomplexobj(g) or g.ndim != 2:
        raise ValueError(f"{path}: expected a complex (n_freq, n_ant) array, got "
                         f"{g.dtype} {g.shape}")
    if n_ant is not None and g.shape[1] != n_ant:
        raise ValueError(f"{path}: {g.shape[1]} antennas, expected {n_ant}")
    g = g.astype(np.complex128)
    return g if channels is None else g[np.asarray(channels, dtype=int)]


def correct_gains(R: np.ndarray, g: np.ndarray) -> np.ndarray:
    """G^-1 R G^-H for one covariance R (n, n) and gains g (n,); NaN gains give 0 rows."""
    ok = np.isfinite(g) & (g != 0)
    inv = np.zeros_like(g, dtype=np.complex128)
    inv[ok] = 1.0 / g[ok]
    return inv[:, None] * R * inv.conj()[None, :]


@dataclass
class StageConfig:
    """Parameters of the stage (names as in the Kotekan config)."""

    beams: list                      # [(l, m), ...]
    positions_x_m: np.ndarray
    positions_y_m: np.ndarray
    samples_per_frame: int
    half_window: int = 10            # cov_half_window
    guard: int = 1                   # cov_guard
    min_frames: int | None = None    # cov_min_frames (default half_window - guard)
    update_every: int = 1
    loading: float = 10.0
    loading_ref: str = "noise"

    def __post_init__(self):
        if self.min_frames is None:
            self.min_frames = self.half_window - self.guard
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")


@dataclass
class StageOutput:
    """Beam power and weights of one run, for the channels processed."""

    power: np.ndarray                # (2, n_frames, n_chan, n_beam): [dft, mvdr]
    n_window: np.ndarray             # (n_frames,) frames in each window
    update_frames: list = field(default_factory=list)
    weights: list = field(default_factory=list)   # per update: (n_chan, n_beam, n_ant)
    wng: list = field(default_factory=list)       # per update: (n_chan, n_beam), N ||w||^2


def run_stage(S: np.ndarray, freqs_hz, cfg: StageConfig, gains=None) -> StageOutput:
    """Run the stage on correlation sums S (n_frames, n_chan, n_ant, n_ant).

    freqs_hz holds one frequency per channel of S; gains, if given, is
    (n_chan, n_ant) for the same channels.
    """
    S = np.asarray(S)
    n_frames, n_chan, n_ant, _ = S.shape
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    if freqs_hz.shape != (n_chan,):
        raise ValueError("freqs_hz must have one value per channel")
    beams = np.asarray(cfg.beams, dtype=float).reshape(-1, 2)
    n_beam = beams.shape[0]
    if gains is not None:
        gains = np.asarray(gains, dtype=np.complex128)
        if gains.shape != (n_chan, n_ant):
            raise ValueError("gains must be (n_chan, n_ant)")

    def corrected(R, c):
        return R if gains is None else correct_gains(R, gains[c])

    good = (np.ones((n_chan, n_ant), bool) if gains is None
            else np.isfinite(gains) & (gains != 0))

    power = np.full((2, n_frames, n_chan, n_beam), np.nan)
    n_window = np.zeros(n_frames, dtype=int)
    out = StageOutput(power=power, n_window=n_window)

    # DFT weights and steering vectors per channel (restricted to good antennas)
    W_dft, D = [], []
    for c in range(n_chan):
        idx = np.flatnonzero(good[c])
        A = steering_vectors(beams[:, 0], beams[:, 1], freqs_hz[c],
                             np.asarray(cfg.positions_x_m)[idx],
                             np.asarray(cfg.positions_y_m)[idx])
        D.append(A)
        W_dft.append(dft_weights(A))

    W_mvdr = None
    for t in range(n_frames):
        k = window_indices(t, n_frames, cfg.half_window, cfg.guard)
        n_window[t] = k.size
        is_update = (W_mvdr is None) or (t % cfg.update_every == 0)
        if is_update and k.size >= cfg.min_frames:
            Ssum = S[k].sum(axis=0) / (k.size * cfg.samples_per_frame)
            W_new = np.zeros((n_chan, n_beam, n_ant), dtype=np.complex128)
            for c in range(n_chan):
                idx = np.flatnonzero(good[c])
                F = corrected(Ssum[c], c)[np.ix_(idx, idx)]
                W_new[c][:, idx] = mvdr_weights(F, D[c], loading=cfg.loading,
                                                loading_ref=cfg.loading_ref)
            W_mvdr = W_new
            out.update_frames.append(t)
            out.weights.append(W_new)
            out.wng.append(n_ant_good(good)[:, None] * np.sum(np.abs(W_new) ** 2, axis=-1))
        for c in range(n_chan):
            idx = np.flatnonzero(good[c])
            R = corrected(S[t, c], c)[np.ix_(idx, idx)] / cfg.samples_per_frame
            power[0, t, c] = np.real(np.einsum("bi,ij,bj->b", W_dft[c].conj(), R, W_dft[c]))
            if W_mvdr is not None:
                w = W_mvdr[c][:, idx]
                power[1, t, c] = np.real(np.einsum("bi,ij,bj->b", w.conj(), R, w))
    return out


def n_ant_good(good: np.ndarray) -> np.ndarray:
    """Number of usable antennas per channel (WNG normalization N)."""
    return good.sum(axis=1)


def load_correlation_stack(corr_dir, num_freq: int, num_ant: int, channels=None):
    """Correlation sums of all dumps in corr_dir, (n_frames, n_chan, n_ant, n_ant).

    Uses data_io.read_astron_correlation (conjugated to R_ij = sum v_i v_j^*).
    Returns (dump indices, stack).
    """
    from .data_io import list_correlation_dumps, read_astron_correlation

    dumps = list_correlation_dumps(corr_dir)
    if not dumps:
        raise FileNotFoundError(f"no correlation dumps in {corr_dir}")
    sel = slice(None) if channels is None else np.asarray(channels, dtype=int)
    idx = sorted(dumps)
    stack = np.stack([read_astron_correlation(dumps[i], num_freq, num_ant)[sel] for i in idx])
    return idx, stack


def read_stage_power(directory, num_freq: int, num_beams: int) -> np.ndarray:
    """Beam power written by chartsMVDRBeams (mvdr_power_%07d.bin), (2, n_frames, num_freq,
    num_beams): plane 0 DFT, plane 1 MVDR (NaN before the first update)."""
    files = numbered_files(directory, "mvdr_power")
    if not files:
        raise FileNotFoundError(f"no mvdr_power_*.bin in {directory}")
    size = 2 * num_freq * num_beams
    frames = []
    for p in files:
        d = np.frombuffer(read_rawfile_payload(p), "<f4")
        if d.size != size:
            raise ValueError(f"{p}: {d.size} values, expected {size}")
        frames.append(d.reshape(2, num_freq, num_beams))
    return np.stack(frames, axis=1)


def read_stage_weights(directory, num_freq: int, num_beams: int, num_elements: int) -> dict:
    """Weights written by chartsMVDRBeams (mvdr_weights_%07d.bin, one per update).

    Returns update_frames, n_window (n_update,), weights (n_update, num_freq, num_beams,
    num_elements) complex64 and wng (n_update, num_freq, num_beams).
    """
    files = numbered_files(directory, "mvdr_weights")
    nw = num_freq * num_beams * num_elements
    out = {"update_frames": [], "n_window": [], "weights": [], "wng": []}
    for p in files:
        d = read_rawfile_payload(p)
        expected = 16 + nw * 8 + num_freq * num_beams * 4
        if len(d) != expected:
            raise ValueError(f"{p}: {len(d)} bytes, expected {expected}")
        hdr = np.frombuffer(d[:16], "<i8")
        out["update_frames"].append(int(hdr[0]))
        out["n_window"].append(int(hdr[1]))
        out["weights"].append(np.frombuffer(d[16:16 + nw * 8], "<c8")
                              .reshape(num_freq, num_beams, num_elements))
        out["wng"].append(np.frombuffer(d[16 + nw * 8:], "<f4").reshape(num_freq, num_beams))
    return {k: np.asarray(v) for k, v in out.items()}
