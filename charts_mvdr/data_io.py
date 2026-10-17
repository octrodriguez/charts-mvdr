"""Baseband I/O for CHARTS simulated data (Kotekan test_charts generator format)."""
from pathlib import Path

import numpy as np
import re
import h5py


SAMPLES_PER_FRAME = 1536
PREFIX_BYTES = 4  # one uint32 written before each frame


def unpack_int4x2(raw: np.ndarray) -> np.ndarray:
    """Unpack 4-bit complex samples: low nibble = real, high nibble = imag (two's complement)."""
    re = (raw & 0x0F).astype(np.int8)
    im = (raw >> 4).astype(np.int8)
    re[re > 7] -= 16
    im[im > 7] -= 16
    return re.astype(np.float32) + 1j * im.astype(np.float32)


def read_baseband_frame(path, num_freq: int, num_ant: int,
                        samples: int = SAMPLES_PER_FRAME) -> np.ndarray:
    """Read one frame and return complex voltages with shape (time, freq, antenna)."""
    raw = np.fromfile(Path(path), dtype=np.uint8, offset=PREFIX_BYTES)
    expected = samples * num_freq * num_ant
    if raw.size != expected:
        raise ValueError(f"{path}: {raw.size} bytes, expected {expected}")
    return unpack_int4x2(raw).reshape(samples, num_freq, num_ant)

FRAME_NAME_RE = re.compile(r"^(?P<window>.+)_(?P<index>\d{7})\.bin$")


def read_window_meta(frame_path):
    """Read the metadata of the window a frame belongs to (<window>_meta.h5).

    Returns a dict with frequencies, antenna positions and, when available,
    "frame_time_s" ({frame index: time [s]}), or None if the metadata file is not
    found next to the frame.
    """
    frame_path = Path(frame_path)
    m = FRAME_NAME_RE.match(frame_path.name)
    if m is None:
        return None
    meta_path = frame_path.with_name(f"{m['window']}_meta.h5")
    if not meta_path.exists():
        return None
    with h5py.File(meta_path, "r") as f:
        meta = {
            "frequencies_mhz": f["frequencies_mhz"][:],
            "antenna_pos_x_m": f["antenna_pos_x_m"][:],
            "antenna_pos_y_m": f["antenna_pos_y_m"][:],
        }
        if "frames/timestamp_s" in f and "frames/output_file_index" in f:
            # {output frame index: time since window start [s]}
            meta["frame_time_s"] = dict(zip(f["frames/output_file_index"][:].astype(int),
                                            f["frames/timestamp_s"][:].astype(float)))
        return meta


def read_astron_correlation(path, num_freq: int, num_ant: int,
                            block_size: int = 2) -> np.ndarray:
    """Read one cudaCorrelatorAstron dump (rawFileWrite) as R[f, i, j] = sum_t v_i conj(v_j).

    Layout (Kotekan test_charts, sim/inspector.py): uint32 metadata size, metadata,
    then int32 pairs (re, im) with shape (freq, num_blocks, block_size, block_size, 2).
    Blocks cover the lower block triangle in row-major order, block index
    ry * (ry + 1) / 2 + rx for ry >= rx, and entry [py, px] of block (ry, rx)
    belongs to element pair (i, j) = (ry * block_size + py, rx * block_size + px).

    Convention (verified against np_correlator, scripts/04_compare_correlators.py):
    the kernel stores S_ij = sum_t conj(v_i) v_j, i.e. the complex conjugate of the
    np_correlator convention. This reader conjugates, so the returned matrix uses
    the same convention as np_correlator.correlate (Hermitian, sums over samples,
    not normalized).

    Caveat: the kernel computes Im(S_ij) = sum Re_i Im_j + Im_i (-Re_j) with the
    negation in 4-bit arithmetic, so -(-8) wraps to -8. Samples equal to -8 (outside
    the F-engine range [-7, 7]) corrupt the imaginary part of the affected
    cross-correlations; data within [-7, 7] are reproduced exactly.
    """
    path = Path(path)
    with open(path, "rb") as f:
        meta_size = int(np.fromfile(f, dtype="<u4", count=1)[0])
    n_rec = num_ant // block_size
    n_blocks = n_rec * (n_rec + 1) // 2
    count = num_freq * n_blocks * block_size * block_size * 2
    raw = np.fromfile(path, dtype="<i4", count=count, offset=4 + meta_size)
    if raw.size != count:
        raise ValueError(f"{path}: {raw.size} int32 values, expected {count}")
    raw = raw.reshape(num_freq, n_blocks, block_size, block_size, 2).astype(np.float64)
    blocks = raw[..., 0] - 1j * raw[..., 1]          # conjugate: S_ij -> R_ij

    ry, rx = np.tril_indices(n_rec)  # row-major order == ry * (ry + 1) / 2 + rx
    R = np.zeros((num_freq, num_ant, num_ant), dtype=np.complex128)
    stored = np.zeros((num_ant, num_ant), dtype=bool)
    for py in range(block_size):
        for px in range(block_size):
            ey, ex = ry * block_size + py, rx * block_size + px
            R[:, ey, ex] = blocks[:, :, py, px]
            stored[ey, ex] = True
    fill = ~stored
    R[:, fill] = np.conj(R.transpose(0, 2, 1))[:, fill]
    return R


CORR_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{7})\.bin$")


def load_window_dir_meta(window_dir) -> dict:
    """Metadata of a simulated window from the <window>_meta.h5 in window_dir.

    Returns antenna positions, frequencies and, when available, the frame
    timestamps ("frame_time_s", time since window start [s]) with the output
    file index of each frame ("frame_out_idx").
    """
    window_dir = Path(window_dir)
    files = sorted(window_dir.glob("*_meta.h5"))
    if not files:
        raise FileNotFoundError(f"no *_meta.h5 in {window_dir}")
    with h5py.File(files[0], "r") as f:
        meta = {k: f[k][:] for k in ("antenna_pos_x_m", "antenna_pos_y_m", "frequencies_mhz")}
        if "frames/timestamp_s" in f and "frames/output_file_index" in f:
            meta["frame_time_s"] = f["frames/timestamp_s"][:].astype(float)
            meta["frame_out_idx"] = f["frames/output_file_index"][:].astype(int)
    return meta


def load_events_file(window_dir) -> list:
    """Event list from the first *events*.json in window_dir ([] if none)."""
    import json

    files = sorted(Path(window_dir).glob("*events*.json"))
    if not files:
        return []
    events = json.loads(files[0].read_text())
    return events if isinstance(events, list) else []


def interferers_by_channel(events) -> dict:
    """{channel: [(l, m, event_type), ...]} for the RFI events of an events list."""
    out = {}
    for ev in events:
        if str(ev.get("event_type", "")).startswith("rfi") and ev.get("channels"):
            for ch in ev["channels"]:
                out.setdefault(int(ch), []).append(
                    (float(ev["l0"]), float(ev["m0"]), ev["event_type"]))
    return out


def list_correlation_dumps(corr_dir) -> dict:
    """{dump index: path} for the corr_%07d.bin files in corr_dir."""
    return {int(m["index"]): p for p in sorted(Path(corr_dir).glob("*.bin"))
            if (m := CORR_NAME_RE.match(p.name))}


def select_frames(indices, meta: dict, t_range=None, exclude=None) -> list:
    """Dump indices whose frame time is inside t_range and outside exclude.

    t_range and exclude are (t0, t1) in seconds since the window start (both
    ends inclusive) or None. exclude gives the complement of an on-pulse
    window, e.g. the signal-free (off-pulse) frames for a KL covariance.
    Without frame timestamps in meta the indices are returned unchanged when
    no selection is asked for; asking for one raises ValueError.
    """
    indices = sorted(int(k) for k in indices)
    if t_range is None and exclude is None:
        return indices
    if "frame_time_s" not in meta:
        raise ValueError("time selection needs frame timestamps in the window metadata")
    t = dict(zip(np.asarray(meta["frame_out_idx"]).astype(int), meta["frame_time_s"]))
    out = []
    for k in indices:
        tk = t.get(k)
        if tk is None:
            continue
        if t_range is not None and not (t_range[0] <= tk <= t_range[1]):
            continue
        if exclude is not None and exclude[0] <= tk <= exclude[1]:
            continue
        out.append(k)
    return out


def mean_correlation(paths, num_freq: int, num_ant: int, samples: int = SAMPLES_PER_FRAME,
                     channels=None) -> np.ndarray:
    """Average of cudaCorrelatorAstron dumps, normalized to power per sample.

    Returns R[f, i, j] (all channels, or only the given channel indices, in
    that order) with the np_correlator convention, divided by the number of
    dumps times the samples per dump.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("no correlation dumps given")
    sel = slice(None) if channels is None else np.asarray(channels, dtype=int)
    R = None
    for p in paths:
        Rk = read_astron_correlation(p, num_freq, num_ant)[sel]
        R = Rk if R is None else R + Rk
    return R / (len(paths) * samples)
