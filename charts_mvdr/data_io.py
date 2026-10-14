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
