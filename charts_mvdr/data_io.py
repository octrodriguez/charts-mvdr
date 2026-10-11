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

    Returns a dict with frequencies and antenna positions, or None if the
    metadata file is not found next to the frame.
    """
    frame_path = Path(frame_path)
    m = FRAME_NAME_RE.match(frame_path.name)
    if m is None:
        return None
    meta_path = frame_path.with_name(f"{m['window']}_meta.h5")
    if not meta_path.exists():
        return None
    with h5py.File(meta_path, "r") as f:
        return {
            "frequencies_mhz": f["frequencies_mhz"][:],
            "antenna_pos_x_m": f["antenna_pos_x_m"][:],
            "antenna_pos_y_m": f["antenna_pos_y_m"][:],
        }