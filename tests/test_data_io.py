"""Tests for charts_mvdr.data_io readers (synthetic files, no external data)."""

import numpy as np
import pytest

from charts_mvdr.data_io import read_astron_correlation, read_baseband_frame
from charts_mvdr.np_correlator import correlate


def write_baseband(path, V):
    """Pack complex integer voltages (time, freq, ant) as int4x2 with a uint32 prefix."""
    re = V.real.astype(np.int8) & 0x0F
    im = V.imag.astype(np.int8) & 0x0F
    packed = (re | (im << 4)).astype(np.uint8)
    with open(path, "wb") as f:
        np.uint32(0).tofile(f)
        packed.tofile(f)


def write_astron(path, R, block_size=2, meta=b""):
    """Write R (freq, N, N) as cudaCorrelatorAstron does: lower block triangle, storing
    S = conj(R) (the kernel computes sum_t conj(v_i) v_j)."""
    R = np.conj(R)
    F, N, _ = R.shape
    n_rec = N // block_size
    ry, rx = np.tril_indices(n_rec)
    out = np.zeros((F, len(ry), block_size, block_size, 2), dtype="<i4")
    for py in range(block_size):
        for px in range(block_size):
            v = R[:, ry * block_size + py, rx * block_size + px]
            out[..., py, px, 0] = np.rint(v.real)
            out[..., py, px, 1] = np.rint(v.imag)
    with open(path, "wb") as f:
        np.uint32(len(meta)).tofile(f)
        f.write(meta)
        out.tofile(f)


@pytest.fixture
def frame(tmp_path):
    rng = np.random.default_rng(0)
    V = (rng.integers(-8, 8, (64, 3, 8)) + 1j * rng.integers(-8, 8, (64, 3, 8)))
    p = tmp_path / "win_0000000.bin"
    write_baseband(p, V)
    return p, V


def test_baseband_roundtrip(frame):
    p, V = frame
    np.testing.assert_array_equal(read_baseband_frame(p, 3, 8, 64), V.astype(np.complex64))


@pytest.mark.parametrize("meta", [b"", b"\x01" * 12])
def test_astron_roundtrip(tmp_path, frame, meta):
    _, V = frame
    R = correlate(V.astype(np.complex128))
    p = tmp_path / "corr_0000000.bin"
    write_astron(p, R, meta=meta)
    np.testing.assert_array_equal(read_astron_correlation(p, 3, 8), R)


def test_astron_size_mismatch(tmp_path, frame):
    _, V = frame
    p = tmp_path / "corr_0000000.bin"
    write_astron(p, correlate(V.astype(np.complex128)))
    with pytest.raises(ValueError):
        read_astron_correlation(p, 4, 8)


def test_astron_negation_overflow_only_affects_minus8(tmp_path):
    """Emulate the kernel's 4-bit negation: exact for [-7, 7], imaginary error with -8."""
    rng = np.random.default_rng(1)
    V = rng.integers(-7, 8, (256, 2, 8)) + 1j * rng.integers(-7, 8, (256, 2, 8))
    re, im = V.real, V.imag
    wneg = lambda x: np.where(x == -8, -8.0, -x)
    ein = lambda a, b: np.einsum("tfi,tfj->fij", a, b)
    S = ein(re, re) + ein(im, im) + 1j * (ein(re, im) + ein(im, wneg(re)))
    p = tmp_path / "corr_0000000.bin"
    write_astron(p, np.conj(S))           # writer conjugates back to S
    np.testing.assert_array_equal(read_astron_correlation(p, 2, 8),
                                  correlate(V.astype(np.complex128)))
