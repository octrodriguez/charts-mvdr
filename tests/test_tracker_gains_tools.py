import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NA, NF, NS = 8, 3, 64


def _make_window(d: Path, n_frames=3, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    for t in range(n_frames):
        re = np.clip(np.rint(rng.normal(0, 2.5, (NS, NF, NA))), -8, 7).astype(np.int16)
        im = np.clip(np.rint(rng.normal(0, 2.5, (NS, NF, NA))), -8, 7).astype(np.int16)
        raw = ((re & 0x0F) | ((im & 0x0F) << 4)).astype(np.uint8)
        np.concatenate([np.zeros(4, np.uint8), raw.ravel()]).tofile(d / f"win_{t:07d}.bin")
    (d / "win_meta.h5").write_bytes(b"")
    (d / "win_events.json").write_text("[]")


def _volts(path):
    raw = np.fromfile(path, np.uint8)[4:]
    re = (raw & 0x0F).astype(np.int16)
    im = (raw >> 4).astype(np.int16)
    re[re > 7] -= 16
    im[im > 7] -= 16
    return (re + 1j * im).reshape(NS, NF, NA)


def _tracker(window: Path, out: Path, corr=None):
    """numpy stand-in for the tracker: y = sum_e w_e v_e, w_e = exp(+j phi_e) * c_e."""
    out.mkdir(parents=True)
    phi = np.linspace(0, 2, NA)[None, :] * (1 + np.arange(NF))[:, None]
    w = np.exp(1j * phi) * (1.0 if corr is None else corr)
    for p in sorted(window.glob("win_*.bin")):
        y = np.einsum("tfe,fe->tf", _volts(p), w).astype(np.complex64)
        meta = b"m" * 9
        with open(out / f"beams_{p.stem[-7:]}.bin", "wb") as f:
            np.uint32(len(meta)).tofile(f)
            f.write(meta)
            f.write(y.tobytes())


def _correctors(g, order="descending"):
    e2a = NA - 1 - np.arange(NA) if order == "descending" else np.arange(NA)
    return 1.0 / g[:NF][:, e2a]


def _run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_gains_roundtrip_and_conventions(tmp_path):
    _make_window(tmp_path / "win")
    r = _run(ROOT / "tools" / "apply_test_gains.py", "--window-dir", tmp_path / "win",
             "--out-root", tmp_path / "t", "--num-frames", 2, "--num-freq", NF, "--num-ant", NA,
             "--samples", NS, "--file-rows", 6, "--amplitude", 2.0)
    assert r.returncode == 0, r.stderr
    t = tmp_path / "t"
    assert len(list((t / "clean").glob("win_*.bin"))) == 2
    assert (t / "gained" / "win_meta.h5").exists()
    g = np.load(t / "gains_test.npy")
    assert g.shape == (6, NA) and g.dtype == np.complex64
    np.testing.assert_allclose(np.abs(g[:NF]), 2.0, rtol=1e-6)
    np.testing.assert_array_equal(g[NF:], 1.0)
    side = json.loads((t / "gains_test.json").read_text())
    # gained data = q * clean except saturated components
    v0, v1 = _volts(t / "clean" / "win_0000000.bin"), _volts(t / "gained" / "win_0000000.bin")
    q_elem = (g[:NF] / 2.0)[:, NA - 1 - np.arange(NA)]
    diff = np.abs(v1 - q_elem[None] * v0)
    assert np.count_nonzero(diff > 0) <= side["saturated_components"]

    _tracker(t / "clean", t / "ref")
    _tracker(t / "gained", t / "cal", corr=_correctors(g))
    _tracker(t / "gained", t / "nocal")
    _tracker(t / "gained", t / "wrong", corr=g[:NF][:, NA - 1 - np.arange(NA)])   # multiply by g
    _tracker(t / "gained", t / "asc", corr=_correctors(g, "ascending"))           # wrong mapping
    cmp = ROOT / "tools" / "compare_tracker_beams.py"
    ok = _run(cmp, "--ref", t / "ref", "--test", t / "cal", "--expected-scale", "0.5")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "MATCH" in ok.stdout
    for bad in ("nocal", "wrong", "asc"):
        r = _run(cmp, "--ref", t / "ref", "--test", t / bad, "--expected-scale", "0.5")
        assert r.returncode == 1, (bad, r.stdout)


def test_rotate_quarter_saturates():
    sys.path.insert(0, str(ROOT / "tools"))
    from apply_test_gains import rotate_quarter

    re, im = np.array([-8, 3]), np.array([2, -8])
    out_re, out_im, sat = rotate_quarter(re, im, np.array([2, 1]))
    assert out_re.tolist() == [7, 7] and out_im.tolist() == [-2, 3] and sat == 2
