"""Tests of charts_mvdr.real_io: both charts-utils generations of cubes and gains files give
the same results through scripts 17, 19 and 20 and tools/kotekan_gains.py.

Synthetic data: four antennas on a 4.2 m square, unequal noise, a source drifting across the
sky in every channel, one emitter in the lower half of the band, complex gains with delays.
Version 1: V1 = conj(v v^H), gains g1 = conj(g_v). Version 2: V2 = v v^H, gains g_v.
"""

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr import real_io
from charts_mvdr.beamforming import steering_vectors

ROOT = Path(__file__).resolve().parents[1]
S17 = ROOT / "scripts" / "17_real_nulling.py"
S19 = ROOT / "scripts" / "19_fringe_recovery.py"
S20 = ROOT / "scripts" / "20_nulling_gains.py"
TOOL = ROOT / "tools" / "kotekan_gains.py"

ANTS = [0, 4, 32, 36]
XY = {0: (0.0, 0.0), 4: (-4.2, 0.0), 32: (0.0, 4.2), 36: (-4.2, 4.2)}
NOISE = np.diag([1.0, 2.0, 1.5, 1.0])
T, NF = 40, 48
FREQS = 370.0 + 0.3 * np.arange(NF)
RFI, CLEAN = "370-377", "378-384.2"


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("real_io")
    rng = np.random.default_rng(7)
    x = np.array([XY[a][0] for a in ANTS])
    y = np.array([XY[a][1] for a in ANTS])
    look = np.stack([np.linspace(-0.45, 0.45, T), 0.31 + 0.03 * np.cos(np.linspace(0, np.pi, T))], axis=1)
    p1 = 30 * (1 + 0.5 * np.sin(np.arange(T) / 6.0)) ** 2
    tau = np.array([0.0, 0.4e-9, 5.4e-9, 2.3e-9])
    amp = np.array([1.0, 1.3, 0.9, 1.25])
    g_v = amp[None, :] * np.exp(-2j * np.pi * FREQS[:, None] * 1e6 * tau[None, :])   # voltage gains
    R = np.zeros((4, 4, NF, T), dtype=np.complex128)                                # v v^H
    for c, f in enumerate(FREQS):
        s = steering_vectors(look[:, 0], look[:, 1], f * 1e6, x, y)                 # (T, 4), simulator sign
        Rc = NOISE[None] + 0.3 * s[:, :, None] * s.conj()[:, None, :]
        if f < 377:
            a1 = steering_vectors(0.3, 0.7, f * 1e6, x, y)[0]
            Rc = Rc + p1[:, None, None] * np.outer(a1, a1.conj())
        Rc = g_v[c][None, :, None] * Rc * g_v[c].conj()[None, None, :]
        R[:, :, c, :] = np.moveaxis(Rc, 0, -1)
    common = dict(freqs=FREQS, times=1.7939e9 + 40.96 * np.arange(T), ants=np.array(ANTS), look=look)
    np.savez(d / "cube_v1.npz", vis=np.conj(R).astype(np.complex64), **common)               # no tag: version 1
    np.savez(d / "cube_v2.npz", vis=R.astype(np.complex64), vis_convention=2, **common)
    np.save(d / "gains_v1.npy", np.conj(g_v).astype(np.complex64))
    interp = np.zeros((NF, 4), dtype=bool)
    interp[5] = True
    np.savez(d / "gains_v2.npz", gains=g_v.astype(np.complex64), gains_solved=g_v.astype(np.complex64),
             interpolated=interp, antennas=np.array(ANTS), freq_mhz=FREQS, format_version=2, ref_ant=0)
    np.savez(d / "gains_v1.npz", gains=np.conj(g_v).astype(np.complex64), antennas=np.array(ANTS),
             format_version=1)
    pos = np.zeros((64, 3))
    for a in ANTS:
        pos[a, :2] = XY[a]
    np.save(d / "pos.npy", pos)
    return d, g_v


def test_loader_convention_and_to_internal():
    old = types.SimpleNamespace(load_gpu_corr_frame=lambda: None)
    new = types.SimpleNamespace(load_gpu_corr_frame=lambda: None, corr_from_astron_storage=np.conj)
    assert real_io.loader_convention(old) == 1 and real_io.loader_convention(new) == 2
    v = np.array([1 + 2j, 3 - 1j])
    np.testing.assert_array_equal(real_io.to_internal(v, 1), v)
    np.testing.assert_array_equal(real_io.to_internal(v, 2), v.conj())
    with pytest.raises(ValueError):
        real_io.to_internal(v, 3)


def test_read_cube_both_versions_agree(data):
    d, _ = data
    c1, c2 = real_io.read_cube(d / "cube_v1.npz"), real_io.read_cube(d / "cube_v2.npz")
    assert (c1["convention"], c2["convention"]) == (1, 2)
    np.testing.assert_allclose(c1["vis"], c2["vis"])
    assert c1["ants"] == ANTS and c1["look"].shape == (T, 2)


def test_read_gains_versions(data, tmp_path):
    d, g_v = data
    want = np.conj(g_v)                                              # internal version
    f2 = real_io.read_gains(d / "gains_v2.npz")
    assert f2.convention == 2 and f2.antennas == ANTS and f2.interpolated[5].all()
    np.testing.assert_allclose(f2.gains, want, rtol=1e-6)
    np.testing.assert_allclose(real_io.read_gains(d / "gains_v1.npz").gains, want, rtol=1e-6)
    assert not real_io.read_gains(d / "gains_v2.npz", key="gains_solved").interpolated.any()
    np.testing.assert_allclose(real_io.read_gains(d / "gains_v1.npy", fmt=1).gains, want, rtol=1e-6)
    np.save(tmp_path / "bare_v2.npy", g_v)
    np.testing.assert_allclose(real_io.read_gains(tmp_path / "bare_v2.npy", fmt=2).gains, want)
    with pytest.raises(ValueError, match="--gains-format"):
        real_io.read_gains(d / "gains_v1.npy")
    (tmp_path / "bare_v2.json").write_text(json.dumps({"format_version": 2}))
    assert real_io.read_gains(tmp_path / "bare_v2.npy").convention == 2      # from the sidecar
    np.save(tmp_path / "cube.npy", np.ones((NF, 4, 3), dtype=complex))
    with pytest.raises(ValueError, match="n_freq, n_ant"):
        real_io.read_gains(tmp_path / "cube.npy", fmt=2)
    with pytest.raises(ValueError, match="no array"):
        real_io.read_gains(d / "gains_v1.npz", key="gains_solved")


def test_gains_from_args_selects_by_antenna(data):
    d, g_v = data
    ns = types.SimpleNamespace(gains=d / "gains_v2.npz", gain_antennas="36,0", gains_format=None,
                               gains_key="gains")
    g, ants, interp = real_io.gains_from_args(ns, n_freq=NF)
    assert ants == [36, 0] and interp.shape == (NF, 2)
    np.testing.assert_allclose(g, np.conj(g_v)[:, [3, 0]], rtol=1e-6)
    ns.gain_antennas = "0,5"
    with pytest.raises(ValueError, match="not in"):
        real_io.gains_from_args(ns)
    ns = types.SimpleNamespace(gains=d / "gains_v1.npy", gain_antennas=None, gains_format=1, gains_key="gains")
    with pytest.raises(ValueError, match="--gain-antennas"):
        real_io.gains_from_args(ns)
    ns.gain_antennas = "0,4,32,36"
    with pytest.raises(ValueError, match="channels"):
        real_io.gains_from_args(ns, n_freq=NF + 1)


V1 = ("cube_v1.npz", "gains_v1.npy", ["--gain-antennas", "0,4,32,36", "--gains-format", "1"])
V2 = ("cube_v2.npz", "gains_v2.npz", [])


def test_scripts_give_the_same_results_for_both_versions(data, tmp_path):
    d, g_v = data
    for tag, (cube, gains, extra) in (("v1", V1), ("v2", V2)):
        r = run(S19, "--vis", d / cube, "--gains", d / gains, *extra, "--positions", d / "pos.npy",
                "--filter", "project", "--k", "auto", "--bands", RFI, CLEAN, "--out", tmp_path / f"fr_{tag}")
        assert r.returncode == 0, r.stderr
        r = run(S20, "--vis", d / cube, "--gains", d / gains, *extra, "--auto", 2, "--rfi", RFI,
                "--ref", CLEAN, "--out", tmp_path / f"ng_{tag}")
        assert r.returncode == 0, r.stderr
    # 19: the source is recovered in the interference band, identically for both versions
    rho1, rho2 = (np.load(tmp_path / f"fr_{t}.npz")["rho"] for t in ("v1", "v2"))
    np.testing.assert_allclose(rho1, rho2, atol=1e-5)
    band = FREQS < 377
    assert np.nanmedian(rho1[0, band]) < 0.3 and np.nanmedian(rho1[1, band]) > 0.9
    # 20: the matrices for raw voltages are physical, so they do not depend on the file version,
    # and M = P G^-1 with the voltage gains G = diag(g_v): M G is a projector
    M1, M2 = (np.load(tmp_path / f"ng_{t}.npz")["M"] for t in ("v1", "v2"))
    np.testing.assert_allclose(M1, M2, atol=1e-5)
    P = M1[0, 3] * g_v[3][None, :]
    np.testing.assert_allclose(P @ P, P, atol=1e-4)


def test_real_nulling_sign_and_interpolated_mask(data, tmp_path):
    d, _ = data
    look = np.load(d / "cube_v1.npz")["look"]
    res = {}
    for tag, (cube, gains, extra) in (("v1", V1), ("v2", V2)):
        r = run(S17, "--vis", d / cube, "--gains", d / gains, *extra, "--positions", d / "pos.npy",
                "--look", f"{look[20, 0]},{look[20, 1]}", "--rfi", RFI, "--clean", CLEAN,
                "--loading", 0.1, "--half-window", 8, "--out", tmp_path / tag)
        assert r.returncode == 0, r.stderr
        assert "-> -1" in r.stdout                                 # internal version 1 for both
        res[tag] = np.load(tmp_path / tag / "real_nulling.npz")
    np.testing.assert_allclose(res["v1"]["power"], res["v2"]["power"], rtol=1e-4, equal_nan=True)
    assert res["v2"]["interp"].tolist() == [i == 5 for i in range(NF)]   # mask of the gain file
    assert not res["v1"]["interp"].any()


def test_mixing_versions_is_detected_by_the_results(data, tmp_path):
    d, _ = data
    r = run(S19, "--vis", d / "cube_v2.npz", "--gains", d / "gains_v1.npy", "--gain-antennas", "0,4,32,36",
            "--gains-format", "2", "--positions", d / "pos.npy", "--filter", "project", "--k", "auto",
            "--bands", CLEAN, "--out", tmp_path / "mixed")
    assert r.returncode == 0, r.stderr
    rho = np.load(tmp_path / "mixed.npz")["rho"]
    assert np.nanmedian(rho[0, FREQS > 378]) < 0.5               # wrong declaration: fringes do not follow the model


def test_kotekan_gains_from_format2_file(data, tmp_path):
    d, g_v = data
    out = tmp_path / "full.npy"
    r = run(TOOL, "--gains", d / "gains_v2.npz", "--num-inputs", 64, "--out", out)
    assert r.returncode == 0, r.stderr
    full = np.load(out)
    assert full.shape == (NF, 64) and full.dtype == np.complex64
    np.testing.assert_allclose(full[:, ANTS], np.conj(g_v), rtol=1e-6)      # the loader conjugates back to g_v
    assert np.isnan(full[:, [a for a in range(64) if a not in ANTS]]).all()
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["format_version"] == 1 and side["antennas"] == ANTS
    assert "1 channels interpolated by the calibration" in side["notes"]
    legacy = tmp_path / "full_v1.npy"
    r = run(TOOL, "--gains", d / "gains_v1.npy", "--gain-antennas", "0,4,32,36", "--gains-format", 1,
            "--num-inputs", 64, "--out", legacy)
    assert r.returncode == 0, r.stderr
    np.testing.assert_allclose(np.load(legacy), full, rtol=1e-6, equal_nan=True)
    assert real_io.read_gains(out).convention == 1                          # sidecar read back
    r = run(TOOL, "--gains", d / "gains_v1.npy", "--gain-antennas", "0,4,32,36", "--num-inputs", 64,
            "--out", tmp_path / "x.npy")
    assert r.returncode == 1 and "--gains-format" in r.stderr
