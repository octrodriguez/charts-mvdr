"""Tests of charts_mvdr.nulling, scripts 19_fringe_recovery.py and 20_nulling_gains.py, and
the figure tools fig_phase2d_nulling.py and fig_beam_power.py.

Synthetic cube (charts-utils convention, V = conj(R)): four antennas on a 4.2 m square with
unequal noise, a weak Sun drifting across the sky in every channel, two emitters with
independently fluctuating powers in 357-380 MHz and one in 410-428 MHz.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr import nulling
from charts_mvdr.beamforming import steering_vectors

ROOT = Path(__file__).resolve().parents[1]
S19 = ROOT / "scripts" / "19_fringe_recovery.py"
S20 = ROOT / "scripts" / "20_nulling_gains.py"
PH2D = ROOT / "tools" / "fig_phase2d_nulling.py"
BEAM = ROOT / "tools" / "fig_beam_power.py"

ANTS = [0, 4, 32, 36]
XY = {0: (0.0, 0.0), 4: (-4.2, 0.0), 32: (0.0, 4.2), 36: (-4.2, 4.2)}
NOISE = np.diag([1.0, 2.0, 1.5, 1.0])
T = 40


def run(script, *args):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True)


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    d = tmp_path_factory.mktemp("nulling")
    freqs = 300 + 0.3 * np.arange(672)
    x = np.array([XY[a][0] for a in ANTS])
    y = np.array([XY[a][1] for a in ANTS])
    look = np.stack([np.linspace(-0.45, 0.45, T), 0.31 + 0.03 * np.cos(np.linspace(0, np.pi, T))], axis=1)
    p1 = 30 * (1 + 0.5 * np.sin(np.arange(T) / 6.0)) ** 2
    p2 = 10 * (1 + 0.5 * np.cos(np.arange(T) / 4.0)) ** 2
    vis = np.zeros((4, 4, 672, T), dtype=np.complex64)
    for c, f in enumerate(freqs):
        s = steering_vectors(look[:, 0], look[:, 1], f * 1e6, x, y).conj()          # (T, 4)
        V = NOISE[None] + 0.3 * s[:, :, None] * s.conj()[:, None, :]
        if 357 <= f <= 380:
            a1 = steering_vectors(0.3, 0.7, f * 1e6, x, y)[0].conj()
            a2 = steering_vectors(-0.5, -0.2, f * 1e6, x, y)[0].conj()
            V = V + p1[:, None, None] * np.outer(a1, a1.conj()) + p2[:, None, None] * np.outer(a2, a2.conj())
        if 410 <= f <= 428:
            a3 = steering_vectors(0.6, -0.4, f * 1e6, x, y)[0].conj()
            V = V + p1[:, None, None] * np.outer(a3, a3.conj())
        vis[:, :, c, :] = np.moveaxis(V, 0, -1)
    np.savez(d / "cube.npz", vis=vis, freqs=freqs, times=1.7939e9 + 40.96 * np.arange(T),
             ants=np.array(ANTS), look=look)
    np.save(d / "gains.npy", np.ones((672, 4), dtype=np.complex64))
    pos = np.zeros((64, 3))
    for a in ANTS:
        pos[a, :2] = XY[a]
    np.save(d / "pos.npy", pos)
    return d


def test_whitened_projector_removes_rank_one():
    rng = np.random.default_rng(1)
    a = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
    R = NOISE + 100 * np.outer(a, a.conj())
    P, ku = nulling.whitened_projector(R, k=1)
    assert int(ku) == 1
    assert np.linalg.norm(P @ a) < 1e-4 * np.linalg.norm(a)
    out = P @ R @ P.conj().T
    assert np.real(np.trace(out)) < 0.1 * np.real(np.trace(R))


def test_auto_threshold_keeps_noise_only_channels():
    P, ku = nulling.whitened_projector(NOISE.astype(complex), thresh=2.0)
    assert int(ku) == 0
    np.testing.assert_allclose(P, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError):
        nulling.whitened_projector(NOISE, k=1, thresh=2.0)


def test_mvdr_filter_reproduces_mvdr_beam_power():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    F = A @ A.conj().T + np.eye(4)
    R = F + 0.3 * np.eye(4)
    d = np.exp(1j * rng.uniform(0, 6, 4))
    Q = nulling.mvdr_filter(F, 0.1, "median")
    delta = 0.1 * np.median(np.linalg.eigvalsh(F))
    Fi = np.linalg.inv(F + delta * np.eye(4))
    w = Fi @ d / (d.conj() @ Fi @ d)
    p_mvdr = np.real(w.conj() @ R @ w)
    p_vis = np.real(d.conj() @ (Q @ R @ Q) @ d) / np.abs(d.conj() @ Q @ d) ** 2
    assert p_vis == pytest.approx(p_mvdr, rel=1e-10)


def test_fringe_correlation_and_local_mean():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((50, 4, 4)) + 1j * rng.standard_normal((50, 4, 4))
    assert nulling.fringe_correlation(D, D) == pytest.approx(1.0)
    E = rng.standard_normal((50, 4, 4)) + 1j * rng.standard_normal((50, 4, 4))
    assert nulling.fringe_correlation(D, E) < 0.3
    L = nulling.local_mean(D, 2)
    np.testing.assert_allclose(L[10], D[8:13].mean(axis=0))
    np.testing.assert_allclose(L[0], D[0:3].mean(axis=0))


def test_voltage_nulling_matrix_on_raw_voltages():
    rng = np.random.default_rng(3)
    g = (1 + 0.3 * rng.standard_normal(4)) * np.exp(1j * rng.uniform(-np.pi, np.pi, 4))
    a = np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
    Rc = NOISE + 50 * np.outer(a, a.conj())                    # calibrated voltage covariance
    G = np.diag(np.conj(g))
    Rraw = G @ Rc @ G.conj().T
    P, _ = nulling.whitened_projector(Rc, k=1)
    M = nulling.voltage_nulling_matrix(P, g)
    np.testing.assert_allclose(M @ Rraw @ M.conj().T, P @ Rc @ P.conj().T, atol=1e-9)


def summary(stdout, band):
    line = next(l for l in stdout.splitlines() if l.startswith(band))
    raw = float(line.split("raw ")[1].split(",")[0])
    filt = float(line.split("filtered ")[1].split(" ")[0])
    return raw, filt


def test_fringe_recovery_projection_and_mvdr(cube, tmp_path):
    d = cube
    common = ["--vis", d / "cube.npz", "--gains", d / "gains.npy", "--gain-antennas", "0,4,32,36",
              "--positions", d / "pos.npy", "--bands", "357-380", "410-428", "381.5-399.5",
              "--examples", "390", "370"]
    res = run(S19, *common, "--filter", "project", "--k", "auto", "--out", tmp_path / "proj")
    assert res.returncode == 0, res.stderr
    raw, filt = summary(res.stdout, "357-380")
    assert raw < 0.3 and filt > 0.9                          # two emitters removed, Sun recovered
    raw_c, filt_c = summary(res.stdout, "381.5-399.5")
    assert abs(filt_c - raw_c) < 0.02                        # clean channels untouched
    assert "modes removed median 2" in next(l for l in res.stdout.splitlines() if l.startswith("357-380"))
    for suffix in ("_phase2d.png", "_correlation.png", "_examples.png", ".npz"):
        assert (tmp_path / f"proj{suffix}").exists()
    res = run(S19, *common, "--filter", "mvdr", "--loading-ref", "min", "--out", tmp_path / "mvdr")
    assert res.returncode == 0, res.stderr
    raw, filt = summary(res.stdout, "357-380")
    assert filt > raw + 0.5
    raw_c, filt_c = summary(res.stdout, "381.5-399.5")
    assert abs(filt_c - raw_c) < 0.05
    z = np.load(tmp_path / "mvdr.npz")
    assert z["rho"].shape == (2, 672) and str(z["filter"]) == "mvdr"


def test_nulling_gains_auto(cube, tmp_path):
    d = cube
    res = run(S20, "--vis", d / "cube.npz", "--gains", d / "gains.npy", "--gain-antennas", "0,4,32,36",
              "--auto", 2, "--rfi", "357-380", "410-428", "--out", tmp_path / "ng")
    assert res.returncode == 0, res.stderr
    clean = next(l for l in res.stdout.splitlines() if l.startswith("381.5-399.5"))
    assert "nulling-1 +0.0 dB, nulling-2 +0.0 dB" in clean and "0% / 0%" in clean
    muos = next(l for l in res.stdout.splitlines() if l.startswith("357-380"))
    d2 = float(muos.split("nulling-2 ")[1].split(" dB")[0])
    assert d2 < -10
    z = np.load(tmp_path / "ng.npz")
    assert z["M"].shape == (2, 672, 4, 4)
    assert (tmp_path / "ng.png").exists()


def test_figure_tools(cube, tmp_path):
    d = cube
    res = run(PH2D, "--vis", d / "cube.npz", "--gains", d / "gains.npy", "--gain-antennas", "0,4,32,36",
              "--pairs", "0,4", "0,32", "--zoom", "345-395", "--null-bands", "357-380", "--demean",
              "--out", tmp_path / "ph.png")
    assert res.returncode == 0, res.stderr
    assert "modes removed per channel (k = 0, 1, 2): [0, 0, 77]" in res.stdout
    assert (tmp_path / "ph.png").exists()
    f = 300 + 0.3 * np.arange(672)
    p = np.ones((2, T, 672))
    p[0][:, (f >= 357) & (f <= 380)] = 10.0
    np.savez(tmp_path / "real_nulling.npz", freqs=f, power=p, times=np.arange(T) * 10.24)
    res = run(BEAM, tmp_path / "real_nulling.npz", "--rfi", "357-380", "--out", tmp_path / "bp.png")
    assert res.returncode == 0, res.stderr
    assert "357-380 MHz: DFT +10.0 dB, MVDR +0.0 dB" in res.stdout
