"""Running-window filters in scripts/19_fringe_recovery.py: noise-floor debiasing, held updates,
--only-bands; and nulling.noise_floor.

Synthetic cube: four antennas on a 4.2 m square, a weak Sun (0.1 of the noise) drifting across
the sky, and in 360-365 MHz an interferer 200 times the noise that jumps between three
directions, more states than the whole-cube projector can remove (at most n - 2 = 2 modes).
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
ANTS = [0, 4, 32, 36]
X = np.array([0.0, -4.2, 0.0, -4.2])
Y = np.array([0.0, 0.0, 4.2, 4.2])
T = 120
FREQS = 300.0 + 0.3 * np.arange(672)
BAND = (FREQS >= 360) & (FREQS <= 365)
CLEAN = (FREQS >= 330) & (FREQS <= 335)


def test_noise_floor_recovers_the_diagonal_of_a_factor_model():
    rng = np.random.default_rng(2)
    L = rng.standard_normal((4, 1)) + 1j * rng.standard_normal((4, 1))
    psi = np.array([1.0, 2.0, 0.5, 1.5])
    R = 50 * L @ L.conj().T + np.diag(psi)
    np.testing.assert_allclose(np.diag(nulling.noise_floor(R, 1)).real, psi, rtol=1e-6)
    np.testing.assert_allclose(nulling.noise_floor(R, 0), np.diag(np.diag(R)))
    stack = nulling.noise_floor(np.stack([R, R]), np.array([1, 0]))
    assert stack.shape == (2, 4, 4)
    np.testing.assert_allclose(np.diag(stack[0]).real, psi, rtol=1e-6)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("fringe_window")
    look = np.stack([np.linspace(0.25, -0.25, T), np.full(T, 0.28)], 1)
    R = np.zeros((T, FREQS.size, 4, 4), complex)
    R[:] = np.diag([1.0, 1.3, 1.1, 0.9])
    for c in np.flatnonzero(BAND | CLEAN):
        s = steering_vectors(look[:, 0], look[:, 1], FREQS[c] * 1e6, X, Y)
        R[:, c] += 0.1 * s[:, :, None] * s.conj()[:, None, :]
    for c in np.flatnonzero(BAND):
        for k3, (l, m) in enumerate([(0.3, -0.6), (-0.5, 0.2), (0.1, 0.7)]):
            b = steering_vectors(l, m, FREQS[c] * 1e6, X, Y)[0]
            R[k3 * T // 3:(k3 + 1) * T // 3, c] += 200 * np.outer(b, b.conj())
    # version-1 cube (conjugate of the raw-voltage covariance), with the Sun direction stored
    np.savez(d / "vis.npz", vis=np.moveaxis(np.conj(R), (0, 1), (3, 2)).astype(np.complex64),
             times=1.7939e9 + 10.24 * np.arange(T), freqs=FREQS, ants=np.array(ANTS), look=look)
    np.save(d / "gains.npy", np.ones((FREQS.size, 4), complex))
    pos = np.zeros((64, 3))
    pos[ANTS, 0], pos[ANTS, 1] = X, Y
    np.save(d / "pos.npy", pos)
    return d


def run(files, out, *extra):
    r = subprocess.run([sys.executable, str(S19), "--vis", str(files / "vis.npz"), "--gains", str(files / "gains.npy"),
                        "--gain-antennas", "0,4,32,36", "--gains-format", "1", "--positions", str(files / "pos.npy"),
                        "--bands", "360-365", "330-335", "--only-bands", "--out", str(out), *map(str, extra)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.load(f"{out}.npz")


def med(z, mask, row=1):
    return float(np.nanmedian(z["rho"][row][mask]))


def test_running_window_recovers_a_jumping_interferer(files, tmp_path):
    whole = run(files, tmp_path / "w0", "--filter", "project", "--k", "auto")
    win = run(files, tmp_path / "w3", "--filter", "project", "--k", "auto", "--window", 3)
    biased = run(files, tmp_path / "w3b", "--filter", "project", "--k", "auto", "--window", 3, "--no-debias")
    assert med(whole, BAND) < 0.2                       # three states, at most two modes removed
    assert med(win, BAND) > 0.9                         # running window with the noise floor subtracted
    assert med(biased, BAND) < 0.7                      # without debiasing the metric is pulled down
    for z in (whole, win, biased):
        assert med(z, CLEAN) == pytest.approx(med(z, CLEAN, row=0), abs=1e-6)   # clean band untouched
    assert bool(win["debias"]) and not bool(biased["debias"]) and not bool(whole["debias"])
    outside = ~(BAND | CLEAN)
    assert np.isnan(win["rho"][:, outside]).all()       # --only-bands


def test_held_updates_let_the_jumps_through(files, tmp_path):
    held = run(files, tmp_path / "u6", "--filter", "project", "--k", "auto", "--window", 3, "--update-every", 6)
    assert med(held, BAND) < 0.5                        # a few stale frames after each jump dominate
    assert int(held["update_every"]) == 6


def test_mvdr_filter_with_window_runs_and_keeps_the_clean_band(files, tmp_path):
    z = run(files, tmp_path / "mv", "--filter", "mvdr", "--window", 3)
    assert med(z, CLEAN) > 0.9 and med(z, BAND) > med(z, BAND, row=0)


def test_option_errors(files, tmp_path):
    base = [sys.executable, str(S19), "--vis", str(files / "vis.npz"), "--gains", str(files / "gains.npy"),
            "--gain-antennas", "0,4,32,36", "--gains-format", "1", "--positions", str(files / "pos.npy"),
            "--out", str(tmp_path / "x")]
    r = subprocess.run(base + ["--update-every", "0"], capture_output=True, text=True)
    assert r.returncode == 1 and "--update-every" in r.stderr
    r = subprocess.run(base + ["--only-bands"], capture_output=True, text=True)
    assert r.returncode == 1 and "--only-bands" in r.stderr
