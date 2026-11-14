"""Stale MVDR weights: charts_mvdr.cadence and scripts/21_weight_cadence.py on synthetic cubes.

Four antennas on a 4.2 m square, unequal noise, three bands: a fixed interferer (weights never
go stale), an interferer whose direction drifts (weights go stale with the lag) and a clean
band. The cube and gains are written in the charts-utils format-2 convention with nontrivial
gains, so the script calibrates before training.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr import cadence
from charts_mvdr.beamforming import steering_vectors

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "21_weight_cadence.py"
ANTS = [0, 4, 32, 36]
X = np.array([0.0, -4.2, 0.0, -4.2])
Y = np.array([0.0, 0.0, 4.2, 4.2])
T = 160
FREQS = 300.0 + 0.3 * np.arange(672)
FIXED = (FREQS >= 362) & (FREQS < 370)
DRIFT = (FREQS >= 415) & (FREQS < 417)
NOISE = np.array([1.0, 1.6, 1.2, 0.9])


def true_covariance(drift_per_frame=2e-4):
    """R (T, n_freq, 4, 4) in the raw-voltage convention; interferers at INR 30 dB."""
    R = np.zeros((T, FREQS.size, 4, 4), complex)
    R[:] = np.diag(NOISE)
    for c in np.flatnonzero(FIXED):
        b = steering_vectors(0.35, -0.6, FREQS[c] * 1e6, X, Y)[0]
        R[:, c] += 1000 * np.outer(b, b.conj())
    for c in np.flatnonzero(DRIFT):
        l = -0.2 + drift_per_frame * np.arange(T)
        b = steering_vectors(l, np.full(T, 0.5), FREQS[c] * 1e6, X, Y)       # (T, 4)
        R[:, c] += 1000 * b[:, :, None] * b.conj()[:, None, :]
    return R


def test_stationary_covariance_gives_no_excess():
    R = np.broadcast_to(np.diag(NOISE) + 50 * np.outer([1, 1j, -1, 1], [1, -1j, -1, 1]), (40, 3, 4, 4)).copy()
    a = np.ones((3, 4), complex)
    W = cadence.training_weights(R, a, 4, 0.1, "min")
    assert np.isnan(W[:4]).all() and np.isfinite(W[4:]).all()
    np.testing.assert_allclose(np.einsum("tci,ci->tc", W[4:].conj(), a), 1.0)          # w^H a = 1
    E = cadence.excess_power(R, W, [1, 5, 20])
    np.testing.assert_allclose(E[:, 24:], 1.0, rtol=1e-10)
    assert np.isnan(E[2, :20]).all()


def test_drifting_interferer_makes_weights_stale_and_fixed_one_does_not():
    R = true_covariance()
    a = np.ones((FREQS.size, 4), complex)
    sel = FIXED | DRIFT
    W = cadence.training_weights(R[:, sel], a[sel], 6, 0.1, "min")
    lags = [1, 6, 30, 100]
    E = cadence.excess_power(R[:, sel], W, lags)
    med, _, nch = cadence.band_excess_db(E, FREQS[sel], [(362, 370), (415, 417)])
    assert nch == [int(FIXED.sum()), int(DRIFT.sum())]
    assert np.all(np.abs(med[:, 0]) < 1e-6)                    # fixed interferer: never stale
    assert np.all(np.diff(med[:, 1]) > 0)                      # drifting: grows with the lag
    assert med[0, 1] < 0.5 and med[-1, 1] > 3.0
    assert cadence.max_lag_below([10, 60, 300, 1000], med[:, 1], 0.5) in (10, 60)
    assert np.isnan(cadence.max_lag_below([10, 60], [0.6, 0.2], 0.5))


def test_training_ignores_nonfinite_channels():
    R = true_covariance()[:, :3].copy()
    R[:, 1] = np.nan
    W = cadence.training_weights(R, np.ones((3, 4), complex), 6, 0.1)
    assert np.isnan(W[:, 1]).all() and np.isfinite(W[6:, [0, 2]]).all()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("cadence")
    rng = np.random.default_rng(5)
    tau = np.array([0.0, 0.4e-9, 5.4e-9, 2.3e-9])
    g = (1 + 0.3 * rng.random(4)) * np.exp(-2j * np.pi * FREQS[:, None] * 1e6 * tau)      # voltage gains
    R = true_covariance()
    Robs = g[None, :, :, None] * R * g.conj()[None, :, None, :]
    np.savez(d / "vis.npz", vis=np.moveaxis(Robs, (0, 1), (3, 2)).astype(np.complex64),
             times=1.7939e9 + 10.24 * np.arange(T), freqs=FREQS, ants=np.array(ANTS), vis_convention=2)
    np.savez(d / "gains.npz", gains=g.astype(np.complex64), gains_solved=g.astype(np.complex64),
             antennas=np.array(ANTS), freq_mhz=FREQS, format_version=2, ref_ant=0)
    pos = np.zeros((64, 3))
    pos[ANTS, 0], pos[ANTS, 1] = X, Y
    np.save(d / "pos.npy", pos)
    return d


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_script_end_to_end(files, tmp_path):
    r = run("--vis", files / "vis.npz", "--gains", files / "gains.npz", "--bands", "362-370", "415-417", "330-340",
            "--loading", "0.1", "--loading-ref", "min", "--lags", 1, 6, 30, 100, "--threshold-db", 0.5,
            "--out", tmp_path / "cad")
    assert r.returncode == 0, r.stderr
    out = json.loads((tmp_path / "cad.json").read_text())
    e = out["loadings"]["0.1"]
    assert out["lags_s"] == pytest.approx([10.24, 61.44, 307.2, 1024.0])
    assert e["362-370"]["max_lag_s_below_threshold"] == pytest.approx(1024.0)
    assert e["330-340"]["max_lag_s_below_threshold"] == pytest.approx(1024.0)
    assert e["415-417"]["max_lag_s_below_threshold"] in (pytest.approx(10.24), pytest.approx(61.44))
    assert e["362-370"]["mvdr_minus_dft_db"] < -10                     # MVDR removes the interferer
    assert abs(e["330-340"]["mvdr_minus_dft_db"]) < 3                  # clean band: little to remove
    assert (tmp_path / "cad_loading0.1.npz").exists()
    assert "> 1024 s" in r.stdout


def test_script_offzenith_look_matches_zenith_for_the_fixed_band(files, tmp_path):
    r = run("--vis", files / "vis.npz", "--gains", files / "gains.npz", "--look", "0.1,0.05",
            "--positions", files / "pos.npy", "--bands", "362-370", "--loading", "0.1", "--loading-ref", "min",
            "--lags", 1, 30, "--out", tmp_path / "off")
    assert r.returncode == 0, r.stderr
    e = json.loads((tmp_path / "off.json").read_text())["loadings"]["0.1"]["362-370"]
    assert max(abs(v) for v in e["median_db"]) < 1e-3


def test_script_errors(files, tmp_path):
    r = run("--vis", files / "vis.npz", "--gains", files / "gains.npz", "--look", "0.1,0.1", "--out", tmp_path / "x")
    assert r.returncode == 1 and "--positions" in r.stderr
    r = run("--vis", files / "vis.npz", "--gains", files / "gains.npz", "--bands", "380-370", "--out", tmp_path / "x")
    assert r.returncode == 1 and "high edge" in r.stderr
