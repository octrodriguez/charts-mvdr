"""Tests of charts_mvdr.tracking and scripts/17_real_nulling.py on synthetic covariances."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr import tracking
from charts_mvdr.beamforming import mvdr_weights, steering_vectors

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_real_nulling.py"
X = np.array([0.0, -0.6, -3.6, -4.2])
Y = np.zeros(4)
T, K = 60, 4000


def synthetic(freqs_mhz, rfi_chans, look_lm, gains, seed=0, p_src=10.0, p_rfi=1000.0, rfi_lm=(0.55, 0.0)):
    """Covariances (4, 4, n_chan, T) from voltages: look source + fixed interferer + noise."""
    rng = np.random.default_rng(seed)
    n_chan = len(freqs_mhz)
    vis = np.zeros((4, 4, n_chan, T), dtype=np.complex64)
    for c, f in enumerate(freqs_mhz * 1e6):
        ai = steering_vectors(*rfi_lm, f, X, Y)[0]
        for t in range(T):
            a = steering_vectors(look_lm[t, 0], look_lm[t, 1], f, X, Y)[0]
            s = np.sqrt(p_src / 2) * (rng.standard_normal(K) + 1j * rng.standard_normal(K))
            v = a[:, None] * s[None, :]
            if c in rfi_chans:
                q = np.sqrt(p_rfi / 2) * (rng.standard_normal(K) + 1j * rng.standard_normal(K))
                v = v + ai[:, None] * q[None, :]
            v = v + np.sqrt(0.5) * (rng.standard_normal((4, K)) + 1j * rng.standard_normal((4, K)))
            v = gains[c][:, None] * v
            vis[:, :, c, t] = v @ v.conj().T / K
    return vis


def setup(seed=1):
    rng = np.random.default_rng(seed)
    freqs = np.array([380.0, 385.0, 390.0, 395.0, 400.0, 405.0])
    tau = np.array([0.0, -5.6e-9, 0.1e-9, -2.1e-9])
    amp = np.array([1.0, 2.0, 0.5, 1.5])
    gains = amp[None, :] * np.exp(2j * np.pi * freqs[:, None] * 1e6 * tau[None, :])
    look = np.stack([np.linspace(0.20, 0.25, T), np.full(T, 0.3)], axis=1)
    return freqs, gains, look, rng


def test_batched_mvdr_matches_reference():
    rng = np.random.default_rng(2)
    F = np.empty((3, 4, 4), complex)
    a = np.exp(1j * rng.uniform(0, 2 * np.pi, (3, 4)))
    for c in range(3):
        M = rng.standard_normal((4, 6)) + 1j * rng.standard_normal((4, 6))
        F[c] = M @ M.conj().T
    w = tracking.batched_mvdr(F, a, 10.0)
    for c in range(3):
        ref = mvdr_weights(F[c], a[c][None], loading=10.0, loading_ref="noise")[0]
        assert np.allclose(w[c], ref)
    assert np.allclose(np.einsum("ci,ci->c", w.conj(), a), 1.0)


def test_tracking_nulls_rfi_and_keeps_look_source():
    freqs, gains, look, _ = setup()
    vis = synthetic(freqs, rfi_chans={4, 5}, look_lm=look, gains=gains)
    r = tracking.run_tracking(vis, freqs * 1e6, X, Y, look, gains=gains, half_window=10, guard=1)
    p = 10 * np.log10(np.nanmedian(r["power"], axis=1))          # (2, n_chan) dB: dft, mvdr
    ratio = p[1] - p[0]
    assert np.all(np.abs(ratio[:4]) < 0.5)            # look source kept in clean channels
    assert np.all(p[0, 4:] > p[0, :4].max() + 3)      # DFT picks up the interferer
    assert np.all(np.abs(p[1, 4:] - p[1, :4].mean()) < 0.3)   # MVDR removes it completely
    assert np.all(np.nanmedian(r["wng"], axis=0) >= 1 - 1e-9)
    wrong = tracking.run_tracking(vis, freqs * 1e6, X, Y, look, gains=gains, sign=-1)
    assert np.nanmedian(wrong["power"][0][:, :4]) < 0.5 * np.nanmedian(r["power"][0][:, :4])
    uncal = tracking.run_tracking(vis, freqs * 1e6, X, Y, look, gains=None)
    assert np.nanmedian(uncal["power"][0][:, :4]) < np.nanmedian(r["power"][0][:, :4])


def test_interpolate_gains_linear_phase():
    freqs = np.linspace(300, 500, 201)
    g = (1 + 0.001 * freqs)[:, None] * np.exp(2j * np.pi * freqs[:, None] * 1e6 * np.array([[0.0, -5.6e-9]]))
    gap = g.copy()
    gap[60:100] = np.nan
    gap[:5, 1] = np.nan
    out, filled = tracking.interpolate_gains(gap, freqs)
    assert filled[60:100].all() and filled[:5, 1].all() and not filled[5:60].any()
    assert np.allclose(out[60:100], g[60:100], rtol=1e-9, atol=1e-9)
    with pytest.raises(ValueError):
        tracking.interpolate_gains(np.full((5, 1), np.nan + 0j), freqs[:5])


def test_write_sidecar(tmp_path):
    p = tracking.write_sidecar(tmp_path / "gains_20261027T135000Z.npy", calibrator="sun", ref_ant=0,
                               antennas=[0, 5], notes="test")
    doc = json.loads(p.read_text())
    assert p.name == "gains_20261027T135000Z.json"
    assert doc["calibrator"] == "sun" and doc["valid_from_utc"] is None and doc["notes"] == "test"
    with pytest.raises(ValueError):
        tracking.write_sidecar(tmp_path / "g.npy", bogus=1)


def test_script_end_to_end(tmp_path):
    freqs, gains, _, _ = setup()
    look = np.tile([0.22, 0.3], (T, 1))
    vis = synthetic(freqs, rfi_chans={4, 5}, look_lm=look, gains=gains)
    vis5 = np.zeros((5, 5) + vis.shape[2:], dtype=vis.dtype)
    vis5[:4, :4] = vis                                 # a fifth antenna without gains
    np.savez(tmp_path / "vis.npz", vis=vis5, freqs=freqs, times=np.arange(T) * 41.0, ants=np.array([0, 5, 32, 37, 40]))
    g = gains.astype(np.complex64).copy()
    g[4:] = np.nan                                     # RFI channels flagged in the calibration
    np.save(tmp_path / "gains.npy", g)
    pos = np.zeros((64, 3))
    pos[[0, 5, 32, 37], 0] = X
    np.save(tmp_path / "pos.npy", pos)
    r = subprocess.run([sys.executable, str(SCRIPT), "--vis", str(tmp_path / "vis.npz"), "--gains", str(tmp_path / "gains.npy"),
                        "--gain-antennas", "0,5,32,37", "--positions", str(tmp_path / "pos.npy"), "--look", "0.22,0.3",
                        "--interp-gains", "--rfi", "398-410", "--clean", "378-397", "--out", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "-> +1" in r.stdout
    z = np.load(tmp_path / "out" / "real_nulling.npz")
    assert z["interp"].tolist() == [False] * 4 + [True] * 2
    assert np.all(np.abs(z["ratio_db"][:4]) < 0.5) and np.all(z["ratio_db"][4:] < -3)
    pm = 10 * np.log10(np.nanmedian(z["power"][1], axis=0))
    assert np.all(np.abs(pm[4:] - pm[:4].mean()) < 0.3)  # interpolated gains: interferer removed
    assert (tmp_path / "out" / "real_nulling.png").exists()
    bad = subprocess.run([sys.executable, str(SCRIPT), "--vis", str(tmp_path / "vis.npz"), "--gains", str(tmp_path / "gains.npy"),
                          "--positions", str(tmp_path / "pos.npy"), "--out", str(tmp_path / "x")], capture_output=True, text=True)
    assert bad.returncode == 1 and "--gain-antennas" in bad.stderr


def test_loading_ref_min_restores_nulling():
    """With N = 4 and two strong interferers the median eigenvalue is not noise:
    loading relative to it collapses MVDR to DFT, loading relative to the minimum does not."""
    freqs, gains, look, _ = setup()
    rng = np.random.default_rng(7)
    vis = synthetic(freqs, rfi_chans={4, 5}, look_lm=look, gains=gains, p_rfi=1000.0)
    extra = synthetic(freqs, rfi_chans={4, 5}, look_lm=look, gains=gains, seed=9, p_src=0.0,
                      p_rfi=1000.0, rfi_lm=(-0.6, 0.0)) - synthetic(freqs, rfi_chans=set(), look_lm=look,
                                                                     gains=gains, seed=9, p_src=0.0)
    vis = vis + extra                                  # second interferer, noise counted once
    kw = dict(gains=gains, half_window=10, guard=1, loading=10.0)
    med = tracking.run_tracking(vis, freqs * 1e6, X, Y, look, loading_ref="median", **kw)
    mn = tracking.run_tracking(vis, freqs * 1e6, X, Y, look, loading_ref="min", **kw)
    r_med = np.nanmedian(10 * np.log10(med["power"][1] / med["power"][0]), axis=0)
    r_min = np.nanmedian(10 * np.log10(mn["power"][1] / mn["power"][0]), axis=0)
    assert np.all(r_min[4:] < r_med[4:] - 3)           # deeper rejection with min reference
    assert np.all(np.abs(r_min[:4]) < 0.5)             # look source still kept
    with pytest.raises(ValueError):
        tracking.batched_mvdr(np.eye(4)[None], np.ones((1, 4)), 1.0, loading_ref="trace")


def test_fig_real_nulling(tmp_path):
    tool = Path(__file__).resolve().parents[1] / "tools" / "fig_real_nulling.py"
    f = np.linspace(360, 430, 50)
    for name, depth in (("a", -1.0), ("b", -6.0)):
        np.savez(tmp_path / f"{name}.npz", freqs=f, ratio_db=np.where(f > 415, depth, 0.0),
                 wng_db=np.zeros(50), interp=f > 415)
    r = subprocess.run([sys.executable, str(tool), f"median={tmp_path / 'a.npz'}",
                        f"min={tmp_path / 'b.npz'}", "--rfi", "415-430", "--clean", "380-400",
                        "--out", str(tmp_path / "fig.png")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig.png").exists()
    lines = [ln for ln in r.stdout.splitlines() if "415-430 MHz" in ln]
    assert len(lines) == 2 and "-1.0 dB" in lines[0] and "-6.0 dB" in lines[1]
