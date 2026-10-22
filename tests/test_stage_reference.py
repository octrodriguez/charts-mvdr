import numpy as np
import pytest

from charts_mvdr.beamforming import mvdr_weights, steering_vectors
from charts_mvdr.stage_reference import (StageConfig, correct_gains, load_gains, run_stage,
                                         window_indices)

F0 = 350e6
KS = 1536


def grid(n=4, d=0.6):
    iy, ix = np.divmod(np.arange(n * n), n)
    return -d * ix.astype(float), d * iy.astype(float)


def scene(n_frames=12, n_chan=2, burst_frame=None, rfi=100.0, sig=0.05, seed=0):
    """Expected covariances (times KS) of noise + RFI + optional burst in one frame."""
    x, y = grid()
    n = x.size
    a_r = steering_vectors(0.5, 0.2, F0, x, y)[0]
    a_s = steering_vectors(-0.1, 0.25, F0, x, y)[0]
    rng = np.random.default_rng(seed)
    noise_c = [np.diag(1.0 + 0.01 * rng.random(n)) for _ in range(n_chan)]
    S = np.empty((n_frames, n_chan, n, n), complex)
    for t in range(n_frames):
        for c in range(n_chan):
            noise = noise_c[c]
            R = noise + rfi * np.outer(a_r, a_r.conj())
            if t == burst_frame:
                R = R + sig * np.outer(a_s, a_s.conj())
            S[t, c] = R * KS
    return S, x, y


def cfg(x, y, **kw):
    return StageConfig(beams=[(-0.1, 0.25)], positions_x_m=x, positions_y_m=y,
                       samples_per_frame=KS, **kw)


def test_window_indices():
    assert window_indices(5, 20, 3, 1).tolist() == [2, 3, 7, 8]
    assert window_indices(0, 20, 3, 1).tolist() == [2, 3]
    assert window_indices(19, 20, 3, 0).tolist() == [16, 17, 18]
    with pytest.raises(ValueError):
        window_indices(5, 20, 2, 2)


def test_matches_mvdr_weights_on_stationary_data():
    S, x, y = scene(burst_frame=None)
    out = run_stage(S, [F0, F0], cfg(x, y, half_window=3, guard=1, min_frames=2))
    R = S[0, 0] / KS
    d = steering_vectors(-0.1, 0.25, F0, x, y)
    w = mvdr_weights(R, d, loading=10.0, loading_ref="noise")[0]
    np.testing.assert_allclose(out.weights[0][0, 0], w, rtol=1e-10, atol=1e-12)
    p = np.real(w.conj() @ R @ w)
    np.testing.assert_allclose(out.power[1, :, 0, 0], p, rtol=1e-10)
    assert out.update_frames == list(range(12))
    assert np.all(out.wng[0] >= 1.0 - 1e-12)


def test_burst_is_not_self_nulled_and_rfi_is_removed():
    sig = 0.05
    S, x, y = scene(burst_frame=6, sig=sig)
    out = run_stage(S, [F0, F0], cfg(x, y, half_window=4, guard=1))
    dft, mvdr = out.power[:, :, 0, 0]
    # distortionless: the burst adds its own power on top of the residual
    np.testing.assert_allclose(mvdr[6] - mvdr[5], sig, rtol=0.02)
    # the DFT background contains RFI leakage, the MVDR one is near the noise level
    assert mvdr[5] < 0.2 * dft[5]


def test_signal_in_window_would_reduce_burst():
    """Control: with guard 0 the burst frame is excluded only from itself; frames
    next to it see it in F, but the burst frame itself is still protected."""
    S, x, y = scene(burst_frame=6, sig=0.05)
    out = run_stage(S, [F0, F0], cfg(x, y, half_window=4, guard=0))
    mvdr = out.power[1, :, 0, 0]
    np.testing.assert_allclose(mvdr[6] - mvdr[3], 0.05, rtol=0.02)


def test_flags_before_first_update_and_update_every():
    S, x, y = scene(n_frames=12)
    out = run_stage(S, [F0, F0], cfg(x, y, half_window=4, guard=1, min_frames=5,
                                     update_every=4))
    # windows: t=0 -> {2,3,4}; t=1 -> {3,4,5}; t=2 -> {0,4,5,6}; t=3 -> {0,1,5,6,7}
    assert out.n_window.tolist()[:4] == [3, 3, 4, 5]
    assert np.isnan(out.power[1, :3]).all()
    assert np.isfinite(out.power[1, 3:]).all()
    assert out.update_frames[0] == 3
    assert out.update_frames[1:] == [4, 8]
    assert np.isfinite(out.power[0]).all()


def test_gains_are_removed():
    S, x, y = scene()
    rng = np.random.default_rng(3)
    n = x.size
    g = (1 + 0.3 * rng.random((2, n))) * np.exp(1j * rng.uniform(-np.pi, np.pi, (2, n)))
    Sg = S * g[None, :, :, None] * g.conj()[None, :, None, :]
    ref = run_stage(S, [F0, F0], cfg(x, y))
    out = run_stage(Sg, [F0, F0], cfg(x, y), gains=g)
    np.testing.assert_allclose(out.power, ref.power, rtol=1e-9)


def test_nan_gain_drops_antenna():
    S, x, y = scene()
    n = x.size
    g = np.ones((2, n), complex)
    g[0, 5] = np.nan
    out = run_stage(S, [F0, F0], cfg(x, y), gains=g)
    assert np.all(out.weights[0][0, :, 5] == 0)
    keep = np.r_[0:5, 6:n]
    sub = run_stage(S[:, :, keep][:, :, :, keep], [F0, F0], cfg(x[keep], y[keep]))
    np.testing.assert_allclose(out.power[:, :, 0], sub.power[:, :, 0], rtol=1e-10)
    np.testing.assert_allclose(out.weights[0][0][:, keep], sub.weights[0][0], rtol=1e-10)
    np.testing.assert_allclose(out.wng[0][0], sub.wng[0][0], rtol=1e-10)


def test_correct_gains_convention():
    """Same convention as charts_utils apply_gains: V_cal = V / (g_i g_j*)."""
    g = np.array([2.0, 1j, 0.5 - 0.5j])
    R = np.arange(9).reshape(3, 3) + 1j
    np.testing.assert_allclose(correct_gains(R, g), R / np.outer(g, g.conj()))


def test_load_gains(tmp_path):
    g = (np.arange(12) + 1j).reshape(4, 3).astype(np.complex64)
    np.save(tmp_path / "g.npy", g)
    out = load_gains(tmp_path / "g.npy", channels=[1, 3], n_ant=3)
    assert out.dtype == np.complex128
    np.testing.assert_array_equal(out, g[[1, 3]])
    np.save(tmp_path / "t.npy", np.zeros((4, 3, 2), complex))
    with pytest.raises(ValueError):
        load_gains(tmp_path / "t.npy")
    with pytest.raises(ValueError):
        load_gains(tmp_path / "g.npy", n_ant=4)


def test_load_correlation_stack_roundtrip(tmp_path):
    import importlib.util
    from pathlib import Path

    from charts_mvdr.stage_reference import load_correlation_stack

    spec = importlib.util.spec_from_file_location(
        "tdio", Path(__file__).with_name("test_data_io.py"))
    tdio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdio)
    rng = np.random.default_rng(1)
    Rs = []
    for k in range(3):
        V = rng.integers(-7, 8, (4, 3, 50)) + 1j * rng.integers(-7, 8, (4, 3, 50))
        R = np.einsum("fit,fjt->fij", V.transpose(1, 0, 2), V.transpose(1, 0, 2).conj())
        tdio.write_astron(tmp_path / f"corr_{k:07d}.bin", R)
        Rs.append(R)
    idx, stack = load_correlation_stack(tmp_path, num_freq=3, num_ant=4, channels=[2, 0])
    assert idx == [0, 1, 2]
    np.testing.assert_allclose(stack, np.stack(Rs)[:, [2, 0]])


def test_script_end_to_end(tmp_path):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    import h5py

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "tdio", Path(__file__).with_name("test_data_io.py"))
    tdio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tdio)

    S, x, y = scene(n_frames=8, n_chan=3, burst_frame=4, sig=0.05)
    win, corr = tmp_path / "win", tmp_path / "corr"
    win.mkdir()
    corr.mkdir()
    with h5py.File(win / "w_meta.h5", "w") as f:
        f["antenna_pos_x_m"], f["antenna_pos_y_m"] = x, y
        f["frequencies_mhz"] = np.full(3, F0 / 1e6)
    for t in range(8):
        tdio.write_astron(corr / f"corr_{t:07d}.bin", S[t])
    out = tmp_path / "ref.npz"
    r = subprocess.run([sys.executable, str(root / "scripts" / "12_stage_reference.py"),
                        "--corr-dir", str(corr), "--window-dir", str(win),
                        "--num-freq", "3", "--num-ant", str(x.size), "--half-window", "3",
                        "--chunk", "2", "--beams", "-0.1", "0.25", "0.5", "0.2",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = np.load(out)
    assert z["power"].shape == (2, 8, 3, 2)
    assert z["weights"].shape == (8, 3, 2, x.size)
    # integer rounding in the dumps: compare with the in-memory reference loosely
    ref = run_stage(S, np.full(3, F0), cfg(x, y, half_window=3))
    np.testing.assert_allclose(z["power"][:, :, :, 0], ref.power[:, :, :, 0], rtol=1e-3)
    # the second beam looks at the interferer: MVDR keeps it (distortionless)
    assert np.all(z["power"][1, :, :, 1] > 50)
