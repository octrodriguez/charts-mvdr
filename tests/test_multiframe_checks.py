"""Tests for the multi-frame checks (#9 stationarity, #10 on/off-pulse)."""

import numpy as np
import pytest

from charts_mvdr import checks as ck
from charts_mvdr import sim_cpu as sc
from charts_mvdr.np_correlator import correlate

F0, K = 344e6, 1024
POS = sc.grid_positions(4, 4)
LOOK, RFI, NEW = (0.47, 0.48), (-0.83, -0.48), (0.1, -0.3)


def cov(seed, signal=1.0, rfi=49.0, new=0.0, quantize=None):
    src = [sc.Source(*LOOK, signal), sc.Source(*RFI, rfi)]
    if new:
        src.append(sc.Source(*NEW, new))
    scn = sc.Scenario(freqs_hz=[F0], positions=POS, noise_sigma=2.0, sources=src)
    V = sc.simulate_voltages(scn, K, rng=seed, quantize=quantize)
    return correlate(V.astype(np.complex128), normalize=True)[0]


def stationarity(Rs):
    M = len(Rs)
    Rm = np.mean(Rs, axis=0)
    zm = [ck.whitened_change(R, Rm, Rm, K) / (1 - 1 / M) for R in Rs]
    zc = [ck.whitened_change(Rs[i], Rs[i + 1], 0.5 * (Rs[i] + Rs[i + 1]), K / 2)
          for i in range(M - 1)]
    return ck.check_stationarity(zm, zc)


def test_whitened_change_expectation_is_one():
    scn = sc.Scenario(freqs_hz=[F0], positions=POS, noise_sigma=2.0,
                      sources=[sc.Source(*RFI, 400.0)])
    R = sc.analytic_covariance(scn)[0]
    z = [ck.whitened_change(cov(s, signal=0.0, rfi=400.0), R, R, K) for s in range(20)]
    assert np.mean(z) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("quantize", [None, "symmetric"])
def test_stationary_scene_passes(quantize):
    res = stationarity([cov(s, quantize=quantize) for s in range(30)])
    assert res.status == ck.Status.PASS, res.note


def test_step_change_is_located():
    Rs = [cov(s, new=4.0 if s >= 15 else 0.0) for s in range(30)]
    res = stationarity(Rs)
    assert res.status != ck.Status.PASS
    assert res.details["worst_pair"] == 14


def test_on_off_signal_in_look_direction_passes():
    off = [cov(s) for s in range(20)]
    on = np.mean([cov(100 + s, signal=5.0) for s in range(3)], axis=0)
    d = sc.steer(F0, *LOOK, POS)
    res = ck.check_on_off(on, 3 * K, np.mean(off[:10], 0), np.mean(off[10:], 0), 10 * K, d)
    assert res.status == ck.Status.PASS, res.note
    assert res.details["alpha"] == pytest.approx(4.0, rel=0.1)


def test_on_off_new_interferer_fails():
    off = [cov(s) for s in range(20)]
    on = np.mean([cov(100 + s, new=1.0) for s in range(3)], axis=0)
    d = sc.steer(F0, *LOOK, POS)
    res = ck.check_on_off(on, 3 * K, np.mean(off[:10], 0), np.mean(off[10:], 0), 10 * K, d)
    assert res.status == ck.Status.FAIL, res.note
