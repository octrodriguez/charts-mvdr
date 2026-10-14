"""Tests for charts_mvdr.beamforming: MVDR diagonal loading references."""

import numpy as np
import pytest

from charts_mvdr import sim_cpu as sc
from charts_mvdr.beamforming import mvdr_weights, noise_floor

F0 = 344.1e6
SUN, RFI = (0.47, 0.48), (-0.830, -0.482)


def covariance(rfi_power, sigma=1.0):
    scn = sc.Scenario(freqs_hz=[F0], noise_sigma=sigma,
                      sources=[sc.Source(*SUN, power=1.0, name="sun"),
                               sc.Source(*RFI, power=rfi_power, name="rfi")])
    return sc.analytic_covariance(scn)[0], sc.steer(F0, *SUN, scn.positions)


def explicit(R, a, delta):
    Ri_a = np.linalg.solve(R + delta * np.eye(R.shape[0]), a)
    return Ri_a / np.vdot(a, Ri_a)


def test_noise_floor_is_sigma_n2_for_few_sources():
    R, _ = covariance(100.0, sigma=1.3)
    assert noise_floor(R) == pytest.approx(2 * 1.3**2)


@pytest.mark.parametrize("ref,delta_fn", [
    ("trace", lambda R: 0.1 * np.real(np.trace(R)) / R.shape[0]),
    ("noise", lambda R: 0.1 * noise_floor(R)),
    ("absolute", lambda R: 0.1),
])
def test_loading_references(ref, delta_fn):
    R, a = covariance(50.0)
    w = mvdr_weights(R, a, loading=0.1, loading_ref=ref)[0]
    np.testing.assert_allclose(w, explicit(R, a, delta_fn(R)), rtol=1e-10, atol=1e-12)


def test_default_is_backward_compatible():
    R, a = covariance(50.0)
    np.testing.assert_allclose(mvdr_weights(R, a, 0.05)[0],
                               mvdr_weights(R, a, 0.05, loading_ref="trace")[0])


def test_noise_reference_independent_of_interference_power():
    """Same loading in noise units gives the same delta whatever the INR;
    the trace reference scales with the interferer power."""
    deltas_noise, deltas_trace = [], []
    for p in (1.0, 100.0, 10000.0):
        R, _ = covariance(p)
        deltas_noise.append(10 * noise_floor(R))
        deltas_trace.append(10 * np.real(np.trace(R)) / R.shape[0])
    assert np.ptp(deltas_noise) < 1e-9
    assert deltas_trace[-1] / deltas_trace[0] > 1000


def test_noise_loading_keeps_null():
    R, a = covariance(1e4)                       # INR 37 dB per element
    w = mvdr_weights(R, a, loading=10, loading_ref="noise")[0]
    a_rfi = sc.steer(F0, *RFI, sc.grid_positions())
    assert 10 * np.log10(abs(np.vdot(w, a_rfi)) ** 2) < -40
    assert abs(np.vdot(w, a)) == pytest.approx(1.0)


def test_invalid_reference():
    R, a = covariance(1.0)
    with pytest.raises(ValueError):
        mvdr_weights(R, a, 0.1, loading_ref="median")
