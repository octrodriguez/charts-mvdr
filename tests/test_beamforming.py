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


# --- look refinement and power budget ----------------------------------------

from charts_mvdr.beamforming import (beam_power, dft_weights, fit_source_powers,  # noqa: E402
                                     power_budget, refine_peak, steering_vectors)

LOOK_TRUE = (0.4677, 0.4801)


def scene(noise_per_ant=None, rfi_power=30.0):
    scn = sc.Scenario(freqs_hz=[F0], noise_sigma=1.7,
                      sources=[sc.Source(*LOOK_TRUE, power=0.45, name="sun"),
                               sc.Source(*RFI, power=rfi_power, name="rfi")])
    R = sc.analytic_covariance(scn)[0]
    if noise_per_ant is not None:
        R = R + np.diag(noise_per_ant)
    x, y = scn.positions[:, 0], scn.positions[:, 1]
    return R, x, y


def test_refine_peak_recovers_off_grid_source():
    # Clean channel, as in 05: in an RFI channel the DFT sidelobes of the
    # interferer shift the peak (by ~0.015 in l, m for this scene).
    R, x, y = scene(rfi_power=0.0)
    def dft_power(L, M):
        return beam_power(R, dft_weights(steering_vectors(L, M, F0, x, y)))
    l, m = refine_peak(dft_power, 0.4625, 0.475, step=0.0125)
    assert abs(l - LOOK_TRUE[0]) < 2e-4 and abs(m - LOOK_TRUE[1]) < 2e-4


def test_fit_source_powers_ignores_per_antenna_noise():
    extra = np.linspace(0.0, 3.0, 64)
    R, x, y = scene(noise_per_ant=extra)
    A = steering_vectors([LOOK_TRUE[0], RFI[0]], [LOOK_TRUE[1], RFI[1]], F0, x, y)
    p, noise = fit_source_powers(R, A)
    np.testing.assert_allclose(p, [0.45, 30.0], rtol=1e-9)
    np.testing.assert_allclose(noise, 2 * 1.7**2 + extra, rtol=1e-9)


@pytest.mark.parametrize("beam", ["dft", "mvdr"])
def test_power_budget_closes_with_exact_model(beam):
    R, x, y = scene(noise_per_ant=np.linspace(0.0, 3.0, 64))
    A = steering_vectors([LOOK_TRUE[0], RFI[0]], [LOOK_TRUE[1], RFI[1]], F0, x, y)
    d = A[:1]
    w = dft_weights(d)[0] if beam == "dft" else mvdr_weights(R, d)[0]
    p, noise = fit_source_powers(R, A)
    b = power_budget(R, w, A, p, noise)
    assert b["residual"] == pytest.approx(0.0, abs=1e-9)
    assert b["source"][0] == pytest.approx(0.45)          # distortionless: |w^H d| = 1
    assert b["wng"] >= 1.0 - 1e-12
    if beam == "mvdr":
        assert b["source"][1] < 1e-6 * b["source"][0]     # interferer nulled


# --- beam_response, regular_grid_spacing ------------------------------------

from charts_mvdr.beamforming import (beam_response, grating_lobes,  # noqa: E402
                                     regular_grid_spacing)


def _grid(nx=8, ny=8, d=0.6):
    k = np.arange(nx * ny)
    return (k % nx) * d, (k // nx) * d


def test_regular_grid_spacing():
    x, y = _grid(8, 4, 0.6)
    assert regular_grid_spacing(x, y) == pytest.approx((0.6, 0.6))
    x2 = x.copy(); x2[5] += 0.07
    assert regular_grid_spacing(x2, y) is None
    assert regular_grid_spacing(np.zeros(4), np.arange(4.0)) is None


def test_beam_response_dft_look_and_grating_lobe():
    x, y = _grid()
    f = 399e6
    d = steering_vectors(0.4677, 0.4801, f, x, y)
    w = dft_weights(d)[0]
    assert beam_response(w, 0.4677, 0.4801, f, x, y)[0] == pytest.approx(1.0)
    lobes = grating_lobes(0.4677, 0.4801, f, 0.6, 0.6)
    assert lobes
    for _, _, l, m in lobes:
        assert beam_response(w, l, m, f, x, y)[0] == pytest.approx(1.0, abs=1e-9)


def test_beam_response_mvdr_null_and_unit_gain():
    x, y = _grid()
    f = 350e6
    d = steering_vectors(0.2, 0.1, f, x, y)
    a = steering_vectors(-0.3, 0.4, f, x, y)
    R = np.eye(64) + 0.1 * np.outer(d[0], d[0].conj()) + 100.0 * np.outer(a[0], a[0].conj())
    w = mvdr_weights(R, d)[0]
    assert beam_response(w, 0.2, 0.1, f, x, y)[0] == pytest.approx(1.0)
    assert beam_response(w, -0.3, 0.4, f, x, y)[0] < 1e-4
    # vectorized over directions
    assert beam_response(w, [0.2, -0.3], [0.1, 0.4], f, x, y).shape == (2,)
