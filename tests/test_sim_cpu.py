"""Unit tests for charts_mvdr.sim_cpu (CPU only, < 30 s)."""

import numpy as np
import pytest

from charts_mvdr import sim_cpu as sc

F0 = 344.1e6
SUN = dict(l=0.47, m=0.48)
RFI = dict(l=-0.830, m=-0.482)


def mvdr(R, a):
    Ria = np.linalg.solve(R, a)
    return Ria / np.vdot(a, Ria)


def scenario(model="gaussian", sun_p=1.0, rfi_p=9.0):
    return sc.Scenario(
        freqs_hz=[F0],
        sources=[
            sc.Source(**SUN, power=sun_p, model=model, name="sun"),
            sc.Source(**RFI, power=rfi_p, model=model, phase=0.7, name="rfi"),
        ],
        noise_sigma=1.0,
    )


def test_steering_unit_modulus_and_grating_lobe():
    pos = sc.grid_positions()
    a = sc.steering_vector(F0, RFI["l"], RFI["m"], pos)
    assert np.allclose(np.abs(a), 1.0)
    lobes = sc.grating_lobes(F0, RFI["l"], RFI["m"], 0.6)
    assert lobes, "expected a visible alias for an interferer near the horizon"
    for la, ma in lobes:
        b = sc.steering_vector(F0, la, ma, pos)
        # identical up to a global phase (positions are centred, not on the lattice origin)
        assert np.isclose(abs(np.vdot(a, b)), len(a))


def test_sample_converges_to_analytic():
    scn = scenario()
    R = sc.analytic_covariance(scn)
    K = 20000
    Rh = sc.sample_covariance(sc.simulate_voltages(scn, K, rng=1))
    err = np.linalg.norm(Rh - R) / np.linalg.norm(R)
    assert err < 5.0 / np.sqrt(K)


def test_tone_sources_are_rank_one():
    """Reproduces the db331bb7 coherence finding."""
    lam_t = np.linalg.eigvalsh(sc.analytic_covariance(scenario("tone"))[0])[::-1]
    lam_g = np.linalg.eigvalsh(sc.analytic_covariance(scenario("gaussian"))[0])[::-1]
    noise = 2.0
    assert np.isclose(lam_t[1], noise)          # Sun lost
    assert lam_g[1] > noise + 0.5 * 64          # Sun recovered (~N p_sun)


@pytest.mark.parametrize("model,should_keep", [("gaussian", True), ("tone", False)])
def test_mvdr_signal_cancellation(model, should_keep):
    scn = scenario(model)
    R = sc.analytic_covariance(scn)[0]
    a_s = sc.steering_vector(F0, SUN["l"], SUN["m"], scn.positions)
    w = mvdr(R, a_s)
    # Noise-free output power. Expected ~1 (= p_sun) if the Sun survives;
    # with coherent sources the cross term w^H a_rfi cancels it.
    p_out = np.real(np.vdot(w, (R - sc.noise_covariance(scn, 0)) @ w))
    assert bool(p_out > 0.9) == should_keep


def test_mvdr_true_rin_is_optimal():
    scn = scenario()
    R_in = sc.analytic_covariance(scn, exclude=["sun"])[0]
    a_s = sc.steering_vector(F0, SUN["l"], SUN["m"], scn.positions)
    w = mvdr(R_in, a_s)
    assert np.isclose(sc.output_sinr(w, a_s, 1.0, R_in), sc.optimal_sinr(a_s, 1.0, R_in))


def test_rmb_mean_loss():
    scn = sc.Scenario(freqs_hz=[F0],
                      sources=[sc.Source(**RFI, power=9.0, name="rfi")])
    R_in = sc.analytic_covariance(scn)[0]
    a_s = sc.steering_vector(F0, SUN["l"], SUN["m"], scn.positions)
    opt = sc.optimal_sinr(a_s, 1.0, R_in)
    K, trials = 256, 200
    rho = [sc.output_sinr(mvdr(sc.sample_covariance(sc.simulate_voltages(scn, K, rng=t))[0], a_s),
                          a_s, 1.0, R_in) / opt for t in range(trials)]
    assert abs(np.mean(rho) - sc.rmb_expected_loss(64, K)) < 0.02


@pytest.mark.parametrize("clip,lo", [("symmetric", -7), ("asymmetric", -8)])
def test_quantize_range(clip, lo):
    x = np.linspace(-20, 20, 401) * (1 + 1j)
    q = sc.quantize_4bit(x, clip)
    assert q.real.min() == lo and q.real.max() == 7
    assert q.imag.min() == lo and q.imag.max() == 7