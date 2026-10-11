"""Unit tests for charts_mvdr.sim_cpu (CPU only, < 30 s)."""

import numpy as np
import pytest

from charts_mvdr import sim_cpu as sc
from charts_mvdr.beamforming import mvdr_weights
from charts_mvdr.checks import check_saturation
from charts_mvdr.np_correlator import correlate

F0 = 344.1e6
SUN = dict(l=0.47, m=0.48)
RFI = dict(l=-0.830, m=-0.482)


def mvdr(R, a):
    return mvdr_weights(R, a)[0]


def rhat(scn, K, rng, **kw):
    """Normalized sample covariance of one channel."""
    return correlate(sc.simulate_voltages(scn, K, rng, **kw), normalize=True)[0]


def scenario(model="gaussian", sun_p=1.0, rfi_p=9.0):
    return sc.Scenario(
        freqs_hz=[F0],
        sources=[
            sc.Source(**SUN, power=sun_p, model=model, name="sun"),
            sc.Source(**RFI, power=rfi_p, model=model, phase=0.7, name="rfi"),
        ],
        noise_sigma=1.0,
    )


def test_layout_and_correlator_convention():
    """(time, freq, antenna) layout; correlate gives R_ij = <x_i x_j^*>."""
    scn = sc.Scenario(freqs_hz=[300e6, 400e6], sources=[sc.Source(**RFI, power=4.0)])
    x = sc.simulate_voltages(scn, 512, rng=0)
    assert x.shape == (512, 2, 64)
    ref = np.einsum("tfi,tfj->fij", x.astype(complex), x.conj().astype(complex)) / 512
    assert np.allclose(correlate(x, normalize=True), ref, rtol=1e-4, atol=1e-4)


def test_steering_unit_modulus_and_grating_lobe():
    pos = sc.grid_positions()
    a = sc.steer(F0, RFI["l"], RFI["m"], pos)
    assert np.allclose(np.abs(a), 1.0)
    lobes = sc.grating_lobes(F0, RFI["l"], RFI["m"], 0.6)
    assert lobes, "expected a visible alias for an interferer near the horizon"
    for la, ma in lobes:
        b = sc.steer(F0, la, ma, pos)
        # identical up to a global phase (positions are centred, not on the lattice origin)
        assert np.isclose(abs(np.vdot(a, b)), len(a))


def test_sample_converges_to_analytic():
    scn = scenario()
    R = sc.analytic_covariance(scn)[0]
    K = 20000
    err = np.linalg.norm(rhat(scn, K, 1) - R) / np.linalg.norm(R)
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
    w = mvdr(R, sc.steer(F0, SUN["l"], SUN["m"], scn.positions))
    # Noise-free output power. Expected ~1 (= p_sun) if the Sun survives;
    # with coherent sources the cross term w^H a_rfi cancels it.
    p_out = np.real(np.vdot(w, (R - sc.noise_covariance(scn, 0)) @ w))
    assert bool(p_out > 0.9) == should_keep


def test_mvdr_true_rin_is_optimal():
    scn = scenario()
    R_in = sc.analytic_covariance(scn, exclude=["sun"])[0]
    a_s = sc.steer(F0, SUN["l"], SUN["m"], scn.positions)
    assert np.isclose(sc.output_sinr(mvdr(R_in, a_s), a_s, 1.0, R_in),
                      sc.optimal_sinr(a_s, 1.0, R_in))


def test_rmb_mean_loss():
    scn = sc.Scenario(freqs_hz=[F0], sources=[sc.Source(**RFI, power=9.0, name="rfi")])
    R_in = sc.analytic_covariance(scn)[0]
    a_s = sc.steer(F0, SUN["l"], SUN["m"], scn.positions)
    opt = sc.optimal_sinr(a_s, 1.0, R_in)
    K, trials = 256, 200
    rho = [sc.output_sinr(mvdr(rhat(scn, K, t), a_s), a_s, 1.0, R_in) / opt
           for t in range(trials)]
    assert abs(np.mean(rho) - sc.rmb_expected_loss(64, K)) < 0.02


@pytest.mark.parametrize("clip,lo", [("symmetric", -7), ("asymmetric", -8)])
def test_quantize_range(clip, lo):
    x = np.linspace(-20, 20, 401) * (1 + 1j)
    q = sc.quantize_4bit(x, clip)
    assert q.real.min() == lo and q.real.max() == 7
    assert q.imag.min() == lo and q.imag.max() == 7


def test_quantized_frame_feeds_checks():
    scn = scenario(rfi_p=200.0)
    x = sc.simulate_voltages(scn, 1536, rng=0, quantize="asymmetric")
    res = check_saturation(x[:, 0, :])
    assert res.value > 0 and res.details["n_minus8"] > 0