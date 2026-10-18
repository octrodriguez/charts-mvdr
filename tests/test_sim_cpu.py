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
    assert res.status.value == "FAIL" and res.value < 0.8   # g^2 for sigma_in ~ 10 LSB
    assert res.details["n_minus8"] > 0


def test_static_path_unchanged_and_drift_needs_times():
    """A zero drift gives exactly the previous output; a drift without times fails."""
    scn = scenario()
    x0 = sc.simulate_voltages(scn, 256, rng=3)
    t = np.linspace(0.0, 1.0, 256)
    np.testing.assert_array_equal(sc.simulate_voltages(scn, 256, rng=3, times_s=t), x0)
    scn.sources[1].drift = (0.02, -0.01)
    with pytest.raises(ValueError):
        sc.simulate_voltages(scn, 256, rng=3)


def test_drifting_source_matches_static_source_at_same_position():
    """With all samples at time t0, a moving source equals a static one at its
    position at t0 (same random draws)."""
    t0, drift = 0.4, (0.03, -0.02)
    moving = scenario(); moving.sources[1].drift = drift
    static = scenario()
    static.sources[1].l += drift[0] * t0
    static.sources[1].m += drift[1] * t0
    xm = sc.simulate_voltages(moving, 128, rng=5, times_s=np.full(128, t0))
    xs = sc.simulate_voltages(static, 128, rng=5)
    np.testing.assert_allclose(xm, xs, atol=1e-5)


# --- signal-in vs signal-free covariance (self-nulling) ---------------------

def test_self_nulling_signal_in_vs_signal_free():
    """With a pointing error, MVDR on R (source inside) nulls part of the source;
    on the signal-free R_in it does not. With exact steering both coincide."""
    f = 344.1e6
    pos = sc.grid_positions(8, 8, 0.6)
    look, rfi = (0.47, 0.48), (-0.83, -0.482)
    p_s = 100.0 * 2.0 / 64                       # array SNR 20 dB with noise_sigma = 1
    for dl, expect_loss in ((0.0, False), (0.01, True)):
        src = (look[0] + dl, look[1])
        scn = sc.Scenario(freqs_hz=[f], positions=pos, noise_sigma=1.0,
                          sources=[sc.Source(*src, power=p_s, name="sun"),
                                   sc.Source(*rfi, power=200.0, name="rfi")])
        R = sc.analytic_covariance(scn)[0]
        R_in = sc.analytic_covariance(scn, exclude=["sun"])[0]
        a = sc.steer(f, *src, pos)
        d = sc.steer(f, *look, pos)
        s_on = sc.output_sinr(mvdr_weights(R, d)[0], a, p_s, R_in)
        s_off = sc.output_sinr(mvdr_weights(R_in, d)[0], a, p_s, R_in)
        if expect_loss:
            assert 10 * np.log10(s_off / s_on) > 10.0
        else:
            assert s_on == pytest.approx(s_off, rel=1e-6)
