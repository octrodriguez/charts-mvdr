import numpy as np
import pytest

from charts_mvdr.beamforming import grating_lobes, steering_vectors
from charts_mvdr.ghosts import (
    ghost_beam_hits,
    ghost_beam_response,
    ghost_directions,
    ghost_ratio,
    ghost_ratio_leading,
    harmonic_amplitudes,
    hermite_weights,
)
from charts_mvdr.quantization import bussgang_gain2
from charts_mvdr.sim_cpu import Scenario, Source, grid_positions, simulate_voltages


def db(x):
    return 10.0 * np.log10(x)


# ---------------------------------------------------------------------------
# Gaussian model
# ---------------------------------------------------------------------------

def test_clip_first_weight_is_bussgang():
    s = 4.0
    a = hermite_weights(s, n_max=11, quantizer="clip")
    assert a[1] == pytest.approx(s**2 * bussgang_gain2(s), rel=1e-12)
    assert np.all(a[::2] == 0.0)


def test_round_bussgang_gain_is_clip_at_seven():
    # c_1 of the staircase is a midpoint sum of the density over [-7, 7]
    rng = np.random.default_rng(1)
    for s in (2.0, 3.0, 5.3, 8.8):
        a_r = hermite_weights(s, 3, "round")[1]
        assert a_r == pytest.approx(hermite_weights(s, 3, "clip", L=7.0)[1], rel=2e-3)
        x = rng.normal(0.0, s, 1_000_000)
        g_mc = 1.0 + np.mean(x * (np.clip(np.rint(x), -7, 7) - x)) / s**2  # control variate
        assert np.sqrt(a_r) / s == pytest.approx(g_mc, abs=3e-3)


def test_fundamental_is_bussgang_for_weak_coherence():
    s, r = 4.0, 0.05
    b1 = harmonic_amplitudes(s, r, (1,), "clip")[0]
    assert b1 / (2 * s**2 * r) == pytest.approx(bussgang_gain2(s), rel=1e-3)


def test_only_odd_alternating_harmonics():
    b = harmonic_amplitudes(4.5, 0.8, (1, -3, 5, -7, -1, 3, -5, 7, 0, 2, -2), "round")
    main, absent = np.abs(b[:4]), np.abs(b[4:])
    assert np.all(main > 0)
    assert np.all(absent < 1e-12 * main[0])


def test_leading_order_matches_series_at_low_coherence():
    # n = 5 contributes to k = -3 only at order r^5; small r isolates n = 3
    for s in (3.5, 5.0, 7.0):
        full = ghost_ratio(s, 0.3, -3, "clip")
        assert db(full) == pytest.approx(db(ghost_ratio_leading(s, 0.3)), abs=0.3)


def test_ghost_falls_steeply_with_sigma():
    r = 0.85
    levels = [ghost_ratio(s, r, -3, "round") for s in (7.0, 5.0, 4.0, 3.0)]
    assert np.all(np.diff(levels) < 0)
    assert db(levels[0]) - db(levels[-1]) > 12.0


def _fit_harmonics(V, a, ks):
    """Least-squares amplitudes of a^k a^k^H in the off-diagonal correlations."""
    K, N = V.shape
    R = V.T @ V.conj() / K
    off = ~np.eye(N, dtype=bool)
    basis = np.stack([np.outer(a**k, (a**k).conj())[off] for k in ks], axis=1)
    return np.linalg.lstsq(basis, R[off], rcond=None)[0].real


def test_series_matches_quantized_simulation():
    f = 350e6
    pos = grid_positions()
    sig_n, sig_r = 2.0, 4.0                       # sigma_in = 4.47, r = 0.8
    scn = Scenario(freqs_hz=np.array([f]), noise_sigma=sig_n, positions=pos,
                   sources=[Source(0.31, 0.17, power=2 * sig_r**2, name="rfi")])
    V = simulate_voltages(scn, 20000, rng=3, quantize="symmetric", dtype=np.complex128)[:, 0, :]
    a = steering_vectors(0.31, 0.17, f, pos[:, 0], pos[:, 1])[0]
    b = _fit_harmonics(V, a, (1, -3, 5, -7))
    s, r = np.hypot(sig_n, sig_r), sig_r**2 / (sig_n**2 + sig_r**2)
    pred = harmonic_amplitudes(s, r, (1, -3), "round")
    assert b[0] == pytest.approx(pred[0], rel=0.02)
    assert db(b[1] / b[0]) == pytest.approx(db(pred[1] / pred[0]), abs=0.7)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        hermite_weights(3.0, quantizer="floor")
    with pytest.raises(ValueError):
        hermite_weights(0.0)
    with pytest.raises(ValueError):
        harmonic_amplitudes(3.0, 1.0)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

F_CASE = 390.6e6                 # emitter near the horizon, look near zenith
EMITTER = (-0.83, -0.482)
LOOK = (-0.10, 0.25)


def test_minus3_ghost_aliases_next_to_look():
    g = [d for d in ghost_directions(*EMITTER, F_CASE, 0.6, ks=(-3,))]
    assert len(g) == 1
    k, p, q, lk, mk = g[0]
    assert (k, p, q) == (-3, -2, -1)
    assert (lk, mk) == pytest.approx((-0.068, 0.167), abs=1e-3)


def test_k1_directions_are_emitter_and_its_grating_lobes():
    l, m = 0.2, -0.1
    d = ghost_directions(l, m, 399e6, 0.6, ks=(1,))
    lobes = grating_lobes(l, m, 399e6, 0.6, 0.6)
    got = sorted((round(x[3], 9), round(x[4], 9)) for x in d)
    want = sorted([(l, m)] + [(round(x[2], 9), round(x[3], 9)) for x in lobes])
    assert got == want


def test_beam_response_and_hits():
    pos = grid_positions()
    resp = ghost_beam_response(*EMITTER, F_CASE, [LOOK, (0.5, 0.5)], pos[:, 0], pos[:, 1],
                               ks=(1, -3))
    assert db(resp[0, 0]) < -30.0                      # emitter far down the sidelobes
    assert db(resp[1, 0]) == pytest.approx(-4.8, abs=0.1)
    hits = ghost_beam_hits(*EMITTER, F_CASE, [LOOK, (0.5, 0.5)], pos[:, 0], pos[:, 1],
                           ks=(-3, 5), threshold_db=-6.0)
    assert [(k, j) for k, j, _ in hits] == [(-3, 0)]


def test_beam_response_any_layout():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3, 3, (40, 2))
    resp = ghost_beam_response(0.3, 0.1, 350e6, [(0.3, 0.1), (-0.9, -0.3)], pos[:, 0], pos[:, 1],
                               ks=(1, -3))
    assert resp[0, 0] == pytest.approx(1.0)
    assert resp[1, 1] == pytest.approx(1.0)            # -3 (0.3, 0.1) as a look direction
