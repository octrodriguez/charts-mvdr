"""Tests for charts_mvdr.quantization and the Bussgang-based saturation check."""

import numpy as np
import pytest

from charts_mvdr import quantization as q
from charts_mvdr import sim_cpu as sc
from charts_mvdr.checks import Status, check_saturation


def gaussian_frame(sigma, K=1536, N=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, sigma, (K, N)) + 1j * rng.normal(0, sigma, (K, N))
    return sc.quantize_4bit(x.astype(np.complex64), "symmetric")


@pytest.mark.parametrize("sigma", [2.0, 3.5, 5.3, 8.0])
def test_sigma_from_rail_roundtrip(sigma):
    f = float(np.asarray(1 - q.erf(q.RAIL_START / (np.sqrt(2) * sigma))))
    assert q.sigma_from_rail(f) == pytest.approx(sigma, rel=1e-6)


@pytest.mark.parametrize("sigma", [1.0, 2.0, 4.0, 6.0])
def test_sigma_estimate_on_quantized_noise(sigma):
    V = gaussian_frame(sigma)
    comps = np.concatenate([V.real.ravel(), V.imag.ravel()])
    est = q.estimate_sigma_in(q.rail_fraction(V), np.std(comps))
    assert est == pytest.approx(sigma, rel=0.05)


def test_gain_monotonic_and_limits():
    g2 = q.bussgang_gain2(np.array([0.5, 2.0, 5.0, 20.0]))
    assert np.all(np.diff(g2) < 0)
    assert g2[0] == pytest.approx(1.0) and g2[-1] < 0.1
    assert q.loss_db(1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("sigma,expected", [(2.0, Status.PASS), (3.6, Status.WARN),
                                            (5.3, Status.FAIL)])
def test_check_saturation_grades_by_loss(sigma, expected):
    res = check_saturation(gaussian_frame(sigma))
    assert res.status == expected, res.note
