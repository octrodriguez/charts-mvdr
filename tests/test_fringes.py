import numpy as np
import pytest

from charts_mvdr import fringes

C = 299_792_458.0


def unit_phases(freqs_mhz, n_time=180, h_span_deg=90.0, dec_deg=-10.0, lat_deg=-33.4):
    """Model phase of unit east and north baselines for a source drifting through the meridian."""
    h = np.radians(np.linspace(-h_span_deg / 2, h_span_deg / 2, n_time))
    dec, lat = np.radians(dec_deg), np.radians(lat_deg)
    s_e = np.cos(dec) * np.sin(h)
    s_n = np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(h)
    k = -2 * np.pi * np.asarray(freqs_mhz)[:, None] * 1e6 / C
    return np.stack([k * s_e[None], k * s_n[None]], axis=1)          # (chan, 2, time)


def make_vis(phase_unit, baseline, noise=0.0, conj=False, seed=0):
    rng = np.random.default_rng(seed)
    n_chan, _, n_time = phase_unit.shape
    phase = np.einsum("k,ckt->ct", np.asarray(baseline, float), phase_unit)
    inst = rng.uniform(0, 2 * np.pi, n_chan)[:, None]                # constant instrumental phase
    v = np.exp(1j * (phase + inst))
    if conj:
        v = np.conj(v)
    v = v + noise * (rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape)) / np.sqrt(2)
    return v.T                                                        # (time, chan)


FREQS = np.arange(300.0, 410.0, 3.0)


def test_coherence_is_one_for_matching_model_and_ignores_instrumental_phase():
    pu = unit_phases(FREQS)
    vis = make_vis(pu, (3.6, 0.0))
    c = fringes.coherence(vis, pu, [(3.6, 0.0), (0.0, 0.0)])
    assert np.allclose(c[0], 1.0)
    assert np.all(c[1] < 0.3)


def test_fit_recovers_east_west_and_north_components():
    pu = unit_phases(FREQS)
    for b in [(4.2, 0.0), (-0.6, 0.0), (3.0, 0.8)]:
        vis = make_vis(pu, b, noise=0.5, seed=1)
        best, c = fringes.fit_baseline(vis, pu, extent=(6, 3))
        assert np.allclose(best, b, atol=0.03), (b, best)
        assert c > 0.8


def test_conjugated_channels_fit_minus_baseline():
    pu = unit_phases(FREQS)
    vis = make_vis(pu, (3.6, 0.0), conj=True)
    best, _ = fringes.fit_baseline(vis, pu, extent=(6, 3))
    assert np.allclose(best, (-3.6, 0.0), atol=0.03)


def test_stationary_signal_fits_zero_baseline():
    pu = unit_phases(FREQS)
    vis = make_vis(pu, (0.0, 0.0), noise=0.3, seed=2)
    best, _ = fringes.fit_baseline(vis, pu, extent=(6, 3))
    assert np.allclose(best, (0.0, 0.0), atol=0.05)


def test_classify_channels_labels_each_case():
    pu = unit_phases(FREQS)
    b = np.array([4.2, 0.0])
    n = len(FREQS)
    vis = np.empty((pu.shape[2], n), complex)
    third = n // 3
    vis[:, :third] = make_vis(pu[:third], b, seed=3)
    vis[:, third:2 * third] = make_vis(pu[third:2 * third], b, conj=True, seed=4)
    vis[:, 2 * third:] = make_vis(pu[2 * third:], (0.0, 0.0), seed=5)
    labels, c = fringes.classify_channels(vis, pu, b)
    assert c.shape == (3, n)
    assert set(labels[:third]) == {"source"}
    assert set(labels[third:2 * third]) == {"conjugate"}
    assert set(labels[2 * third:]) == {"stationary"}


def test_classify_flags_noise_and_short_baselines():
    pu = unit_phases(FREQS, h_span_deg=5.0)                           # model nearly constant
    vis = make_vis(pu, (0.6, 0.0), seed=6)
    labels, _ = fringes.classify_channels(vis, pu, (0.6, 0.0))
    assert set(labels) == {"ambiguous"}
    rng = np.random.default_rng(7)
    pu = unit_phases(FREQS)
    noise = rng.normal(size=(pu.shape[2], len(FREQS))) + 1j * rng.normal(size=(pu.shape[2], len(FREQS)))
    labels, _ = fringes.classify_channels(noise, pu, (4.2, 0.0))
    assert set(labels) == {"none"}


def test_shape_errors():
    pu = unit_phases(FREQS)
    with pytest.raises(ValueError):
        fringes.coherence(np.ones((pu.shape[2] + 1, len(FREQS))), pu, [(1.0, 0.0)])
    with pytest.raises(ValueError):
        fringes.coherence(np.ones((pu.shape[2], len(FREQS))), pu, [(1.0, 0.0, 0.0)])
    with pytest.raises(ValueError):
        fringes.fit_baseline(np.ones((pu.shape[2], len(FREQS))), pu, extent=(6,))
