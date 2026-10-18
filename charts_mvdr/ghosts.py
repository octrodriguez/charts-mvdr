"""Intermodulation ghosts of component-wise 4-bit requantization.

A strong coherent emitter at direction (l_i, m_i) gives antenna n the phase
theta_n = theta_0 + phi_n, with phi_n = -2 pi f (l_i x_n + m_i y_n) / c. The
quantizer acts separately on the real and imaginary parts, so its output is a
Fourier series in theta, c_1 e^{j theta} - c_3 e^{-j3 theta} + c_5 e^{j5 theta}
- ..., and each term is a plane wave at k (l_i, m_i), k = 1, -3, 5, -7, ...
(odd k alternating in sign, because the quantizer is odd and acts on cos and
sin). In the correlations these are terms ``B_k exp(j k (phi_i - phi_j))``.
On a regular grid with spacings (dx, dy) a plane wave at k (l_i, m_i) is
indistinguishable from any shift by multiples of (c / (f dx), c / (f dy)), so a
ghost can land in the visible sky, and in a search beam, far from the emitter.
A spatial filter cannot remove a ghost that falls inside the look beam.

Gaussian model (``harmonic_amplitudes``). Let the input of each antenna be the
emitter, a circular Gaussian waveform common to all antennas, plus independent
circular Gaussian noise; ``sigma_in`` is the total standard deviation per real
component and ``r = P_rfi / (P_rfi + P_noise)`` the normalized correlation
magnitude between antennas. For a memoryless odd quantizer Q the output
cross-correlation of two antennas (i != j) is, by the Hermite (Price)
expansion,

    E[y_i y_j^*] = 2 sum_{n odd} (c_n^2 / n!) r^n [cos^n(phi) + j sin^n(phi)],

with phi = phi_i - phi_j and c_n = E[Q(x) He_n(x / sigma_in)]. Expanding
cos^n + j sin^n in e^{jk phi} gives the amplitude B_k of every harmonic. The
leading terms are B_1 ~= g^2 * 2 sigma_in^2 r (Bussgang) and, from n = 3,
B_{-3} = c_3^2 r^3 / 12. The beam power of a plane-wave term is proportional to
its amplitude, so |B_k / B_1| is the ghost power relative to the fundamental
as seen by a beam pointed at each of them.

By Stein's lemma c_n = sigma^n E[Q^(n)(x)], which has a closed form for both
quantizer models:

* ``"round"``: Q(x) = clip(round(x), -7, 7), unit steps at the thresholds
  tau = +-0.5, ..., +-6.5, so c_n = sum_tau He_{n-1}(tau / sigma) phi(tau / sigma).
* ``"clip"``: continuous clipper at L (default 7.5, the Bussgang model of
  ``quantization``), c_1 = sigma g and, for odd n >= 3,
  c_n = -2 sigma He_{n-2}(L / sigma) phi(L / sigma).

Here phi(u) is the standard normal density. ``"round"`` is exact for the
F-engine quantizer. Its Bussgang gain is c_1 / sigma = sum_tau phi(tau / sigma)
/ sigma, a midpoint sum of the density over [-7, 7], i.e. erf(7 / (sqrt(2)
sigma)) to within 0.1 % for sigma >= 1: the equivalent clipper has L = 7, not
7.5. Leading order for ``"clip"``
(``ghost_ratio_leading``):

    P_{-3} / P_{+1} ~= L^2 exp(-L^2 / sigma^2) r^2 / (12 pi sigma^2 g^2).

A constant-envelope carrier (CW tone) is not Gaussian and clips harder; its
ghosts are larger (up to c_3 / c_1 = 1/3 in amplitude for hard clipping
without noise) and must be simulated (``sim_cpu`` with ``model="tone"``).
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb, erf

from charts_mvdr.beamforming import C_LIGHT, steering_vectors
from charts_mvdr.quantization import L_SAT, RAIL

QUANTIZERS = ("round", "clip")


# ---------------------------------------------------------------------------
# Gaussian model: harmonic amplitudes
# ---------------------------------------------------------------------------

def _hermite_normalized(u: np.ndarray, n_max: int) -> np.ndarray:
    """h_n(u) = He_n(u) / sqrt(n!) for n = 0..n_max, shape (n_max + 1, *u.shape)."""
    u = np.asarray(u, dtype=float)
    h = np.empty((n_max + 1,) + u.shape)
    h[0] = 1.0
    if n_max >= 1:
        h[1] = u
    for n in range(1, n_max):
        h[n + 1] = (u * h[n] - np.sqrt(n) * h[n - 1]) / np.sqrt(n + 1)
    return h


def _std_normal_pdf(u):
    return np.exp(-0.5 * np.asarray(u, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


def hermite_weights(sigma_in: float, n_max: int = 201, quantizer: str = "round",
                    L: float = L_SAT) -> np.ndarray:
    """a_n = c_n^2 / n! for n = 0..n_max (zero for even n), in LSB^2.

    c_n = E[Q(x) He_n(x / sigma_in)] for x ~ N(0, sigma_in^2), see module docstring.
    ``L`` is the saturation level of the ``"clip"`` model (ignored for ``"round"``).
    """
    if quantizer not in QUANTIZERS:
        raise ValueError(f"quantizer must be one of {QUANTIZERS}, got {quantizer!r}")
    s = float(sigma_in)
    if s <= 0:
        raise ValueError("sigma_in must be positive")
    a = np.zeros(n_max + 1)
    n = np.arange(n_max + 1)
    odd = (n % 2 == 1)
    if quantizer == "round":
        tau = np.concatenate([-(np.arange(RAIL) + 0.5), np.arange(RAIL) + 0.5]) / s
        # c_n / sqrt((n-1)!) = sum_tau h_{n-1}(tau) phi(tau)  ->  a_n = (.)^2 / n
        h = _hermite_normalized(tau, n_max - 1)
        q = h @ _std_normal_pdf(tau)                      # index n-1
        a[1:] = q ** 2 / n[1:]
    else:
        u = L / s
        g = erf(u / np.sqrt(2.0))
        a[1] = (s * g) ** 2
        # c_n = -2 s He_{n-2}(u) phi(u)  ->  a_n = 4 s^2 h_{n-2}(u)^2 phi(u)^2 / (n (n-1))
        h = _hermite_normalized(np.array(u), n_max - 2)
        nn = n[3:]
        a[3:] = 4.0 * s**2 * h[nn - 2] ** 2 * _std_normal_pdf(u) ** 2 / (nn * (nn - 1))
    a[~odd] = 0.0
    return a


def harmonic_amplitudes(sigma_in: float, r: float, ks=(1, -3, 5, -7),
                        quantizer: str = "round", n_max: int = 201,
                        n_phi: int = 512, L: float = L_SAT) -> np.ndarray:
    """Amplitudes B_k (LSB^2, real) of the harmonics exp(j k phi) in the output
    cross-correlation of two antennas, Gaussian emitter plus white noise.

    ``sigma_in``: total input std per real component (LSB); ``r``: normalized
    correlation magnitude P_rfi / (P_rfi + P_noise), 0 <= r < 1. The series
    converges as r^n; accuracy degrades as r -> 1 (raise ``n_max``).
    """
    r = float(r)
    if not 0.0 <= r < 1.0:
        raise ValueError("r must satisfy 0 <= r < 1")
    a = hermite_weights(sigma_in, n_max, quantizer, L)
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    c, s = np.cos(phi), np.sin(phi)
    corr = np.zeros(n_phi, dtype=complex)
    for n in range(1, n_max + 1, 2):
        if a[n] == 0.0:
            continue
        corr += 2.0 * a[n] * r**n * (c**n + 1j * s**n)
    spec = np.fft.fft(corr) / n_phi                       # spec[k] multiplies e^{jk phi}
    ks = np.atleast_1d(np.asarray(ks, dtype=int))
    if np.any(np.abs(ks) >= n_phi // 2):
        raise ValueError("|k| must be below n_phi / 2")
    return spec[ks % n_phi].real


def ghost_ratio(sigma_in: float, r: float, k: int = -3, quantizer: str = "round",
                n_max: int = 201, L: float = L_SAT) -> float:
    """Ghost power at harmonic k relative to the fundamental, |B_k / B_1|."""
    b = harmonic_amplitudes(sigma_in, r, (1, k), quantizer, n_max, L=L)
    return float(abs(b[1] / b[0]))


def ghost_ratio_leading(sigma_in, r, L: float = L_SAT):
    """Leading-order |B_{-3} / B_1| for the continuous clipper (n = 3 term only)."""
    s = np.asarray(sigma_in, dtype=float)
    g = erf(L / (np.sqrt(2.0) * s))
    return L**2 * np.exp(-L**2 / s**2) * np.asarray(r, dtype=float) ** 2 / (12.0 * np.pi * s**2 * g**2)


def sky_gain2(sigma_in: float, r: float, quantizer: str = "round", n_max: int = 201,
              L: float = L_SAT) -> float:
    """Flux-scale gain of a weak sky source with a strong coherent emitter present.

    Bussgang's g^2 = E[Q']^2 assumes independent clipping at the two antennas.
    With a common emitter the clipping events are correlated and, by Price's
    theorem, the weak source passes with E[Q'(x_i) Q'(x_j)], larger than g^2.
    Averaging over the relative phases of emitter and source (a beam far from
    the emitter and its ghosts),

        G_sky = (1 / sigma^2) sum_{n even} (c_{n+1}^2 / n!) r^n binom(n, n/2) / 2^n,

    whose n = 0 term is g^2. Off-diagonal correlations only; a beam that also
    contains a ghost of the emitter sees a lower value (mixing products).
    """
    a = hermite_weights(sigma_in, n_max, quantizer, L)
    n = np.arange(0, n_max, 2)
    return float(np.sum((n + 1) * a[n + 1] * float(r) ** n * comb(n, n // 2) / 2.0**n)
                 / float(sigma_in) ** 2)


# ---------------------------------------------------------------------------
# Geometry: where the ghosts land and which beams they hit
# ---------------------------------------------------------------------------

def ghost_directions(l: float, m: float, freq_hz: float, dx: float, dy: float | None = None,
                     ks=(-3, 5)) -> list:
    """Visible directions of the ghosts of an emitter at (l, m) on a regular grid.

    The ghost of order k is a plane wave at k (l, m); on a grid with spacings
    (dx, dy) it is the same as k (l, m) + (p c / (f dx), q c / (f dy)) for any
    integers (p, q). Returns (k, p, q, l_k, m_k) tuples with l_k^2 + m_k^2 < 1,
    ordered by k as given and then by (p, q). Empty for a k whose aliases are
    all outside the visible region.
    """
    dy = dx if dy is None else dy
    pl, pm = C_LIGHT / (freq_hz * dx), C_LIGHT / (freq_hz * dy)
    out = []
    for k in ks:
        l0, m0 = k * l, k * m
        p_range = range(int(np.ceil((-1.0 - l0) / pl)), int(np.floor((1.0 - l0) / pl)) + 1)
        q_range = range(int(np.ceil((-1.0 - m0) / pm)), int(np.floor((1.0 - m0) / pm)) + 1)
        for p in p_range:
            for q in q_range:
                lk, mk = l0 + p * pl, m0 + q * pm
                if lk * lk + mk * mk < 1.0:
                    out.append((int(k), int(p), int(q), float(lk), float(mk)))
    return out


def ghost_beam_response(l: float, m: float, freq_hz: float, looks, pos_x, pos_y,
                        ks=(-3, 5)) -> np.ndarray:
    """DFT beam response |a_look^H a_k|^2 / N^2 of each look to each ghost.

    The ghost steering vector is a(l, m)^k elementwise, i.e. the steering vector
    of k (l, m), so this works for any layout (no grid needed; on a grid the
    aliases are implied). ``looks``: (n_look, 2) array of (l, m). Returns shape
    (len(ks), n_look), in linear power (1 = ghost at the centre of the beam).
    """
    looks = np.atleast_2d(np.asarray(looks, dtype=float))
    A = steering_vectors(looks[:, 0], looks[:, 1], freq_hz, pos_x, pos_y)
    n = A.shape[1]
    out = np.empty((len(ks), looks.shape[0]))
    for i, k in enumerate(ks):
        ak = steering_vectors(k * l, k * m, freq_hz, pos_x, pos_y)[0]
        out[i] = np.abs(A.conj() @ ak) ** 2 / n**2
    return out


def ghost_beam_hits(l: float, m: float, freq_hz: float, looks, pos_x, pos_y,
                    ks=(-3, 5), threshold_db: float = -3.0):
    """Looks that receive a ghost above ``threshold_db`` (DFT response).

    Returns a list of (k, look_index, response_db) for every pair above the
    threshold, ordered by k and then by look index.
    """
    resp = ghost_beam_response(l, m, freq_hz, looks, pos_x, pos_y, ks)
    resp_db = 10.0 * np.log10(np.maximum(resp, 1e-30))
    hits = []
    for i, k in enumerate(ks):
        for j in np.flatnonzero(resp_db[i] >= threshold_db):
            hits.append((int(k), int(j), float(resp_db[i, j])))
    return hits
