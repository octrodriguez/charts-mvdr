"""Steering vectors and beamformer weights (DFT and MVDR).

Sign convention (matches the Kotekan test_charts generator and the NumPy
correlator, R_ij = sum_t v_i v_j^*): a plane wave from direction cosines
(l, m) arrives at antenna n with voltage proportional to

    a_n(l, m) = exp(-j 2 pi f (l x_n + m y_n) / c).

Verify this convention on real data before using it there.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


def steering_vectors(l, m, freq_hz: float, pos_x, pos_y) -> np.ndarray:
    """Steering vectors for one or many directions; shape (n_dir, n_ant)."""
    l = np.atleast_1d(np.asarray(l, dtype=float))
    m = np.atleast_1d(np.asarray(m, dtype=float))
    x = np.asarray(pos_x, dtype=float)
    y = np.asarray(pos_y, dtype=float)
    delay = (l[:, None] * x[None, :] + m[:, None] * y[None, :]) / C_LIGHT
    return np.exp(-2j * np.pi * freq_hz * delay)


def grating_lobes(l0: float, m0: float, freq_hz: float, dx: float, dy: float,
                  max_order: int = 3) -> list:
    """Visible grating lobes of a beam steered to (l0, m0) on a rectangular grid.

    On a grid with spacings dx, dy the steering vector is periodic in l with
    period c / (f dx) and in m with period c / (f dy), so directions
    (l0 + p c / (f dx), m0 + q c / (f dy)) with integers (p, q) != (0, 0) have
    exactly the same steering vector as (l0, m0). An interferer there cannot be
    nulled without nulling the look direction. Only lobes inside the visible
    region l^2 + m^2 < 1 are returned, as (p, q, l, m) tuples ordered by (p, q).
    """
    pl, pm = C_LIGHT / (freq_hz * dx), C_LIGHT / (freq_hz * dy)
    lobes = []
    for p in range(-max_order, max_order + 1):
        for q in range(-max_order, max_order + 1):
            if p == 0 and q == 0:
                continue
            l, m = l0 + p * pl, m0 + q * pm
            if l * l + m * m < 1.0:
                lobes.append((p, q, float(l), float(m)))
    return lobes


def dft_weights(A: np.ndarray) -> np.ndarray:
    """DFT (conventional) weights w = a / N for each steering vector in A (n_dir, n_ant)."""
    A = np.atleast_2d(A)
    return A / A.shape[1]


LOADING_REFS = ("trace", "noise", "absolute")


def noise_floor(R: np.ndarray) -> float:
    """Noise power per element estimated as the median eigenvalue of R.

    Robust as long as the number of sources is well below N/2 (checks #3, #7):
    source eigenvalues sit above the median and do not move it.
    """
    return float(np.median(np.linalg.eigvalsh(R)))


def mvdr_weights(R: np.ndarray, A: np.ndarray, loading: float = 0.0,
                 loading_ref: str = "trace") -> np.ndarray:
    """MVDR weights w = (R + delta I)^-1 a / (a^H (R + delta I)^-1 a) per steering vector.

    The loading delta is ``loading`` times a reference set by ``loading_ref``:
      "trace"     mean eigenvalue trace(R)/N (default, backward compatible). It
                  grows with interference power, so a fixed fraction does not
                  give a fixed loading relative to the noise.
      "noise"     noise floor (median eigenvalue, see noise_floor). Recommended:
                  robust loading needs sigma_n^2 << delta << N * INR * sigma_n^2.
                  In the 03_validate_mvdr sweep (N = 64, INR = 20 dB, K = 1536),
                  10-1000 sigma_n^2 stayed within 0.5 dB of the optimum for
                  steering phase errors up to 5 deg rms (30-1000 for 15 deg).
      "absolute"  delta = loading, in the units of R.
    Returns an array with the same shape as A (n_dir, n_ant).
    """
    A = np.atleast_2d(A)
    N = R.shape[0]
    if loading_ref == "trace":
        delta = loading * np.real(np.trace(R)) / N
    elif loading_ref == "noise":
        delta = loading * noise_floor(R) if loading else 0.0
    elif loading_ref == "absolute":
        delta = loading
    else:
        raise ValueError(f"loading_ref must be one of {LOADING_REFS}, got {loading_ref!r}")
    Rl = R + delta * np.eye(N)
    X = np.linalg.solve(Rl, A.T)                          # (n_ant, n_dir) = R^-1 a
    denom = np.einsum("in,in->n", A.T.conj(), X)          # a^H R^-1 a
    return (X / denom).T


def beam_power(R: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Output power w^H R w for each weight vector in W (n_dir, n_ant)."""
    W = np.atleast_2d(W)
    return np.real(np.einsum("ni,ij,nj->n", W.conj(), R, W))


def lm_grid(n: int = 201, extent: float = 1.0):
    """Square (l, m) grid restricted to the visible region l^2 + m^2 < 1.

    Returns flattened l, m of the visible points and the boolean mask that maps
    them back onto the n x n grid.
    """
    ax = np.linspace(-extent, extent, n)
    L, M = np.meshgrid(ax, ax)
    mask = L ** 2 + M ** 2 < 1.0
    return L[mask], M[mask], mask


def refine_peak(fn, l0: float, m0: float, step: float, n: int = 7, n_iter: int = 5):
    """Maximize fn(l, m) near (l0, m0) with successively finer local grids.

    fn takes arrays of l and m and returns one value per direction. Each
    iteration evaluates an n x n grid spanning +-step around the current best
    point (visible region only) and shrinks step to the grid spacing, so the
    final resolution is step * (2 / (n - 1)) ** n_iter. Starting from a coarse
    image peak, step should be the coarse grid spacing.
    """
    l, m = float(l0), float(m0)
    for _ in range(n_iter):
        ax = np.linspace(-step, step, n)
        L, M = np.meshgrid(l + ax, m + ax)
        L, M = L.ravel(), M.ravel()
        vis = L ** 2 + M ** 2 < 1.0
        k = int(np.argmax(fn(L[vis], M[vis])))
        l, m = float(L[vis][k]), float(M[vis][k])
        step = 2.0 * step / (n - 1)
    return l, m


def fit_source_powers(R: np.ndarray, A: np.ndarray, fixed: dict | None = None):
    """Least-squares powers of sources with known steering vectors.

    Fits the off-diagonal entries of R with sum_k p_k a_k a_k^H, A of shape
    (n_src, N). The diagonal is excluded, so per-antenna noise (and any
    uncorrelated quantization distortion) does not bias the powers. Sources
    that are missing from A, move during the integration or have steering
    errors bias the fit; power_budget reports what the model leaves unexplained.

    fixed maps source indices to known powers: those terms are subtracted from
    R and only the others are fitted. Needed when steering vectors are (nearly)
    collinear, e.g. an interferer at a grating lobe of the look direction,
    where the joint fit cannot split the power between them.

    Returns (p, noise): p (n_src,) powers in the units of R, and noise (N,) the
    per-antenna remainder diag(R) - sum_k p_k |a_kn|^2.
    """
    A = np.atleast_2d(A)
    N = R.shape[0]
    fixed = dict(fixed or {})
    p = np.zeros(A.shape[0])
    Rf = np.array(R, dtype=complex)
    for k, pk in fixed.items():
        p[k] = pk
        Rf = Rf - pk * np.outer(A[k], A[k].conj())
    free = [k for k in range(A.shape[0]) if k not in fixed]
    if free:
        off = ~np.eye(N, dtype=bool)
        T = np.stack([np.outer(A[k], A[k].conj())[off] for k in free], axis=1)
        T_ri = np.concatenate([T.real, T.imag])
        b_ri = np.concatenate([Rf[off].real, Rf[off].imag])
        p[free] = np.linalg.lstsq(T_ri, b_ri, rcond=None)[0]
    noise = np.real(np.diag(R)) - (np.abs(A) ** 2 * p[:, None]).sum(axis=0)
    return p, noise


def power_budget(R: np.ndarray, w: np.ndarray, A: np.ndarray, p: np.ndarray,
                 noise: np.ndarray) -> dict:
    """Split the output power w^H R w of one beamformer with a source model.

    A (n_src, N) and p (n_src,) are the modelled sources and noise (N,) the
    per-antenna noise, e.g. from fit_source_powers. Returns
      total     w^H R w
      noise     sum_n |w_n|^2 noise_n (white-noise term, uses the measured
                per-antenna noise rather than a single floor)
      source    p_k |w^H a_k|^2 for each source (n_src,)
      residual  total - noise - sum(source): power the model does not explain
      wng       N ||w||^2, white-noise gain relative to the DFT (1 for w = a / N)
    """
    A = np.atleast_2d(A)
    total = float(np.real(np.vdot(w, R @ w)))
    noise_term = float(np.sum(np.abs(w) ** 2 * noise))
    source = np.asarray(p) * np.abs(A.conj() @ w) ** 2
    return {"total": total, "noise": noise_term, "source": source,
            "residual": total - noise_term - float(source.sum()),
            "wng": float(w.size * np.sum(np.abs(w) ** 2))}

