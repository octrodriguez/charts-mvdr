"""Steering vectors and beamformer weights (DFT and MVDR).

Sign convention (matches the Kotekan test_charts generator and the NumPy
correlator, R_ij = sum_t v_i v_j^*): a plane wave from direction cosines
(l, m) arrives at antenna n with voltage proportional to

    a_n(l, m) = exp(-j 2 pi f (l x_n + m y_n) / c).

Verify this convention on real data before using it there.
"""
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
