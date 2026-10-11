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


def mvdr_weights(R: np.ndarray, A: np.ndarray, loading: float = 0.0) -> np.ndarray:
    """MVDR weights w = R^-1 a / (a^H R^-1 a) for each steering vector in A.

    loading: diagonal loading as a fraction of the mean eigenvalue, trace(R)/N.
    Returns an array with the same shape as A (n_dir, n_ant).
    """
    A = np.atleast_2d(A)
    N = R.shape[0]
    Rl = R + loading * (np.real(np.trace(R)) / N) * np.eye(N)
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