"""NumPy reference correlator (CPU)."""
import numpy as np


def correlate(V: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Covariance per channel: R[f, i, j] = sum_t V[t, f, i] * conj(V[t, f, j]).

    V has shape (time, freq, antenna). Kotekan returns the sum; with
    normalize=True the result is divided by the number of samples.
    """
    R = np.einsum("tfi,tfj->fij", V, V.conj())
    return R / V.shape[0] if normalize else R