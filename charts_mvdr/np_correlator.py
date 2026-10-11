"""NumPy reference correlator (CPU)."""
import numpy as np


def correlate(V: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Covariance per channel: R[f, i, j] = sum_t V[t, f, i] * conj(V[t, f, j]).

    V has shape (time, freq, antenna). Kotekan returns the sum; with
    normalize=True the result is divided by the number of samples.
    """
    Vf = np.ascontiguousarray(V.transpose(1, 2, 0))       # (freq, antenna, time)
    R = Vf @ Vf.conj().transpose(0, 2, 1)                 # batched BLAS matmul
    return R / V.shape[0] if normalize else R