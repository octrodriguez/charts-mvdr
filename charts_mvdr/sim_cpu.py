"""CPU-tier simulation for MVDR validation.

Provides two levels of simulation that produce the same covariance cube
``R[f, i, j]`` as the Kotekan tier:

* ``analytic_covariance``: exact ensemble covariance (no estimation noise,
  no quantization). Used to test beamformer algebra in isolation.
* ``simulate_voltages``: complex baseband samples with independent complex
  Gaussian sources, optional per-antenna gains and optional 4-bit
  quantization. The output layout (time, freq, antenna) is the same as
  ``data_io.read_baseband_frame``, so simulated and generator frames go
  through the same code (``np_correlator.correlate``, ``checks``).

Conventions
-----------
* Steering vectors come from ``beamforming.steering_vectors``:
  ``a_n = exp(-2j*pi*f*(l*x_n + m*y_n)/c)``, unnormalized (``||a||^2 = N``).
* Covariance: ``R = E[x x^H]``, i.e. ``R[i, j] = E[x_i conj(x_j)]``.
* Source power ``power`` is the complex variance ``E|s|^2``. Noise is
  specified by ``noise_sigma``, the standard deviation per real component,
  so the complex noise variance is ``2 * noise_sigma**2``.
* Source models:
    - ``"gaussian"``: ``s[k] ~ CN(0, power)``, independent across sources
      and samples (physically correct for sky, FRB and most RFI).
    - ``"tone"``: constant phasor ``sqrt(power) * exp(j*phi)`` within the
      integration, reproducing the db331bb7 generator. All ``"tone"``
      sources are mutually coherent and collapse into one rank-1 term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from charts_mvdr.beamforming import C_LIGHT, steering_vectors

SourceModel = Literal["gaussian", "tone"]
ClipMode = Literal["symmetric", "asymmetric"]


# ---------------------------------------------------------------------------
# Geometry and steering
# ---------------------------------------------------------------------------

def grid_positions(nx: int = 8, ny: int = 8, spacing_m: float = 0.6) -> np.ndarray:
    """Regular grid antenna positions, shape (nx*ny, 2), centred at origin.

    Default matches the simulated CHARTS-64 layout (8 x 8, 0.6 m).
    Row-major ordering: index = iy * nx + ix.
    """
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    xy = np.stack([ix.ravel(), iy.ravel()], axis=1).astype(float) * spacing_m
    return xy - xy.mean(axis=0)


def positions_from_meta(meta: dict) -> np.ndarray:
    """Antenna positions (N, 2) from ``data_io.read_window_meta``."""
    return np.column_stack([meta["antenna_pos_x_m"], meta["antenna_pos_y_m"]]).astype(float)


def steer(freq_hz: float, l: float, m: float, positions: np.ndarray) -> np.ndarray:
    """Single steering vector, shape (N,); thin wrapper of ``steering_vectors``."""
    return steering_vectors(l, m, freq_hz, positions[:, 0], positions[:, 1])[0]


def grating_lobes(freq_hz: float, l: float, m: float, spacing_m: float,
                  visible_only: bool = True) -> list[tuple[float, float]]:
    """Directions (l, m) with the same steering vector as (l, m) on a regular grid.

    Aliases occur at shifts of ``c / (f d)`` in l and m.
    """
    step = C_LIGHT / (freq_hz * spacing_m)
    out = []
    for p in (-2, -1, 0, 1, 2):
        for q in (-2, -1, 0, 1, 2):
            if p == 0 and q == 0:
                continue
            la, ma = l + p * step, m + q * step
            if not visible_only or la**2 + ma**2 <= 1.0:
                out.append((la, ma))
    return out


# ---------------------------------------------------------------------------
# Source description
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """Point source in direction cosines (l, m).

    ``power`` may be a scalar or an array with one value per frequency.
    ``phase`` is only used by the ``"tone"`` model (radians).
    ``drift`` is the rate (dl/dt, dm/dt) in 1/s of a moving source (e.g. a LEO
    satellite): at time t the source is at (l + dl/dt t, m + dm/dt t). It is
    used by ``simulate_voltages`` when sample times are given; the analytic
    covariance uses the position at t = 0.
    """
    l: float
    m: float
    power: float | np.ndarray
    model: SourceModel = "gaussian"
    phase: float = 0.0
    name: str = ""
    drift: tuple[float, float] = (0.0, 0.0)

    @property
    def moving(self) -> bool:
        return self.drift[0] != 0.0 or self.drift[1] != 0.0

    def __post_init__(self) -> None:
        if self.l**2 + self.m**2 > 1.0:
            raise ValueError(f"Source {self.name!r} below horizon: l^2+m^2 > 1")
        if self.model not in ("gaussian", "tone"):
            raise ValueError(f"Unknown source model {self.model!r}")

    def power_at(self, ifreq: int) -> float:
        p = np.asarray(self.power, dtype=float)
        return float(p) if p.ndim == 0 else float(p[ifreq])


@dataclass
class Scenario:
    """Full simulation scenario for one window."""
    freqs_hz: np.ndarray
    sources: list[Source]
    positions: np.ndarray = field(default_factory=grid_positions)
    noise_sigma: float = 1.0
    gains: np.ndarray | None = None  # complex, shape (N,) or (F, N)

    def __post_init__(self) -> None:
        self.freqs_hz = np.atleast_1d(np.asarray(self.freqs_hz, dtype=float))

    @property
    def n_ant(self) -> int:
        return self.positions.shape[0]

    def gain_at(self, ifreq: int) -> np.ndarray:
        if self.gains is None:
            return np.ones(self.n_ant, dtype=complex)
        g = np.asarray(self.gains, dtype=complex)
        return g if g.ndim == 1 else g[ifreq]


# ---------------------------------------------------------------------------
# Analytic covariance
# ---------------------------------------------------------------------------

def _source_terms(scn: Scenario, ifreq: int, sources: Sequence[Source]) -> np.ndarray:
    """Covariance contribution of ``sources`` at one frequency, shape (N, N)."""
    f = scn.freqs_hz[ifreq]
    g = scn.gain_at(ifreq)
    n = scn.n_ant
    R = np.zeros((n, n), dtype=complex)
    a_eff = np.zeros(n, dtype=complex)  # coherent sum of tone sources
    for s in sources:
        a = g * steer(f, s.l, s.m, scn.positions)
        p = s.power_at(ifreq)
        if s.model == "gaussian":
            R += p * np.outer(a, a.conj())
        else:
            a_eff += np.sqrt(p) * np.exp(1j * s.phase) * a
    R += np.outer(a_eff, a_eff.conj())
    return R


def noise_covariance(scn: Scenario, ifreq: int) -> np.ndarray:
    """Receiver noise covariance. Gains are not applied to noise here, i.e.
    the noise is referred to the digitizer input, where it is white."""
    return 2.0 * scn.noise_sigma**2 * np.eye(scn.n_ant, dtype=complex)


def analytic_covariance(scn: Scenario, exclude: Sequence[str] = ()) -> np.ndarray:
    """Ensemble covariance cube, shape (F, N, N).

    ``exclude`` drops sources by name (e.g. the desired signal, to build the
    interference-plus-noise covariance R_in).
    """
    keep = [s for s in scn.sources if s.name not in set(exclude)]
    out = np.empty((scn.freqs_hz.size, scn.n_ant, scn.n_ant), dtype=complex)
    for i in range(scn.freqs_hz.size):
        out[i] = _source_terms(scn, i, keep) + noise_covariance(scn, i)
    return out


# ---------------------------------------------------------------------------
# Voltage-level simulation
# ---------------------------------------------------------------------------

def quantize_4bit(x: np.ndarray, clip: ClipMode = "symmetric") -> np.ndarray:
    """Round real and imaginary parts to integers and clip.

    ``"symmetric"``: [-7, 7] (docstring and C++ stage).
    ``"asymmetric"``: [-8, 7] (current Python writer, db331bb7).
    """
    lo = -7 if clip == "symmetric" else -8
    re = np.clip(np.rint(x.real), lo, 7)
    im = np.clip(np.rint(x.imag), lo, 7)
    return (re + 1j * im).astype(x.dtype)


def _cn(rng: np.random.Generator, shape, var: float, dtype) -> np.ndarray:
    s = np.sqrt(var / 2.0)
    return (rng.normal(0.0, s, shape) + 1j * rng.normal(0.0, s, shape)).astype(dtype)


def simulate_voltages(scn: Scenario, n_samples: int,
                      rng: np.random.Generator | int | None = None,
                      quantize: ClipMode | None = None,
                      dtype=np.complex64,
                      times_s: np.ndarray | None = None) -> np.ndarray:
    """Complex baseband samples, shape (time, freq, antenna) = (K, F, N).

    Same layout as ``data_io.read_baseband_frame``. Gaussian sources are
    independent per sample and per frequency; tone sources are a constant
    phasor over the K samples.

    ``times_s`` (K,) gives the time of each sample; it is required when a source
    has a nonzero ``drift`` and then sets its direction sample by sample. The
    random draws do not depend on ``times_s`` or ``drift``, so a moving and a
    static version of the same scenario share noise and source waveforms.
    """
    rng = np.random.default_rng(rng)
    F, N, K = scn.freqs_hz.size, scn.n_ant, n_samples
    if any(s.moving for s in scn.sources):
        if times_s is None:
            raise ValueError("times_s is required for sources with nonzero drift")
        times_s = np.asarray(times_s, dtype=float)
        if times_s.shape != (K,):
            raise ValueError(f"times_s must have shape ({K},), got {times_s.shape}")
    x = np.empty((K, F, N), dtype=dtype)
    for i, f in enumerate(scn.freqs_hz):
        g = scn.gain_at(i)
        xi = _cn(rng, (K, N), 2.0 * scn.noise_sigma**2, dtype)
        for s in scn.sources:
            if s.moving:     # (K, N): one steering vector per sample
                a = g[None, :] * steering_vectors(s.l + s.drift[0] * times_s,
                                                  s.m + s.drift[1] * times_s, f,
                                                  scn.positions[:, 0], scn.positions[:, 1])
            else:
                a = (g * steer(f, s.l, s.m, scn.positions))[None, :]
            a = a.astype(dtype)
            p = s.power_at(i)
            if s.model == "gaussian":
                sig = _cn(rng, (K, 1), p, dtype)
            else:
                sig = np.full((K, 1), np.sqrt(p) * np.exp(1j * s.phase), dtype=dtype)
            xi += sig * a
        x[:, i, :] = quantize_4bit(xi, quantize) if quantize else xi
    return x


# ---------------------------------------------------------------------------
# Ground-truth figures of merit
# ---------------------------------------------------------------------------

def output_sinr(w: np.ndarray, a_s: np.ndarray, p_s: float, R_in: np.ndarray) -> float:
    """SINR = p_s |w^H a_s|^2 / (w^H R_in w), with true (not estimated) R_in."""
    num = p_s * np.abs(np.vdot(w, a_s)) ** 2
    den = np.real(np.vdot(w, R_in @ w))
    return float(num / den)


def optimal_sinr(a_s: np.ndarray, p_s: float, R_in: np.ndarray) -> float:
    """SINR_opt = p_s a_s^H R_in^{-1} a_s (attained by MVDR with true R_in)."""
    return float(p_s * np.real(np.vdot(a_s, np.linalg.solve(R_in, a_s))))


def rmb_expected_loss(n_ant: int, n_samples: int) -> float:
    """Reed-Mallett-Brennan mean SINR fraction rho = (K + 2 - N) / (K + 1).

    Valid when the covariance is estimated from signal-free (R_in) samples.
    """
    return (n_samples + 2 - n_ant) / (n_samples + 1)