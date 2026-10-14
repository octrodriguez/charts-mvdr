"""4-bit quantization and saturation model (Bussgang).

The F-engine quantizes each real component to integers in [-7, 7] by rounding,
so inputs above 6.5 LSB map to the rail value 7 and the output saturates for
|x| >= 7.5. For a zero-mean Gaussian input with standard deviation sigma per
component, a weak signal embedded in it passes with the Bussgang amplitude gain

    g = erf(L / (sqrt(2) sigma)),    L = 7.5,

so its power in the cross-correlations is scaled by g^2. sigma is estimated from
the fraction of components at the rails, f_rail = erfc(6.5 / (sqrt(2) sigma)),
when saturation is measurable, and from the output standard deviation otherwise.
Accuracy of the predicted g^2 against simulations: about 0.03.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf, erfcinv

RAIL = 7                 # largest magnitude of the symmetric 4-bit range
RAIL_START = 6.5         # inputs above this round to the rail value
L_SAT = 7.5              # effective saturation level of the rounding quantizer
MIN_RAIL_FRACTION = 1e-4 # below this, sigma is estimated from the output std


def rail_fraction(V: np.ndarray, rail: int = RAIL) -> float:
    """Fraction of real components with |x| >= rail (complex input)."""
    return float(np.mean(np.abs(np.concatenate([V.real.ravel(), V.imag.ravel()])) >= rail))


def sigma_from_rail(f_rail):
    """Input std per component from the rail fraction: f = erfc(6.5 / (sqrt(2) sigma))."""
    f = np.clip(np.asarray(f_rail, dtype=float), 1e-300, 1.0 - 1e-15)
    return RAIL_START / (np.sqrt(2.0) * erfcinv(f))


def estimate_sigma_in(f_rail, std_out):
    """Input std per component: rail-based where saturation is measurable,
    otherwise the output std minus uniform quantization noise (1/12 LSB^2)."""
    f_rail = np.asarray(f_rail, dtype=float)
    std_out = np.asarray(std_out, dtype=float)
    from_std = np.sqrt(np.maximum(std_out**2 - 1.0 / 12.0, 1e-12))
    return np.where(f_rail >= MIN_RAIL_FRACTION, sigma_from_rail(f_rail), from_std)


def bussgang_gain2(sigma_in):
    """Power gain g^2 = erf(L / (sqrt(2) sigma))^2 on a weak signal."""
    return erf(L_SAT / (np.sqrt(2.0) * np.asarray(sigma_in, dtype=float))) ** 2


def loss_db(gain2):
    """Signal power loss in dB (positive number) for a power gain g^2."""
    return -10.0 * np.log10(np.maximum(np.asarray(gain2, dtype=float), 1e-30))
