"""Pre-flight checks of the assumptions required for MVDR beamforming.

Each check returns a CheckResult with a uniform format so that results can be
collected into a report by scripts/00_check_assumptions.py or reused by other
scripts before computing beamformer weights.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    NA = "N/A"


@dataclass
class CheckResult:
    name: str
    status: Status
    value: float | None
    threshold: str
    note: str = ""
    details: dict = field(default_factory=dict)


def _grade(value: float, warn: float, fail: float) -> Status:
    """Lower is better: PASS below warn, WARN below fail, FAIL otherwise."""
    if value < warn:
        return Status.PASS
    if value < fail:
        return Status.WARN
    return Status.FAIL

def _grade_high(value: float, warn: float, fail: float) -> Status:
    """Higher is better: PASS above warn, WARN above fail, FAIL otherwise."""
    if value > warn:
        return Status.PASS
    if value > fail:
        return Status.WARN
    return Status.FAIL

def check_narrowband(pos_x, pos_y, channel_width_hz: float,
                     warn: float = 0.05, fail: float = 0.2) -> CheckResult:
    """#1 Narrowband condition: B * tau_max << 1 for the longest baseline."""
    x = np.asarray(pos_x, dtype=float)
    y = np.asarray(pos_y, dtype=float)
    d_max = float(np.max(np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])))
    tau_max = d_max / C_LIGHT
    b_tau = channel_width_hz * tau_max
    return CheckResult(
        name="narrowband",
        status=_grade(b_tau, warn, fail),
        value=b_tau,
        threshold=f"B*tau < {warn} (WARN < {fail})",
        note=f"D_max = {d_max:.2f} m, tau_max = {tau_max * 1e9:.1f} ns, "
             f"coherence loss sinc^2 = {np.sinc(b_tau) ** 2:.4f}",
        details={"d_max_m": d_max, "tau_max_s": tau_max, "b_tau": b_tau},
    )


def check_sample_support(num_samples: int, num_ant: int,
                         warn_db: float = 1.0, fail_db: float = 3.0) -> CheckResult:
    """#2 Sample support (Reed-Mallett-Brennan): expected SINR loss from estimating R."""
    K, N = int(num_samples), int(num_ant)
    if K < N:
        return CheckResult(
            name="sample_support", status=Status.FAIL, value=None,
            threshold=f"K >= N, loss < {warn_db} dB",
            note=f"K = {K} < N = {N}: sample covariance is singular",
        )
    rho = (K + 2 - N) / (K + 1)
    loss_db = -10.0 * np.log10(rho)
    return CheckResult(
        name="sample_support",
        status=_grade(loss_db, warn_db, fail_db),
        value=loss_db,
        threshold=f"loss < {warn_db} dB (WARN < {fail_db} dB)",
        note=f"K = {K}, N = {N}, K/N = {K / N:.1f}, SINR retained = {rho:.3f}",
        details={"rho": rho, "K": K, "N": N},
    )


def check_source_count(R: np.ndarray, expected: int | None = None,
                       threshold_factor: float = 3.0, channel=None) -> CheckResult:
    """#3 Eigenvalues above the noise floor (one per mutually incoherent source).

    Coherent sources collapse into a single eigenvalue. Without an expected
    count the result is informational (N/A): coherence cannot be inferred.
    """
    ev = np.linalg.eigvalsh(R)[::-1]
    floor = float(np.median(ev))
    n_above = int(np.sum(ev > threshold_factor * floor))
    name = "source_count" if channel is None else f"source_count[{channel}]"
    lead = ", ".join(f"{x:.3g}" for x in ev[:max(n_above + 1, 2)])
    note = f"leading eigenvalues: {lead}; floor (median) = {floor:.3g}"
    if expected is None:
        status = Status.NA
        threshold = f"> {threshold_factor} x median (no expected count)"
    else:
        threshold = f"expected {expected}"
        if n_above == expected:
            status = Status.PASS
        elif n_above < expected:
            status = Status.FAIL
            note += "; fewer than expected: coherent or undetectable sources"
        else:
            status = Status.WARN
            note += "; more than expected: unmodeled source or non-white noise"
    return CheckResult(name=name, status=status, value=n_above, threshold=threshold,
                       note=note, details={"eigenvalues": ev, "floor": floor})


def check_conditioning(R: np.ndarray, warn: float = 0.3, fail: float = 0.05,
                       channel=None) -> CheckResult:
    """#8 Smallest eigenvalue relative to the noise floor (need for diagonal loading)."""
    ev = np.linalg.eigvalsh(R)  # ascending
    floor = float(np.median(ev))
    ratio = float(ev[0] / floor)
    name = "conditioning" if channel is None else f"conditioning[{channel}]"
    return CheckResult(
        name=name,
        status=_grade_high(ratio, warn, fail),
        value=ratio,
        threshold=f"lambda_min/median > {warn} (WARN > {fail})",
        note=f"condition number = {ev[-1] / ev[0]:.3g}; "
             f"WARN/FAIL suggests diagonal loading",
        details={"lambda_min": float(ev[0]), "lambda_max": float(ev[-1]), "floor": floor},
    )

def format_results(results: list[CheckResult]) -> str:
    """Plain-text table of check results."""
    lines = [f"{'check':<16} {'status':<6} {'value':>10}  threshold / note"]
    for r in results:
        val = "n/a" if r.value is None else f"{r.value:.4g}"
        lines.append(f"{r.name:<16} {r.status.value:<6} {val:>10}  {r.threshold}")
        if r.note:
            lines.append(f"{'':<35}{r.note}")
    return "\n".join(lines)