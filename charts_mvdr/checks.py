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


def format_results(results: list[CheckResult]) -> str:
    """Plain-text table of check results."""
    lines = [f"{'check':<16} {'status':<6} {'value':>10}  threshold / note"]
    for r in results:
        val = "n/a" if r.value is None else f"{r.value:.4g}"
        lines.append(f"{r.name:<16} {r.status.value:<6} {val:>10}  {r.threshold}")
        if r.note:
            lines.append(f"{'':<35}{r.note}")
    return "\n".join(lines)