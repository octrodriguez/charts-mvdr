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


def check_noise_correlation(R: np.ndarray, num_samples: int, n_sources: int | None = None,
                            threshold_factor: float = 3.0, warn: float = 1.5, fail: float = 3.0,
                            worst_warn: float = 4.5, worst_fail: float = 6.0,
                            dead_fraction: float = 0.1, channel=None) -> CheckResult:
    """#4 Noise uncorrelated between antennas (residual after removing sources).

    Removes the excess of each eigenvalue above the floor and normalizes the
    residual to correlation coefficients. Two statistics, both ~O(1) for
    independent noise: sqrt(K) * rms|rho_ij| (global) and sqrt(K) * max|rho_ij|
    (localized coupling such as crosstalk). Dead antennas are excluded.
    """
    p = np.real(np.diag(R))
    good = np.flatnonzero(p >= dead_fraction * np.median(p))
    Rg = R[np.ix_(good, good)]
    ev, U = np.linalg.eigh(Rg)
    ev, U = ev[::-1], U[:, ::-1]
    floor = float(np.median(ev))
    n = int(np.sum(ev > threshold_factor * floor)) if n_sources is None else int(n_sources)
    R_res = Rg - (U[:, :n] * (ev[:n] - floor)) @ U[:, :n].conj().T
    d = np.sqrt(np.maximum(np.real(np.diag(R_res)), 1e-12))
    rho = np.abs(R_res / np.outer(d, d))
    np.fill_diagonal(rho, 0.0)
    m = len(good)
    rms = float(np.sqrt(np.sum(rho ** 2) / (m * (m - 1)) * num_samples))
    i, j = np.unravel_index(np.argmax(rho), rho.shape)
    worst = float(rho[i, j] * np.sqrt(num_samples))
    status = max(_grade(rms, warn, fail), _grade(worst, worst_warn, worst_fail),
                 key=[Status.PASS, Status.WARN, Status.FAIL].index)
    name = "noise_corr" if channel is None else f"noise_corr[{channel}]"
    return CheckResult(
        name=name, status=status, value=rms,
        threshold=f"rms < {warn} (WARN < {fail}); worst pair < {worst_warn} (WARN < {worst_fail})",
        note=f"{n} source(s) removed; {len(p) - m} dead antenna(s) excluded; "
             f"worst pair ({good[i]},{good[j]}): sqrt(K)*|rho| = {worst:.2f}",
        details={"n_removed": n, "worst_pair": (int(good[i]), int(good[j])), "worst": worst},
    )


def check_antenna_health(R: np.ndarray, low: float = 0.5, high: float = 2.0,
                         dead_fraction: float = 0.1, channel=None) -> CheckResult:
    """#5 Dead or outlier antennas from the autocorrelations (ratio to the median)."""
    p = np.real(np.diag(R))
    ratio = p / np.median(p)
    dead = np.flatnonzero(ratio < dead_fraction)
    alive = np.setdiff1d(np.arange(len(p)), dead)
    outliers = alive[(ratio[alive] < low) | (ratio[alive] > high)]
    if dead.size:
        status = Status.FAIL
    elif outliers.size:
        status = Status.WARN
    else:
        status = Status.PASS
    name = "antenna_health" if channel is None else f"antenna_health[{channel}]"
    return CheckResult(
        name=name, status=status, value=int(outliers.size + dead.size),
        threshold=f"power/median in [{low}, {high}]; dead < {dead_fraction}",
        note=f"power/median range (alive) = [{ratio[alive].min():.2f}, {ratio[alive].max():.2f}]; "
             f"outliers: {outliers.tolist()}; dead: {dead.tolist()}",
        details={"ratio": ratio, "outliers": outliers, "dead": dead},
    )


def format_results(results: list[CheckResult]) -> str:
    """Plain-text table of check results."""
    w = max([16] + [len(r.name) for r in results])
    lines = [f"{'check':<{w}} {'status':<6} {'value':>10}  threshold / note"]
    for r in results:
        val = "n/a" if r.value is None else f"{r.value:.4g}"
        lines.append(f"{r.name:<{w}} {r.status.value:<6} {val:>10}  {r.threshold}")
        if r.note:
            lines.append(" " * (w + 20) + r.note)
    return "\n".join(lines)