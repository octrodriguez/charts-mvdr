"""Pre-flight checks of the assumptions required for MVDR beamforming.

Each check returns a CheckResult with a uniform format so that results can be
collected into a report by scripts/00_check_assumptions.py or reused by other
scripts before computing beamformer weights.
"""
from dataclasses import dataclass, field
from enum import Enum
from charts_mvdr import quantization as q
from charts_mvdr.beamforming import steering_vectors
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

def check_saturation(Vc: np.ndarray, warn: float = 0.95, fail: float = 0.80,
                     channel=None) -> CheckResult:
    """#6 Linearity: expected power gain g^2 on a weak signal (Bussgang model).

    Vc holds the 4-bit complex voltages of one channel, shape (time, antenna).
    sigma_in is estimated from the fraction of components at the rails (or from
    the output std when there is no measurable saturation) and the gain is
    g^2 = erf(7.5 / (sqrt(2) sigma_in))^2, see charts_mvdr.quantization.
    Graded on g^2 (higher is better): defaults correspond to losses of 0.2 dB
    (WARN) and 1 dB (FAIL). Also reports samples at -8 (outside [-7, 7]).
    """
    comps = np.concatenate([Vc.real.ravel(), Vc.imag.ravel()])
    f_rail = float(np.mean(np.abs(comps) >= q.RAIL))
    std = float(np.std(comps))
    sigma_in = float(q.estimate_sigma_in(f_rail, std))
    g2 = float(q.bussgang_gain2(sigma_in))
    n_m8 = int(np.sum(comps <= -8))
    name = "saturation" if channel is None else f"saturation[{channel}]"
    return CheckResult(
        name=name, status=_grade_high(g2, warn, fail), value=g2,
        threshold=f"g^2 > {warn:g} (WARN > {fail:g})",
        note=f"loss = {float(q.loss_db(g2)):.2f} dB; sigma_in = {sigma_in:.2f} LSB; "
             f"std out = {std:.2f} LSB; fraction at rails = {f_rail:.2e}; samples at -8: {n_m8}",
        details={"gain2": g2, "sigma_in": sigma_in, "fraction": f_rail, "std": std,
                 "n_minus8": n_m8},
    )


def check_degrees_of_freedom(n_sources: int, num_ant: int, warn: float = 0.25,
                             fail: float = 0.5, channel=None) -> CheckResult:
    """#7 Degrees of freedom: number of sources relative to the number of antennas."""
    frac = n_sources / num_ant
    name = "dof" if channel is None else f"dof[{channel}]"
    return CheckResult(
        name=name, status=_grade(frac, warn, fail), value=frac,
        threshold=f"n_sources/N < {warn} (WARN < {fail})",
        note=f"{n_sources} source(s), N = {num_ant}; MVDR can null at most N - 1",
    )


def check_spatial_separation(look_lm, interferers_lm, freq_hz: float, pos_x, pos_y,
                             warn: float = 0.1, fail: float = 0.5, channel=None) -> CheckResult:
    """#12 Similarity between the look direction and each interferer (includes aliasing).

    cos^2 = |d^H a_i|^2 / N^2; nulling a strong interferer scales the SINR by
    about (1 - cos^2). Grating lobes appear as cos^2 -> 1.
    """
    name = "separation" if channel is None else f"separation[{channel}]"
    if len(interferers_lm) == 0:
        return CheckResult(name=name, status=Status.NA, value=None,
                           threshold="no interferer directions given")
    d = steering_vectors(*look_lm, freq_hz, pos_x, pos_y)[0]
    l_i, m_i = np.asarray(interferers_lm, dtype=float).T
    A = steering_vectors(l_i, m_i, freq_hz, pos_x, pos_y)
    cos2 = np.abs(A.conj() @ d) ** 2 / len(d) ** 2
    k = int(np.argmax(cos2))
    loss_db = -10 * np.log10(max(1.0 - cos2[k], 1e-12))
    return CheckResult(
        name=name, status=_grade(float(cos2[k]), warn, fail), value=float(cos2[k]),
        threshold=f"max cos^2 < {warn} (WARN < {fail})",
        note=f"closest interferer at ({l_i[k]:+.2f}, {m_i[k]:+.2f}); "
             f"expected SINR loss from nulling ~ {loss_db:.2f} dB",
        details={"cos2": cos2},
    )

def whitened_change(R_a: np.ndarray, R_b: np.ndarray, R_ref: np.ndarray,
                    num_samples: float) -> float:
    """K ||R_ref^(-1/2) (R_a - R_b) R_ref^(-H/2)||_F^2 / N^2.

    For circular Gaussian data and R_a an estimate of R_ref with K samples the
    expectation is 1 (whitening makes the N^2 entries of the error independent
    with variance 1/K), whatever the source powers. Quantized data give a bias of
    a few percent to ~20 %, which the checks below remove with a baseline
    measured on the same data.
    """
    L = np.linalg.cholesky(R_ref)
    X = np.linalg.solve(L, R_a - R_b)
    W = np.linalg.solve(L, X.conj().T).conj().T
    return float(num_samples * np.sum(np.abs(W) ** 2) / R_ref.shape[0] ** 2)


def check_stationarity(z_mean: np.ndarray, z_consec: np.ndarray, warn: float = 1.25,
                       fail: float = 2.0, times_s=None, channel=None) -> CheckResult:
    """#9 Stationarity over a window of M frames.

    z_mean[k]   = whitened_change(R_k, R_mean, R_mean, K) / (1 - 1/M)
    z_consec[k] = whitened_change(R_k, R_k+1, (R_k + R_k+1)/2, K/2)
    Both are ~c for a stationary scene (c = 1 for Gaussian data, slightly more
    after 4-bit quantization). The baseline c is the median of z_consec, and two
    ratios are graded (worst of both):
      step   max_k z_consec[k] / c   -> abrupt change between consecutive frames
      global median(z_mean) / c      -> scene differs across the window
                                        (events covering part of it, drifts)
    """
    z_mean, z_consec = np.asarray(z_mean, float), np.asarray(z_consec, float)
    c = float(np.median(z_consec))
    r = z_consec / c
    k = int(np.argmax(r))
    step = float(r[k])
    glob = float(np.median(z_mean) / c)
    value = max(step, glob)
    when = f" (t = {times_s[k]:.3f}-{times_s[k + 1]:.3f} s)" if times_s is not None else ""
    name = "stationarity" if channel is None else f"stationarity[{channel}]"
    return CheckResult(
        name=name, status=_grade(value, warn, fail), value=value,
        threshold=f"max(step, global) < {warn} (WARN < {fail}); ~1 if stationary",
        note=f"step = {step:.2f} between frames {k} and {k + 1}{when}; "
             f"global = {glob:.2f}; baseline c = {c:.3f}",
        details={"step": step, "global": glob, "baseline": c, "worst_pair": k,
                 "z_mean": z_mean, "z_consec": z_consec},
    )


def check_on_off(R_on: np.ndarray, K_on: float, R_off_a: np.ndarray, R_off_b: np.ndarray,
                 K_off_half: float, d: np.ndarray, warn: float = 1.25, fail: float = 2.0,
                 channel=None) -> CheckResult:
    """#10 The off-pulse covariance represents the on-pulse interference.

    R_off = (R_off_a + R_off_b) / 2 from two halves of the off window (K_off_half
    samples each). After whitening by R_off, Delta = R_on - R_off should contain
    only the desired signal alpha b b^H (b = R_off^(-1/2) d, d the look-direction
    steering vector) plus estimation noise. The residual is normalized by its
    Gaussian expectation, tr(R_off^-1 R_on)^2 / K_on + N^2 / K_off, and divided by
    the baseline from the two off halves (same statistic, pure estimation noise,
    which absorbs the quantization bias), so the ratio is ~1 when the
    interference is unchanged. A change of interferer power in the same direction
    barely moves the ratio, consistently with MVDR, whose nulls depend on the
    interference directions. alpha is the excess power per element in the look
    direction (in units of R).
    """
    N = d.size
    R_off = 0.5 * (R_off_a + R_off_b)
    L = np.linalg.cholesky(R_off)
    Dw = np.linalg.solve(L, R_on - R_off)
    Dw = np.linalg.solve(L, Dw.conj().T).conj().T
    b = np.linalg.solve(L, d)
    bb = np.real(np.vdot(b, b))
    alpha_w = float(np.real(np.vdot(b, Dw @ b)) / bb**2)
    resid = Dw - alpha_w * np.outer(b, b.conj())
    K_off = 2.0 * K_off_half
    # Expected ||residual||^2 for Gaussian estimation noise: for an estimate of R
    # with K samples whitened by R_off, E||W||^2 = tr(R_off^-1 R)^2 / K, so the
    # on window (which also contains the signal) contributes tr(R_off^-1 R_on)^2 / K_on.
    t_on = float(np.real(np.trace(np.linalg.solve(R_off, R_on))))
    expected = t_on**2 / K_on + N**2 / K_off
    z = float(np.sum(np.abs(resid) ** 2) / expected)
    base = whitened_change(R_off_a, R_off_b, R_off, K_off_half / 2.0)
    ratio = z / base
    alpha = float(np.real(np.vdot(d, (R_on - R_off) @ d)) / N**2)
    name = "on_off" if channel is None else f"on_off[{channel}]"
    note = (f"residual z = {z:.3f}, off-window baseline = {base:.3f}; excess power in look "
            f"direction alpha = {alpha:.3g}; K_on = {K_on:.0f}, K_off = {K_off:.0f}")
    if alpha <= 0:
        note += "; no excess power in the look direction during the on window"
    return CheckResult(
        name=name, status=_grade(ratio, warn, fail), value=ratio,
        threshold=f"residual / baseline < {warn} (WARN < {fail}); ~1 if unchanged",
        note=note, details={"alpha": alpha, "z": z, "baseline": base},
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
