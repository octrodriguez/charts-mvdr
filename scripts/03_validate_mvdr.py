#!/usr/bin/env python
"""Validate DFT and MVDR beamforming on CPU-tier simulations.

Six sweeps, all with ground truth from ``charts_mvdr.sim_cpu``:

  position : SINR loss vs interferer direction (2D map + 1D cuts through the
             main beam at several frequencies, grating lobes marked).
  inr      : SINR vs interference-to-noise ratio, without quantization and
             with 4-bit quantization (symmetric and asymmetric clipping).
  k        : SINR loss vs number of samples K, against Reed-Mallett-Brennan.
  loading  : SINR loss vs diagonal loading, with steering-vector phase errors.
  offpulse : MVDR weights from the signal-in R vs the signal-free (off-pulse)
             F under pointing and per-antenna phase errors, several source SNRs.
  leo      : signal retained by MVDR in a channel with a strong LEO-like
             interferer, 2 x 2 design: static vs moving interferer, without vs
             with 4-bit quantization (Kotekan-tier frame structure: several
             1536-sample frames spaced in time).

SINR loss is reported relative to the interference-free matched-filter SINR,
SINR_0 = N p_s / sigma_n^2 (sigma_n^2 = complex noise variance).

Weight variants:
  dft       : w = a / N
  mvdr_true : MVDR with the true R_in (upper bound, = SINR_opt)
  mvdr_off  : MVDR with sample R_in (signal-free, "off-pulse")
  mvdr_on   : MVDR with sample R including the desired signal

Outputs one .npz and one .png per sweep in --out.

Example:
  python scripts/03_validate_mvdr.py --sweep all --out $SCRATCH/mvdr_validation
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr import sim_cpu as sc
from charts_mvdr.beamforming import (dft_weights, fit_source_powers, mvdr_weights,
                                     power_budget, steering_vectors)
from charts_mvdr.checks import check_saturation
from charts_mvdr.np_correlator import correlate

# Defaults taken from the quick_day test window (2026-03-20T15:00 UTC).
SUN_LM = (0.47, 0.48)
RFI_LM = (-0.830, -0.482)
NOISE_SIGMA = 1.1        # LSB per real component (measured)
P_SUN = 1.0              # complex variance of the desired signal
K_FRAME = 1536           # samples per 5.12 ms frame
N_SIDE, SPACING = 8, 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def w_dft(a: np.ndarray) -> np.ndarray:
    return dft_weights(a)[0]


def w_mvdr(R: np.ndarray, a: np.ndarray, loading_abs: float = 0.0) -> np.ndarray:
    """MVDR weights with absolute diagonal loading (same units as R)."""
    return mvdr_weights(R, a, loading=loading_abs, loading_ref="absolute")[0]


def rhat(scn, K, rng, quantize=None):
    """Simulated frame and its normalized sample covariance (single channel)."""
    x = sc.simulate_voltages(scn, K, rng, quantize=quantize)
    return x, correlate(x, normalize=True)[0]


def cos2(freq, look_lm, l, m, pos):
    """|d^H a|^2 / N^2 for interferer directions (l, m) (as in checks #12)."""
    d = sc.steer(freq, *look_lm, pos)
    A = steering_vectors(l, m, freq, pos[:, 0], pos[:, 1])
    return np.abs(A.conj() @ d) ** 2 / pos.shape[0] ** 2


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def make_scn(freq, rfi_lm, p_rfi, p_sun=P_SUN, sigma=NOISE_SIGMA, pos=None):
    """Sun first, RFI second: draws are identical for p_sun = 0 and p_sun > 0
    with the same seed, which allows paired on/off realizations."""
    return sc.Scenario(
        freqs_hz=[freq],
        sources=[sc.Source(*SUN_LM, power=p_sun, name="sun"),
                 sc.Source(*rfi_lm, power=p_rfi, name="rfi")],
        positions=sc.grid_positions(N_SIDE, N_SIDE, SPACING) if pos is None else pos,
        noise_sigma=sigma,
    )


def sinr0(n_ant, p_s=P_SUN, sigma=NOISE_SIGMA):
    return n_ant * p_s / (2.0 * sigma**2)


# ---------------------------------------------------------------------------
# Sweep 1: interferer position
# ---------------------------------------------------------------------------

def sweep_position(args, out: Path):
    freq, inr_db = args.freq, args.inr_db
    p_rfi = 2.0 * NOISE_SIGMA**2 * 10 ** (inr_db / 10)
    pos = sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    N = pos.shape[0]
    a_s = sc.steer(freq, *SUN_LM, pos)
    ref = sinr0(N)

    # 2D map, analytic only
    g = np.linspace(-1, 1, args.grid)
    L, M = np.meshgrid(g, g, indexing="xy")
    loss_dft = np.full(L.shape, np.nan)
    loss_mvdr = np.full(L.shape, np.nan)
    for idx in np.ndindex(L.shape):
        l, m = L[idx], M[idx]
        if l * l + m * m > 1.0:
            continue
        R_in = sc.analytic_covariance(make_scn(freq, (l, m), p_rfi, p_sun=0.0, pos=pos))[0]
        loss_dft[idx] = sc.output_sinr(w_dft(a_s), a_s, P_SUN, R_in) / ref
        loss_mvdr[idx] = sc.optimal_sinr(a_s, P_SUN, R_in) / ref

    # 1D cuts along m = m_sun, several frequencies, analytic + sampled
    cut_freqs = args.cut_freqs
    m0 = SUN_LM[1]
    lmax = np.sqrt(1 - m0**2) - 1e-6
    lc = np.linspace(-lmax, lmax, args.cut_points)
    cuts = {}
    rng = np.random.default_rng(args.seed)
    for f in cut_freqs:
        a_f = sc.steer(f, *SUN_LM, pos)
        res = {k: np.zeros(lc.size) for k in ("dft", "mvdr_true", "mvdr_off", "mvdr_on")}
        for i, l in enumerate(lc):
            scn_on = make_scn(f, (l, m0), p_rfi, pos=pos)
            scn_off = make_scn(f, (l, m0), p_rfi, p_sun=0.0, pos=pos)
            R_in = sc.analytic_covariance(scn_off)[0]
            res["dft"][i] = sc.output_sinr(w_dft(a_f), a_f, P_SUN, R_in)
            res["mvdr_true"][i] = sc.optimal_sinr(a_f, P_SUN, R_in)
            off, on = 0.0, 0.0
            for _ in range(args.trials):
                _, Ro = rhat(scn_off, K_FRAME, rng)
                _, Rn = rhat(scn_on, K_FRAME, rng)
                off += sc.output_sinr(w_mvdr(Ro, a_f), a_f, P_SUN, R_in)
                on += sc.output_sinr(w_mvdr(Rn, a_f), a_f, P_SUN, R_in)
            res["mvdr_off"][i] = off / args.trials
            res["mvdr_on"][i] = on / args.trials
        cuts[f] = {k: v / ref for k, v in res.items()}
        # Strong-interferer limit of SINR_opt / SINR_0 (checks #12)
        print(f"[position] cut {f/1e6:.1f} MHz done", flush=True)
        cuts[f]["theory"] = 1.0 - cos2(f, SUN_LM, lc, np.full_like(lc, m0), pos)

    np.savez(out / "sweep_position.npz", L=L, M=M, loss_dft=loss_dft, loss_mvdr=loss_mvdr,
             l_cut=lc, m_cut=m0, cut_freqs=cut_freqs,
             **{f"cut_{int(f/1e3)}kHz_{k}": v for f in cut_freqs for k, v in cuts[f].items()},
             freq=freq, inr_db=inr_db)

    fig = plt.figure(figsize=(13, 4 + 2.6 * len(cut_freqs)))
    gs = fig.add_gridspec(1 + len(cut_freqs), 2, height_ratios=[1.6] + [1] * len(cut_freqs))
    for j, (data, title) in enumerate([(loss_dft, "DFT"), (loss_mvdr, "MVDR (true $R_{in}$)")]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.pcolormesh(L, M, db(data), vmin=-40, vmax=0, shading="auto", cmap="viridis")
        ax.plot(*SUN_LM, "w*", ms=12, label="look direction")
        for la, ma in sc.grating_lobes(freq, *SUN_LM, SPACING):
            ax.plot(la, ma, "wo", mfc="none", ms=10)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="w", lw=0.8))
        ax.set_aspect("equal"); ax.set_xlabel("$l_{RFI}$"); ax.set_ylabel("$m_{RFI}$")
        ax.set_title(f"{title}: SINR loss [dB], {freq/1e6:.1f} MHz, INR {inr_db:.0f} dB")
        fig.colorbar(im, ax=ax)
    for r, f in enumerate(cut_freqs, start=1):
        ax = fig.add_subplot(gs[r, :])
        for k, style in [("dft", "k-"), ("mvdr_true", "C0-"), ("mvdr_off", "C1--"),
                         ("mvdr_on", "C3:"), ("theory", "C2-.")]:
            ax.plot(lc, db(cuts[f][k]), style, label=k)
        ax.axvline(SUN_LM[0], color="0.5", lw=0.8)
        for la, ma in sc.grating_lobes(f, *SUN_LM, SPACING, visible_only=False):
            if abs(ma - m0) < 1e-9 and abs(la) <= lmax:
                ax.axvline(la, color="C2", ls="--", lw=0.8)
        ax.set_ylim(-45, 3); ax.set_xlabel(f"$l_{{RFI}}$ at $m = {m0}$")
        ax.set_ylabel("SINR loss [dB]"); ax.set_title(f"{f/1e6:.1f} MHz (grey: look, green: grating lobe)")
        ax.grid(alpha=0.3)
        if r == 1:
            ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sweep_position.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 2: INR and quantization
# ---------------------------------------------------------------------------

def _eval_covariances(scn_on, scn_off, k_eval, seed, quant):
    """Paired on/off realizations with the same seed (identical noise and RFI
    draws). Returns their sample covariances so any number of weights can be
    evaluated without re-simulating: SINR = (w^H R_on w - w^H R_off w) / w^H R_off w.
    """
    _, Ron = rhat(scn_on, k_eval, seed, quantize=quant)
    _, Roff = rhat(scn_off, k_eval, seed, quantize=quant)
    return Ron, Roff


def _empirical_sinr(w, Ron, Roff):
    pon = np.real(np.vdot(w, Ron @ w))
    poff = np.real(np.vdot(w, Roff @ w))
    return (pon - poff) / poff


def sweep_inr(args, out: Path):
    freq = args.freq
    inrs = np.arange(args.inr_min, args.inr_max + 1e-9, args.inr_step)
    modes = [None, "symmetric", "asymmetric"]
    pos = sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    a_s = sc.steer(freq, *SUN_LM, pos)
    ref = sinr0(pos.shape[0])
    res = {(m, w): np.zeros(inrs.size) for m in modes for w in ("dft", "mvdr_on")}
    opt = np.zeros(inrs.size)
    rails = {m: np.zeros(inrs.size) for m in modes[1:]}
    for i, inr in enumerate(inrs):
        p_rfi = 2.0 * NOISE_SIGMA**2 * 10 ** (inr / 10)
        scn_on = make_scn(freq, RFI_LM, p_rfi, pos=pos)
        scn_off = make_scn(freq, RFI_LM, p_rfi, p_sun=0.0, pos=pos)
        opt[i] = sc.optimal_sinr(a_s, P_SUN, sc.analytic_covariance(scn_off)[0]) / ref
        for m in modes:
            acc = {"dft": 0.0, "mvdr_on": 0.0}
            for t in range(args.trials):
                seed = args.seed + 1000 * t + i
                x, R = rhat(scn_on, K_FRAME, seed, quantize=m)
                if m is not None:
                    rails[m][i] += check_saturation(x[:, 0, :]).value / args.trials
                Ron, Roff = _eval_covariances(scn_on, scn_off, args.k_eval, seed + 10**6, m)
                for wname, w in (("dft", w_dft(a_s)), ("mvdr_on", w_mvdr(R, a_s))):
                    acc[wname] += _empirical_sinr(w, Ron, Roff) / args.trials
            for wname in acc:
                res[(m, wname)][i] = acc[wname] / ref
        print(f"[inr] {inr:+.1f} dB done", flush=True)

    np.savez(out / "sweep_inr.npz", inr_db=inrs, opt=opt,
             **{f"{m or 'none'}_{w}": v for (m, w), v in res.items()},
             **{f"rails_{m}": v for m, v in rails.items()}, freq=freq)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.5, 1]})
    ax.plot(inrs, db(opt), "k-", lw=2, label="SINR$_{opt}$ (analytic)")
    colors = {None: "C0", "symmetric": "C1", "asymmetric": "C3"}
    for m in modes:
        lab = m or "no quantization"
        ax.plot(inrs, db(res[(m, "mvdr_on")]), "-o", ms=3, color=colors[m], label=f"MVDR, {lab}")
        ax.plot(inrs, db(res[(m, "dft")]), "--", color=colors[m], label=f"DFT, {lab}")
    ax.axhline(db(np.pi / 4), color="0.5", ls=":", lw=1, label=r"$\pi/4$ (hard limiter)")
    ax.set_ylabel("SINR loss [dB]"); ax.set_ylim(-40, 3); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{freq/1e6:.1f} MHz, RFI at {RFI_LM}, $\\sigma$ = {NOISE_SIGMA} LSB")
    for m in modes[1:]:
        ax2.plot(inrs, 100 * rails[m], color=colors[m], label=m)
    ax2.set_xlabel("INR per element [dB]"); ax2.set_ylabel("components at rail [%]")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sweep_inr.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 3: number of samples
# ---------------------------------------------------------------------------

def sweep_k(args, out: Path):
    freq = args.freq
    pos = sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    N = pos.shape[0]
    a_s = sc.steer(freq, *SUN_LM, pos)
    p_rfi = 2.0 * NOISE_SIGMA**2 * 10 ** (args.inr_db / 10)
    scn_on = make_scn(freq, RFI_LM, p_rfi, pos=pos)
    scn_off = make_scn(freq, RFI_LM, p_rfi, p_sun=0.0, pos=pos)
    R_in = sc.analytic_covariance(scn_off)[0]
    opt = sc.optimal_sinr(a_s, P_SUN, R_in)
    Ks = np.unique(np.round(np.geomspace(N + 8, 16 * K_FRAME, 14)).astype(int))
    rng = np.random.default_rng(args.seed)
    rho_off = np.zeros((Ks.size, args.trials)); rho_on = np.zeros_like(rho_off)
    for i, K in enumerate(Ks):
        for t in range(args.trials):
            _, Ro = rhat(scn_off, K, rng)
            _, Rn = rhat(scn_on, K, rng)
            rho_off[i, t] = sc.output_sinr(w_mvdr(Ro, a_s), a_s, P_SUN, R_in) / opt
            rho_on[i, t] = sc.output_sinr(w_mvdr(Rn, a_s), a_s, P_SUN, R_in) / opt
    rmb = np.array([sc.rmb_expected_loss(N, K) for K in Ks])
    np.savez(out / "sweep_k.npz", K=Ks, rho_off=rho_off, rho_on=rho_on, rmb=rmb, freq=freq)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(Ks, db(rmb), "k-", lw=2, label="RMB $(K+2-N)/(K+1)$")
    for data, c, lab in [(rho_off, "C1", "signal-free $\\hat R_{in}$"), (rho_on, "C3", "signal-in $\\hat R$")]:
        mu = db(data.mean(1)); lo, hi = db(np.percentile(data, 10, 1)), db(np.percentile(data, 90, 1))
        ax.semilogx(Ks, mu, "o-", color=c, ms=3, label=lab)
        ax.fill_between(Ks, lo, hi, color=c, alpha=0.2)
    ax.axvline(K_FRAME, color="0.5", ls=":", label=f"one frame (K={K_FRAME})")
    ax.set_xlabel("K"); ax.set_ylabel("SINR / SINR$_{opt}$ [dB]"); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"N = {N}, INR {args.inr_db:.0f} dB, per-element SNR {db(P_SUN/(2*NOISE_SIGMA**2)):.1f} dB")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sweep_k.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 4: diagonal loading vs steering-vector mismatch
# ---------------------------------------------------------------------------

def sweep_loading(args, out: Path):
    freq = args.freq
    pos = sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    N = pos.shape[0]
    a_nom = sc.steer(freq, *SUN_LM, pos)
    p_rfi = 2.0 * NOISE_SIGMA**2 * 10 ** (args.inr_db / 10)
    sig_n2 = 2.0 * NOISE_SIGMA**2
    loads = np.concatenate([[0.0], np.geomspace(1e-2, 1e5, 29)])   # in units of sigma_n^2
    errs_deg = args.phase_err_deg
    rng = np.random.default_rng(args.seed)
    ref = sinr0(N)
    res = {e: np.zeros((loads.size, args.trials)) for e in errs_deg}
    dft = {e: np.zeros(args.trials) for e in errs_deg}
    for e in errs_deg:
        for t in range(args.trials):
            phi = np.deg2rad(e) * rng.standard_normal(N)
            gains = np.exp(1j * phi)       # true response = nominal * gains (signal and RFI)
            scn_on = make_scn(freq, RFI_LM, p_rfi, pos=pos); scn_on.gains = gains
            scn_off = make_scn(freq, RFI_LM, p_rfi, p_sun=0.0, pos=pos); scn_off.gains = gains
            R_in = sc.analytic_covariance(scn_off)[0]
            a_true = gains * a_nom
            _, R = rhat(scn_on, K_FRAME, rng)
            dft[e][t] = sc.output_sinr(w_dft(a_nom), a_true, P_SUN, R_in) / ref
            for j, d in enumerate(loads):
                res[e][j, t] = sc.output_sinr(w_mvdr(R, a_nom, d * sig_n2), a_true, P_SUN, R_in) / ref
    np.savez(out / "sweep_loading.npz", loading=loads, phase_err_deg=errs_deg,
             **{f"mvdr_on_{e}deg": v for e, v in res.items()},
             **{f"dft_{e}deg": v for e, v in dft.items()}, freq=freq)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.where(loads == 0, loads[1] / 3, loads)
    for k, e in enumerate(errs_deg):
        c = f"C{k}"
        ax.semilogx(x, db(res[e].mean(1)), "o-", ms=3, color=c, label=f"MVDR signal-in, {e}° rms")
        ax.axhline(db(dft[e].mean()), color=c, ls=":", lw=1)
    ax.set_xlabel(r"diagonal loading $\delta / \sigma_n^2$ (leftmost point: $\delta = 0$)")
    ax.set_ylabel("SINR loss [dB]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"K = {K_FRAME}, INR {args.inr_db:.0f} dB (dotted: DFT with same errors)")
    fig.tight_layout(); fig.savefig(out / "sweep_loading.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 5: strong moving interferer (LEO) and 4-bit quantization
# ---------------------------------------------------------------------------

def frame_times(n_frames: int, spacing_s: float, k_frame: int = K_FRAME,
                dt_s: float = 10e-6 / 3) -> np.ndarray:
    """Sample times of n_frames frames of k_frame samples, frame starts spaced by spacing_s."""
    return (np.arange(n_frames)[:, None] * spacing_s + np.arange(k_frame)[None, :] * dt_s).ravel()


def sweep_leo(args, out: Path):
    """Separate the effects of interferer motion and 4-bit saturation on MVDR.

    For each case and trial, paired realizations with the Sun on and off (same
    noise and interferer draws) give R_on and R_off. MVDR weights come from R_on
    (signal in the training data, as on real data). Two figures of merit:
      retained_true  w^H (R_on - R_off) w / d^H (R_on - R_off) d * N^2: signal
                     power through MVDR relative to the DFT (ground truth,
                     includes the effect of quantization on the signal)
      retained_fit   signal / p_look from the power budget of 05_mvdr_correlations
                     (fit with the interferer at its position at the start of the
                     window, as given by an events file)
    If retained_fit drops while retained_true does not, the loss seen at the
    Kotekan tier is an artifact of the budget model, not of the beamformer.
    """
    f, pos = args.leo_freq, sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    N = pos.shape[0]
    times = frame_times(args.leo_frames, args.leo_spacing)
    K = times.size
    d = sc.steer(f, *SUN_LM, pos)
    a_start = sc.steer(f, *args.leo_lm, pos)
    p_leo = args.leo_amp ** 2                    # complex variance (amplitude A, w ~ CN(0, 1))
    cases = [(mv, q) for mv in (False, True) for q in (None, "symmetric")]
    loads = (0.0, 10.0)
    keys = ("true", "fit", "wng_db", "null_db")
    res = {(c, dl, k): np.zeros(args.trials) for c in cases for dl in loads for k in keys}
    for t in range(args.trials):
        seed = args.seed + t
        for c in cases:
            moving, quant = c
            drift = tuple(args.leo_drift) if moving else (0.0, 0.0)
            scn = lambda p_sun: sc.Scenario(
                freqs_hz=[f], positions=pos, noise_sigma=args.leo_sigma,
                sources=[sc.Source(*SUN_LM, power=p_sun, name="sun"),
                         sc.Source(*args.leo_lm, power=p_leo, name="leo", drift=drift)])
            x_on = sc.simulate_voltages(scn(args.leo_p_sun), K, seed, quantize=quant, times_s=times)
            x_off = sc.simulate_voltages(scn(0.0), K, seed, quantize=quant, times_s=times)
            R_on = correlate(x_on.astype(np.complex128), normalize=True)[0]
            dR = R_on - correlate(x_off.astype(np.complex128), normalize=True)[0]
            sig_dft = np.real(np.vdot(d, dR @ d)) / N**2
            A = np.vstack([d, a_start])
            p, noise = fit_source_powers(R_on, A)
            for dl in loads:
                w = mvdr_weights(R_on, d, loading=dl, loading_ref="noise")[0]
                b = power_budget(R_on, w, A, p, noise)
                res[(c, dl, "true")][t] = np.real(np.vdot(w, dR @ w)) / sig_dft
                res[(c, dl, "fit")][t] = (b["total"] - b["noise"] - b["source"][1]) / p[0]
                res[(c, dl, "wng_db")][t] = db(b["wng"])
                res[(c, dl, "null_db")][t] = db(np.abs(np.vdot(w, a_start)) ** 2)
        print(f"[leo] trial {t + 1}/{args.trials} done", flush=True)

    drift_win = np.hypot(*args.leo_drift) * (times[-1] - times[0])
    print(f"\n[leo] {f / 1e6:.1f} MHz, K = {K} ({args.leo_frames} frames over "
          f"{times[-1] - times[0]:.2f} s), interferer A = {args.leo_amp} LSB at {tuple(args.leo_lm)}, "
          f"drift {tuple(args.leo_drift)} /s ({drift_win:.4f} over the window), "
          f"sigma = {args.leo_sigma} LSB, p_sun = {args.leo_p_sun}")
    print(f"{'motion':>7} {'quant':>9} {'delta':>6} {'retained_true':>14} {'retained_fit':>13} "
          f"{'WNG dB':>7} {'null dB':>8}")
    for c in cases:
        for dl in loads:
            r = {k: res[(c, dl, k)] for k in keys}
            print(f"{'moving' if c[0] else 'static':>7} {c[1] or 'none':>9} {dl:6.0f} "
                  f"{r['true'].mean():8.3f} ±{r['true'].std():.3f} "
                  f"{r['fit'].mean():7.3f} ±{r['fit'].std():.3f} "
                  f"{r['wng_db'].mean():7.2f} {r['null_db'].mean():8.1f}")

    np.savez(out / "sweep_leo.npz", freq=f, K=K, leo_lm=args.leo_lm, leo_drift=args.leo_drift,
             leo_amp=args.leo_amp, sigma=args.leo_sigma, p_sun=args.leo_p_sun, loadings=loads,
             **{f"{'moving' if c[0] else 'static'}_{c[1] or 'none'}_d{dl:g}_{k}": v
                for (c, dl, k), v in res.items()})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    xlab = [f"{'moving' if c[0] else 'static'}\n{c[1] or 'no quant.'}" for c in cases]
    xx = np.arange(len(cases))
    for ax, dl in zip(axes, loads):
        for j, (k, lab) in enumerate((("true", "retained (true)"), ("fit", "retained (budget fit)"))):
            m = [res[(c, dl, k)].mean() for c in cases]
            e = [res[(c, dl, k)].std() for c in cases]
            ax.bar(xx + (j - 0.5) * 0.36, m, 0.36, yerr=e, capsize=3, label=lab)
        ax.set_xticks(xx, xlab); ax.axhline(1.0, color="0.4", lw=0.8)
        ax.set_title(f"MVDR, loading {dl:g} $\\hat\\sigma_n^2$"); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("signal retained"); axes[0].set_ylim(0.5, 1.1); axes[0].legend(fontsize=8)
    fig.suptitle(f"Strong interferer A = {args.leo_amp} LSB, K = {K}", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "sweep_leo.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep 6: signal-in vs signal-free (off-pulse) covariance under steering errors
# ---------------------------------------------------------------------------

OP_BEAMS = ("dft", "on_d0", "off_d0", "on_d10", "off_d10")


def sweep_offpulse(args, out: Path):
    """MVDR weights from the signal-in covariance R vs the signal-free F.

    The KL filter of Andrew et al. (b proportional to F^-1 d) takes F from
    off-pulse data. With an exact steering vector F^-1 d and R^-1 d have the same
    direction (matrix inversion lemma), so both give the same SINR up to finite-K
    effects. With a steering error the source in R is seen as an interferer
    slightly off the look direction and MVDR partially nulls it (self-nulling);
    the loss grows with the source SINR. F does not contain the source, so it
    cannot self-null.

    For each array SNR (SINR_0 = N p_s / sigma_n^2) and each steering error,
    independent training sets of K samples give R (source on) and F (source
    off); SINR is evaluated with the true steering vector and the true R_in.
    Two error types:
      pointing  true source at (l0 + dl, m0), weights steered to (l0, m0)
      phase     per-antenna phase errors (rms, applied to all sources, as
                uncalibrated complex gains), weights with the nominal vector
    Loadings 0 and 10 sigma_n^2 (absolute, true noise power).
    """
    freq = args.freq
    pos = sc.grid_positions(N_SIDE, N_SIDE, SPACING)
    N = pos.shape[0]
    sig_n2 = 2.0 * NOISE_SIGMA ** 2
    p_rfi = sig_n2 * 10 ** (args.inr_db / 10)
    d = sc.steer(freq, *SUN_LM, pos)
    rng = np.random.default_rng(args.seed)
    errors = {"pointing": np.asarray(args.op_pointing, float),
              "phase": np.asarray(args.op_phase_deg, float)}
    snrs = np.asarray(args.op_snr_db, float)
    res = {(s, kind, i, b): np.zeros(args.trials) for s in snrs for kind in errors
           for i in range(errors[kind].size) for b in OP_BEAMS}
    wng = {(s, kind, i, b): np.zeros(args.trials) for s in snrs for kind in errors
           for i in range(errors[kind].size) for b in OP_BEAMS[1:]}
    opt = {(s, kind, i): np.zeros(args.trials) for s in snrs for kind in errors
           for i in range(errors[kind].size)}

    for s in snrs:
        p_s = sig_n2 * 10 ** (s / 10) / N
        for kind, vals in errors.items():
            for i, e in enumerate(vals):
                for t in range(args.trials):
                    if kind == "pointing":
                        src_lm, gains = (SUN_LM[0] + e, SUN_LM[1]), None
                    else:
                        src_lm = SUN_LM
                        gains = np.exp(1j * np.deg2rad(e) * rng.standard_normal(N))

                    def scn(p_sun):
                        return sc.Scenario(
                            freqs_hz=[freq], positions=pos, noise_sigma=NOISE_SIGMA, gains=gains,
                            sources=[sc.Source(*src_lm, power=p_sun, name="sun"),
                                     sc.Source(*RFI_LM, power=p_rfi, name="rfi")])

                    R_in = sc.analytic_covariance(scn(0.0))[0]
                    g = np.ones(N) if gains is None else gains
                    a_true = g * sc.steer(freq, *src_lm, pos)
                    _, R = rhat(scn(p_s), args.op_k, rng.integers(2 ** 32))
                    _, F = rhat(scn(0.0), args.op_k, rng.integers(2 ** 32))
                    opt[(s, kind, i)][t] = sc.optimal_sinr(a_true, p_s, R_in)
                    weights = {"dft": w_dft(d)}
                    for name, C in (("on", R), ("off", F)):
                        for dl in (0, 10):
                            weights[f"{name}_d{dl}"] = w_mvdr(C, d, dl * sig_n2)
                    for b, w in weights.items():
                        res[(s, kind, i, b)][t] = sc.output_sinr(w, a_true, p_s, R_in)
                        if b != "dft":
                            wng[(s, kind, i, b)][t] = db(N * np.sum(np.abs(w) ** 2))
            print(f"[offpulse] SNR {s:+.0f} dB done", flush=True)

    ref = {s: sig_n2 * 10 ** (s / 10) for s in snrs}       # SINR_0 = N p_s / sigma_n^2
    print(f"\n[offpulse] {freq / 1e6:.1f} MHz, K = {args.op_k}, INR {args.inr_db:.0f} dB per "
          f"element, RFI at {RFI_LM}; SINR loss [dB] relative to SINR_0 = N p_s / sigma_n^2")
    print(f"{'SNR':>5} {'error':>9} {'value':>7} {'opt':>6} " + " ".join(f"{b:>8}" for b in OP_BEAMS)
          + "   WNG on_d0 / off_d0 [dB]")
    for s in snrs:
        for kind, vals in errors.items():
            for i, e in enumerate(vals):
                lo = [db(res[(s, kind, i, b)].mean() / ref[s] * sig_n2) for b in OP_BEAMS]
                print(f"{s:5.0f} {kind:>9} {e:7.4g} {db(opt[(s, kind, i)].mean() / ref[s] * sig_n2):6.2f} "
                      + " ".join(f"{v:8.2f}" for v in lo)
                      + f"   {wng[(s, kind, i, 'on_d0')].mean():5.2f} / "
                        f"{wng[(s, kind, i, 'off_d0')].mean():5.2f}")

    np.savez(out / "sweep_offpulse.npz", freq=freq, K=args.op_k, inr_db=args.inr_db,
             snr_db=snrs, pointing=errors["pointing"], phase_deg=errors["phase"],
             **{f"snr{s:g}_{kind}{i}_{b}": v for (s, kind, i, b), v in res.items()},
             **{f"snr{s:g}_{kind}{i}_{b}_wng_db": v for (s, kind, i, b), v in wng.items()},
             **{f"snr{s:g}_{kind}{i}_opt": v for (s, kind, i), v in opt.items()})

    fig, axes = plt.subplots(2, snrs.size, figsize=(4.2 * snrs.size, 7.5), squeeze=False,
                             sharey="row")
    style = {"dft": ("k", ":"), "on_d0": ("C3", "-"), "off_d0": ("C0", "-"),
             "on_d10": ("C3", "--"), "off_d10": ("C0", "--")}
    label = {"dft": "DFT", "on_d0": r"$R^{-1}d$, $\delta=0$", "off_d0": r"$F^{-1}d$, $\delta=0$",
             "on_d10": r"$R^{-1}d$, $\delta=10\sigma_n^2$",
             "off_d10": r"$F^{-1}d$, $\delta=10\sigma_n^2$"}
    xlab = {"pointing": r"pointing error $\Delta l$", "phase": "phase error rms [deg]"}
    for c, s in enumerate(snrs):
        for r, (kind, vals) in enumerate(errors.items()):
            ax = axes[r, c]
            ax.plot(vals, [db(opt[(s, kind, i)].mean() / ref[s] * sig_n2) for i in range(vals.size)],
                    color="0.6", lw=3, alpha=0.6, label="optimum (true $a$, $R_{in}$)")
            for b in OP_BEAMS:
                ax.plot(vals, [db(res[(s, kind, i, b)].mean() / ref[s] * sig_n2)
                               for i in range(vals.size)], "o", ms=3, color=style[b][0],
                        ls=style[b][1], label=label[b])
            ax.set_xlabel(xlab[kind]); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"array SNR {s:g} dB")
            if c == 0:
                ax.set_ylabel("SINR loss [dB]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Signal-in vs signal-free covariance: {freq / 1e6:.1f} MHz, K = {args.op_k}, "
                 f"INR {args.inr_db:.0f} dB", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "sweep_offpulse.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------

SWEEPS = {"position": sweep_position, "inr": sweep_inr, "k": sweep_k, "loading": sweep_loading,
          "leo": sweep_leo, "offpulse": sweep_offpulse}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", choices=[*SWEEPS, "all"], default="all")
    p.add_argument("--out", type=Path, default=Path("mvdr_validation"))
    p.add_argument("--freq", type=float, default=344.1e6, help="Hz (default: persistent-RFI channel 94)")
    p.add_argument("--inr-db", type=float, default=20.0, help="per-element INR for position/k/loading")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="coarse grids for a smoke test")
    # position
    p.add_argument("--grid", type=int, default=121)
    p.add_argument("--cut-points", type=int, default=241)
    p.add_argument("--cut-freqs", type=float, nargs="+", default=[300e6, 344.1e6, 400e6])
    # inr
    p.add_argument("--inr-min", type=float, default=-10.0)
    p.add_argument("--inr-max", type=float, default=30.0)
    p.add_argument("--inr-step", type=float, default=2.5)
    p.add_argument("--k-eval", type=int, default=32 * 1536)
    # loading
    p.add_argument("--phase-err-deg", type=float, nargs="+", default=[0.0, 5.0, 15.0])
    # leo
    p.add_argument("--leo-freq", type=float, default=349.8e6, help="Hz")
    p.add_argument("--leo-lm", type=float, nargs=2, default=[0.089, 0.071],
                   help="interferer (l, m) at the start of the window")
    p.add_argument("--leo-drift", type=float, nargs=2, default=[0.018, 0.018],
                   help="interferer drift (dl/dt, dm/dt) in 1/s for the moving case")
    p.add_argument("--leo-amp", type=float, default=12.2, help="interferer amplitude [LSB]")
    p.add_argument("--leo-sigma", type=float, default=2.0, help="noise std per component [LSB]")
    p.add_argument("--leo-p-sun", type=float, default=0.43, help="desired-signal complex variance")
    p.add_argument("--leo-frames", type=int, default=11)
    p.add_argument("--leo-spacing", type=float, default=0.05, help="frame spacing [s]")
    # offpulse
    p.add_argument("--op-snr-db", type=float, nargs="+", default=[0.0, 10.0, 20.0],
                   help="array SNR N p_s / sigma_n^2 of the desired source [dB]")
    p.add_argument("--op-pointing", type=float, nargs="+",
                   default=[0.0, 0.0025, 0.005, 0.01, 0.02], help="pointing errors in l")
    p.add_argument("--op-phase-deg", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0],
                   help="per-antenna phase errors, rms [deg]")
    p.add_argument("--op-k", type=int, default=8 * 1536,
                   help="training samples for R and for F (each)")
    args = p.parse_args()

    if args.quick:
        args.grid, args.cut_points, args.trials = 41, 61, 3
        args.inr_step, args.k_eval = 10.0, 4 * 1536
        args.leo_frames = 3
        args.op_k = 1536

    args.out.mkdir(parents=True, exist_ok=True)
    for name in (SWEEPS if args.sweep == "all" else [args.sweep]):
        t0 = time.time()
        SWEEPS[name](args, args.out)
        print(f"[{name}] done in {time.time() - t0:.1f} s -> {args.out}/sweep_{name}.png", flush=True)


if __name__ == "__main__":
    main()
