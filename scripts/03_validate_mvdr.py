#!/usr/bin/env python
"""Validate DFT and MVDR beamforming on CPU-tier simulations.

Four sweeps, all with ground truth from ``charts_mvdr.sim_cpu``:

  position : SINR loss vs interferer direction (2D map + 1D cuts through the
             main beam at several frequencies, grating lobes marked).
  inr      : SINR vs interference-to-noise ratio, without quantization and
             with 4-bit quantization (symmetric and asymmetric clipping).
  k        : SINR loss vs number of samples K, against Reed-Mallett-Brennan.
  loading  : SINR loss vs diagonal loading, with steering-vector phase errors.

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
from charts_mvdr.beamforming import dft_weights, mvdr_weights, steering_vectors
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
    """MVDR weights with absolute diagonal loading (same units as R).

    ``beamforming.mvdr_weights`` takes loading as a fraction of trace(R)/N,
    which scales with the interference power; convert here so the sweep is
    expressed relative to the noise floor.
    """
    frac = loading_abs / (np.real(np.trace(R)) / R.shape[0]) if loading_abs else 0.0
    return mvdr_weights(R, a, loading=frac)[0]


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

SWEEPS = {"position": sweep_position, "inr": sweep_inr, "k": sweep_k, "loading": sweep_loading}


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
    args = p.parse_args()

    if args.quick:
        args.grid, args.cut_points, args.trials = 41, 61, 3
        args.inr_step, args.k_eval = 10.0, 4 * 1536

    args.out.mkdir(parents=True, exist_ok=True)
    for name in (SWEEPS if args.sweep == "all" else [args.sweep]):
        t0 = time.time()
        SWEEPS[name](args, args.out)
        print(f"[{name}] done in {time.time() - t0:.1f} s -> {args.out}/sweep_{name}.png", flush=True)


if __name__ == "__main__":
    main()