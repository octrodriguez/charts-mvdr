#!/usr/bin/env python
"""Sun fringe recovery in RFI bands after nulling the interference in the visibilities.

The MVDR beamformer gives one beam, a power per frame; fringes live in the visibilities of
each antenna pair. This script filters the calibrated visibilities of a cube saved by
16_rfi_census.py extract and compares them with the Sun's fringe model filtered the same
way, channel by channel:

  --filter mvdr     Q = (F + delta I)^-1 with the stage's loading rule (--loading,
                    --loading-ref); d^H (Q V Q) d / |d^H Q d|^2 is the MVDR beam power for
                    any steering d, so Q V Q is what the MVDR sees, pair by pair.
  --filter project  remove interference eigenmodes of the noise-whitened covariance:
                    --k N (fixed) or --k auto (eigenvalue > --thresh x the smallest).

F (or the eigenmodes) comes from the time average of the whole cube (--window 0: the Sun's
moving fringes average out) or from a running window of +-W frames (as the stage; then the
Sun enters F and is partly nulled). Outputs:
  OUT_phase2d.png      phase of one pair relative to the model, raw and filtered, time x
                       frequency (0 = follows the Sun); the default diagnostic
  OUT_correlation.png  fringe correlation per channel, raw and filtered (1 = follows the Sun)
  OUT_examples.png     (--examples) time series of one pair against the model
  OUT.npz              freqs, rho (2, n_chan), k_used, parameters
If the cube holds a 'look' array (n_frames, 2), it replaces the Sun's direction.

Example:
  python scripts/19_fringe_recovery.py --vis CUBE.npz --gains GAINS.npy --gain-antennas 0,4,32,36 \\
      --positions POS.npy --filter mvdr --bands 357-380 410-428 381.5-399.5 --out OUT/fringe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from charts_mvdr import nulling


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", required=True, help="Cube from 16_rfi_census.py extract.")
    ap.add_argument("--gains", required=True, help="charts-utils gains, one column per --gain-antennas.")
    ap.add_argument("--gain-antennas", required=True)
    ap.add_argument("--positions", required=True, help="Positions .npy (n, 3), ENU m, logical order.")
    ap.add_argument("--filter", default="mvdr", choices=("mvdr", "project"))
    ap.add_argument("--loading", type=float, default=0.1)
    ap.add_argument("--loading-ref", default="median", choices=nulling.LOADING_REFS)
    ap.add_argument("--k", default="auto", help="project: number of modes or 'auto'.")
    ap.add_argument("--thresh", type=float, default=2.0, help="project --k auto: eigenvalue threshold relative to the smallest (whitened).")
    ap.add_argument("--window", type=int, default=0, help="Half-width in frames of a running window (0 = whole cube).")
    ap.add_argument("--bands", nargs="*", default=[], help="MHz ranges summarized; the last one is drawn as reference.")
    ap.add_argument("--pair", default=None, help="Pair for the 2-D phase figure (default: first two antennas).")
    ap.add_argument("--examples", nargs="*", type=float, default=[], help="Channels [MHz] for time-series examples.")
    ap.add_argument("--out", required=True, help="Output prefix.")
    return ap.parse_args(argv)


def rng(s):
    lo, hi = (float(x) for x in s.split("-"))
    return lo, hi


def main(argv=None) -> int:
    args = parse_args(argv)
    ants = [int(a) for a in args.gain_antennas.split(",")]
    if args.filter == "project" and args.k != "auto" and not args.k.isdigit():
        print("error: --k must be an integer or 'auto'", file=sys.stderr)
        return 1
    V, freqs, times, look = nulling.load_cube(args.vis, ants)
    g = np.load(args.gains)
    if g.shape != (len(freqs), len(ants)):
        print(f"error: gains {g.shape} do not match ({len(freqs)}, {len(ants)})", file=sys.stderr)
        return 1
    Vc, ok = nulling.calibrate(V, g, freqs)
    pos = np.load(args.positions)
    if look is None:
        look = nulling.sun_lm(times, pos)
    x, y = pos[ants, 0], pos[ants, 1]
    pair = [ants.index(int(a)) for a in args.pair.split(",")] if args.pair else [0, 1]
    n_t, n_c, n = Vc.shape[:3]
    rho = np.full((2, n_c), np.nan)
    k_used = np.zeros(n_c, dtype=int)
    ph = np.full((2, n_t, n_c), np.nan)
    ex_ch = [int(np.argmin(np.abs(freqs - f))) for f in args.examples]
    ex = {}
    for c in np.flatnonzero(ok):
        D = Vc[:, c]
        M = nulling.sun_model(look, freqs[c] * 1e6, x, y)
        F = nulling.local_mean(D, args.window) if args.window > 0 else D.mean(axis=0)[None]
        if args.filter == "mvdr":
            Q = nulling.mvdr_filter(F, args.loading, args.loading_ref)
        elif args.k == "auto":
            Q, ku = nulling.whitened_projector(F, thresh=args.thresh)
            k_used[c] = int(np.max(ku))
        else:
            Q, ku = nulling.whitened_projector(F, k=int(args.k))
            k_used[c] = int(np.max(ku))
        Fd = Q @ D @ np.conj(np.swapaxes(Q, -1, -2))
        Fm = Q @ M @ np.conj(np.swapaxes(Q, -1, -2))
        rho[0, c] = nulling.fringe_correlation(D, M)
        rho[1, c] = nulling.fringe_correlation(Fd, Fm)
        i, j = pair
        ph[0, :, c] = np.angle(D[:, i, j] * np.conj(M[:, i, j]))
        ph[1, :, c] = np.angle(Fd[:, i, j] * np.conj(Fm[:, i, j]))
        if c in ex_ch:
            ex[c] = (D, M, Fd, Fm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    label = (f"MVDR filter, loading {args.loading:g} x {args.loading_ref} eigenvalue" if args.filter == "mvdr"
             else (f"projection, k = {args.k}" + (f" (eigenvalue > {args.thresh:g})" if args.k == "auto" else "")))
    label += f", window +-{args.window} frames" if args.window > 0 else ", whole cube"
    np.savez_compressed(f"{out}.npz", freqs=freqs, rho=rho, k_used=k_used, ants=np.array(ants),
                        filter=args.filter, k=str(args.k), thresh=args.thresh, loading=args.loading,
                        loading_ref=args.loading_ref, window=args.window)
    plot(out, freqs, times, rho, ph, ex, ex_ch, ants, pair, label, args.bands)
    for b in args.bands:
        lo, hi = rng(b)
        m = (freqs >= lo) & (freqs <= hi)
        line = (f"{b} MHz: raw {np.nanmedian(rho[0, m]):.2f}, filtered {np.nanmedian(rho[1, m]):.2f} "
                f"(median over {m.sum()} channels)")
        if args.filter == "project":
            line += f", modes removed median {np.median(k_used[m]):.0f}"
        print(line)
    print(f"saved {out}.npz and figures ({label})")
    return 0


def plot(out, freqs, times, rho, ph, ex, ex_ch, ants, pair, label, bands):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tm = (times - times[0]) / 60.0
    i, j = pair
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for k, lab in enumerate(["raw", "filtered"]):
        im = axs[k].pcolormesh(tm, freqs, ph[k].T, cmap="twilight", vmin=-np.pi, vmax=np.pi, shading="auto")
        axs[k].set_title(f"V({ants[i]},{ants[j]}) phase relative to the Sun model, {lab}", loc="left", fontsize=9)
        axs[k].set_xlabel("time [min]")
    axs[0].set_ylabel("Frequency [MHz]")
    fig.colorbar(im, ax=axs, label="phase [rad] (0 = follows the Sun)")
    fig.suptitle(label, fontsize=9)
    fig.savefig(f"{out}_phase2d.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for k, b in enumerate(bands):
        lo, hi = rng(b)
        ax.axvspan(lo, hi, color="tab:green" if k == len(bands) - 1 else "tab:red", alpha=0.1, lw=0)
    ax.plot(freqs, rho[0], ".", ms=3, color="0.5", label="raw")
    ax.plot(freqs, rho[1], ".", ms=3, color="tab:blue", label=label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("correlation with Sun fringe model")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}_correlation.png", dpi=130)
    plt.close(fig)

    if not ex:
        return

    def norm(v):
        v = v - v.mean()
        s = np.sqrt(np.mean(np.abs(v) ** 2))
        return v / s if s > 0 else v

    chans = [c for c in ex_ch if c in ex]
    fig, axs = plt.subplots(len(chans), 2, figsize=(12, 2.6 * len(chans)), sharex=True, squeeze=False)
    for r, c in enumerate(chans):
        D, M, Fd, Fm = ex[c]
        for col, (d, m, lab, rr) in enumerate([(D, M, "raw", rho[0, c]), (Fd, Fm, "filtered", rho[1, c])]):
            a = axs[r, col]
            a.plot(tm, norm(d[:, i, j]).real, color="tab:blue", lw=1, label="data")
            a.plot(tm, norm(m[:, i, j]).real, color="k", lw=1, ls="--", label="Sun model")
            a.set_title(f"{freqs[c]:.1f} MHz, V({ants[i]},{ants[j]}), {lab}: correlation {rr:.2f}", loc="left", fontsize=9)
            a.set_yticks([])
    axs[0, 0].legend(loc="upper right", fontsize=8)
    for a in axs[-1]:
        a.set_xlabel("time [min]")
    fig.suptitle("Real part of the visibility (mean removed, normalized) vs the Sun fringe model", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}_examples.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
