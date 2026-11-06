#!/usr/bin/env python
"""Per-antenna power before and after nulling interference eigenmodes (BURSTT-style figure),
and the matrices to apply to raw voltages.

Voltage domain: R = v v^H = conj(V) for a cube V saved by 16_rfi_census.py extract;
calibrated voltages v_cal = G^-1 v, G = diag(conj(g)) for charts-utils gains g. Per channel,
on the time-averaged calibrated covariance, the interference eigenmodes of the
noise-whitened matrix are removed (charts_mvdr.nulling.whitened_projector), k = 1 and k = 2
("nulling-1", "nulling-2"); with --auto T only modes whose eigenvalue exceeds T times the
smallest one are removed (up to k), so clean channels are left as they are. The matrix for
raw voltages is M_k = P_k G^-1 (v' = M_k v): the instrumental gains with k interference
modes suppressed, a full N x N matrix per channel.

In noise-only channels removing k of N modes lowers each antenna's power by about
10 log10((N - k)/N) (-1.2 and -3.0 dB for N = 4). Absolute levels in channels where the
F-engine digital gains differ from their neighbours are not on a common scale unless those
gains are removed first.

Outputs: OUT.png (one panel per antenna; dB relative to the median original power in
--ref) and OUT.npz (freqs, ants, M (2, n_chan, N, N), k_used (2, n_chan)).
Example:
  python scripts/20_nulling_gains.py --vis CUBE.npz --gains GAINS.npy --gain-antennas 0,4,32,36 \\
      --auto 2 --rfi 357-380 410-428 --out OUT/nulling_gains
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from charts_mvdr import nulling


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--gains", required=True)
    ap.add_argument("--gain-antennas", required=True)
    ap.add_argument("--ref", default="381.5-399.5", help="Clean band [MHz] for the dB reference.")
    ap.add_argument("--rfi", nargs="*", default=[], help="MHz ranges shaded and summarized.")
    ap.add_argument("--auto", type=float, default=0.0,
                    help="Remove only modes whose eigenvalue exceeds AUTO x the smallest (whitened).")
    ap.add_argument("--no-whiten", action="store_true")
    ap.add_argument("--out", required=True, help="Output prefix.")
    return ap.parse_args(argv)


def rng(s):
    lo, hi = (float(x) for x in s.split("-"))
    return lo, hi


def main(argv=None) -> int:
    args = parse_args(argv)
    ants = [int(a) for a in args.gain_antennas.split(",")]
    V, freqs, _, _ = nulling.load_cube(args.vis, ants)
    g = np.load(args.gains)
    if g.shape != (len(freqs), len(ants)):
        print(f"error: gains {g.shape} do not match ({len(freqs)}, {len(ants)})", file=sys.stderr)
        return 1
    Vc, ok = nulling.calibrate(V, g, freqs)
    from charts_mvdr import tracking

    gi, _ = tracking.interpolate_gains(g, freqs)
    Rc = np.conj(np.nanmean(Vc, axis=0))                     # calibrated voltage covariance, time average
    n_c, n = Rc.shape[0], Rc.shape[1]
    pw = np.full((3, n_c, n), np.nan)
    M = np.full((2, n_c, n, n), np.nan, dtype=np.complex64)
    k_used = np.zeros((2, n_c), dtype=int)
    for c in np.flatnonzero(ok):
        pw[0, c] = np.real(np.diag(Rc[c]))
        for k in (1, 2):
            if args.auto > 0:
                P, ku = nulling.whitened_projector(Rc[c], thresh=args.auto, whiten="none" if args.no_whiten else "noise", kmax=k)
            else:
                P, ku = nulling.whitened_projector(Rc[c], k=k, whiten="none" if args.no_whiten else "noise")
            k_used[k - 1, c] = int(ku)
            pw[k, c] = np.real(np.diag(P @ Rc[c] @ P.conj().T))
            M[k - 1, c] = nulling.voltage_nulling_matrix(P, gi[c]).astype(np.complex64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{out}.npz", freqs=freqs, ants=np.array(ants), M=M, k_used=k_used,
                        note="v' = M[k-1, c] @ v on raw voltages, logical antenna order; M = P_k G^-1, G = diag(conj(g))")
    plot(out, freqs, pw, ants, args)
    r0, r1 = rng(args.ref)
    for b in args.rfi + [args.ref]:
        lo, hi = rng(b)
        m = (freqs >= lo) & (freqs <= hi)
        with np.errstate(invalid="ignore", divide="ignore"):
            d1 = 10 * np.log10(np.nanmedian(pw[1, m] / pw[0, m]))
            d2 = 10 * np.log10(np.nanmedian(pw[2, m] / pw[0, m]))
        print(f"{b} MHz: change vs original (median over channels and antennas): nulling-1 {d1:+.1f} dB, "
              f"nulling-2 {d2:+.1f} dB; channels with modes removed {np.mean(k_used[0, m] > 0):.0%} / "
              f"{np.mean(k_used[1, m] > 0):.0%}")
    print(f"saved {out}.png and {out}.npz")
    return 0


def plot(out, freqs, pw, ants, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r0, r1 = rng(args.ref)
    refm = (freqs >= r0) & (freqs <= r1)
    n = len(ants)
    fig, axs = plt.subplots(2, (n + 1) // 2, figsize=(12, 7), sharex=True, sharey=True, squeeze=False)
    for i, ax in enumerate(axs.flat[:n]):
        norm = np.nanmedian(pw[0, refm, i])
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, (lab, st) in enumerate([("original", "-"), ("nulling-1", "--"), ("nulling-2", "-.")]):
                ax.plot(freqs, 10 * np.log10(pw[k, :, i] / norm), st, lw=1, label=lab)
        for b in args.rfi:
            lo, hi = rng(b)
            ax.axvspan(lo, hi, color="tab:red", alpha=0.08, lw=0)
        ax.set_title(f"input {ants[i]}", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("Frequency [MHz]")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"power [dB rel. median {args.ref} MHz]")
    title = "Calibrated power per antenna before and after nulling interference eigenmodes"
    title += "" if args.no_whiten else " (noise-whitened)"
    title += f", only eigenvalues > {args.auto:g}" if args.auto > 0 else ""
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
