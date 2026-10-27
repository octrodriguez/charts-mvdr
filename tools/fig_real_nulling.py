#!/usr/bin/env python
"""Report figure of the real-data nulling test from several 17_real_nulling outputs.

Each input is LABEL=PATH/real_nulling.npz (for instance the same window with
loading relative to the median and to the smallest eigenvalue). Top: median over
frames of the MVDR/DFT power ratio per channel; bottom: WNG = N ||w||^2 (0 dB for
DFT, higher is worse). Channels whose gain was interpolated (flagged in the
calibration) are drawn hollow. Shaded bands: --rfi (red) and --clean (green)
ranges, also summarized per run on stdout (median, p10 of the ratio, median WNG).

Example:
  python tools/fig_real_nulling.py "median x10=RUN_A/real_nulling.npz" \\
      "min x0.1=RUN_B/real_nulling.npz" --rfi 362-380 415-428 --clean 381.5-399.5 403-414 \\
      --out fig_real_nulling.png
"""
from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_ranges(items):
    return [tuple(float(x) for x in item.split("-")) for item in items or []]


def load_run(spec):
    if "=" not in spec:
        raise ValueError(f"expected LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    z = np.load(path)
    return label, {k: z[k] for k in ("freqs", "ratio_db", "wng_db", "interp")}


def summarize(label, run, ranges):
    lines = []
    for lo, hi in ranges:
        m = (run["freqs"] >= lo) & (run["freqs"] < hi)
        r, w = run["ratio_db"][m], run["wng_db"][m]
        lines.append(f"  {label:<14} {lo:g}-{hi:g} MHz: n={m.sum():3d}  MVDR/DFT median {np.nanmedian(r):+6.1f} dB, "
                     f"p10 {np.nanpercentile(r, 10):+6.1f} dB, WNG median {np.nanmedian(w):+5.1f} dB")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="LABEL=PATH/real_nulling.npz")
    p.add_argument("--rfi", nargs="*", default=[])
    p.add_argument("--clean", nargs="*", default=[])
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    runs = [load_run(s) for s in args.runs]
    rfi, clean = parse_ranges(args.rfi), parse_ranges(args.clean)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for axi in ax:
        for lo, hi in rfi:
            axi.axvspan(lo, hi, color="tab:red", alpha=0.10, lw=0)
        for lo, hi in clean:
            axi.axvspan(lo, hi, color="tab:green", alpha=0.10, lw=0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, run) in enumerate(runs):
        c, f, it = colors[i % len(colors)], run["freqs"], run["interp"].astype(bool)
        for axi, key in ((ax[0], "ratio_db"), (ax[1], "wng_db")):
            axi.plot(f[~it], run[key][~it], "o", ms=2.5, color=c, label=label if axi is ax[0] else None)
            axi.plot(f[it], run[key][it], "o", ms=2.5, mfc="none", mec=c, mew=0.6)
        print(f"{label}:")
        print("\n".join(summarize(label, run, rfi + clean)))
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].set_ylabel("MVDR / DFT power [dB]\n(median over frames)")
    ax[0].legend(fontsize=8, loc="lower left", title="loading (hollow: interpolated gain)", title_fontsize=8)
    if args.title:
        ax[0].set_title(args.title, fontsize=9, loc="left")
    ax[1].set_ylabel("WNG [dB]")
    ax[1].set_xlabel("Frequency [MHz]")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
