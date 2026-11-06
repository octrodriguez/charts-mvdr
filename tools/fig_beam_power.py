#!/usr/bin/env python
"""Beam power without (DFT) and with (MVDR) nulling from a 17_real_nulling.py output.

Panels: median over frames of the DFT and MVDR beam power per channel, in dB relative to the
median DFT power of a clean band (--ref); a zoom on one band; and power vs time, median over
the channels of that band, with the clean band for reference. Absolute levels in channels
whose F-engine digital gains differ from their neighbours are not on a common scale.

Example:
  python tools/fig_beam_power.py RUN/real_nulling.npz --zoom 355-385 --ref 381.5-399.5 \\
      --rfi 357-380 410-428 --out beam_power.png
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def rng(s):
    lo, hi = (float(x) for x in s.split("-"))
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="real_nulling.npz from 17_real_nulling.py")
    ap.add_argument("--zoom", default="355-385")
    ap.add_argument("--ref", default="381.5-399.5")
    ap.add_argument("--rfi", nargs="*", default=[])
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = np.load(a.npz)
    f, p, t = z["freqs"], z["power"], z["times"]
    r0, r1 = rng(a.ref)
    z0, z1 = rng(a.zoom)
    ref = (f >= r0) & (f <= r1)
    zm = (f >= z0) & (f <= z1)
    norm = np.nanmedian(p[0][:, ref])
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(np.nanmedian(p, axis=1) / norm)
        ts = 10 * np.log10(np.nanmedian(p[:, :, zm], axis=2) / norm)
        tr = 10 * np.log10(np.nanmedian(p[:, :, ref], axis=2) / norm)
    tm = (t - t[0]) / 60.0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
    ax, axz, axt = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    for x in (ax, axz):
        for s in a.rfi:
            lo, hi = rng(s)
            x.axvspan(lo, hi, color="tab:red", alpha=0.1, lw=0)
        x.axvspan(r0, r1, color="tab:green", alpha=0.1, lw=0)
        x.plot(f, db[0], ".", ms=3, color="0.4", label="DFT (no nulling)")
        x.plot(f, db[1], ".", ms=3, color="tab:blue", label="MVDR (nulling)")
        x.set_ylabel("beam power [dB rel. clean-band DFT]")
        x.set_xlabel("Frequency [MHz]")
    axz.set_xlim(z0, z1)
    ax.legend(loc="upper right")
    ax.set_title(a.title or "Beam power towards the look direction, median over frames", loc="left", fontsize=10)
    axz.set_title(f"zoom {z0:g}-{z1:g} MHz", loc="left", fontsize=10)
    axt.plot(tm, ts[0], color="0.4", label=f"DFT, {z0:g}-{z1:g} MHz")
    axt.plot(tm, ts[1], color="tab:blue", label=f"MVDR, {z0:g}-{z1:g} MHz")
    axt.plot(tm, tr[0], "--", color="0.6", label=f"DFT, clean {r0:g}-{r1:g} MHz")
    axt.plot(tm, tr[1], "--", color="tab:cyan", label=f"MVDR, clean {r0:g}-{r1:g} MHz")
    axt.set_xlabel("time [min]")
    axt.set_ylabel("median power over channels [dB]")
    axt.set_title("power vs time", loc="left", fontsize=10)
    axt.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    plt.close(fig)
    for lo, hi in [rng(s) for s in a.rfi] + [(r0, r1)]:
        m = (f >= lo) & (f <= hi)
        print(f"{lo:g}-{hi:g} MHz: DFT {np.nanmedian(db[0][m]):+.1f} dB, MVDR {np.nanmedian(db[1][m]):+.1f} dB (rel. clean DFT)")
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
