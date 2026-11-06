#!/usr/bin/env python
"""Visibility phase vs time and frequency for antenna pairs, before and after nulling the
interference inside selected bands (other channels untouched).

Per channel inside --null-bands: whitened eigenmode removal on the time average of the whole
cube (charts_mvdr.nulling.whitened_projector; --k N, or --k auto: eigenvalues above --thresh
times the smallest), applied to
every frame, V' = P V P^H. The projection makes the noise correlated between antennas
(P sigma^2 I P^H = sigma^2 P), a constant bias that dominates the phase of the nulled band
even when the interference is removed perfectly; --demean subtracts each channel's time mean
(in all channels) so that the time-varying fringes show. The nulled fringes of a pair are a
mixture of antennas (P s s^H P^H) and need not look like those of the neighbouring channels;
19_fringe_recovery.py compares them with the model filtered the same way.

Example:
  python tools/fig_phase2d_nulling.py --vis CUBE.npz --gains GAINS.npy --gain-antennas 0,4,32,36 \\
      --pairs 0,4 0,32 --zoom 345-395 --null-bands 357-380 --demean --out phase2d.png
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from charts_mvdr import nulling


def rng(s):
    lo, hi = (float(x) for x in s.split("-"))
    return lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--gains", required=True)
    ap.add_argument("--gain-antennas", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="Pairs a,b of logical antennas.")
    ap.add_argument("--zoom", default=None, help="Frequency range [MHz] shown.")
    ap.add_argument("--null-bands", nargs="+", required=True)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--thresh", type=float, default=2.0)
    ap.add_argument("--demean", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ants = [int(x) for x in a.gain_antennas.split(",")]
    V, freqs, times, _ = nulling.load_cube(a.vis, ants)
    Vc, ok = nulling.calibrate(V, np.load(a.gains), freqs)
    sel = np.ones(len(freqs), bool) if a.zoom is None else (freqs >= rng(a.zoom)[0]) & (freqs <= rng(a.zoom)[1])
    bands = [rng(b) for b in a.null_bands]
    Vn = Vc.copy()
    k_used = []
    for c in np.flatnonzero(ok & sel):
        if not any(lo <= freqs[c] <= hi for lo, hi in bands):
            continue
        Rm = Vc[:, c].mean(axis=0)
        if a.k == "auto":
            P, ku = nulling.whitened_projector(Rm, thresh=a.thresh)
        else:
            P, ku = nulling.whitened_projector(Rm, k=int(a.k))
        k_used.append(int(ku))
        Vn[:, c] = P[None] @ Vc[:, c] @ P.conj().T[None]
    if a.demean:
        Vc = Vc - np.nanmean(Vc, axis=0, keepdims=True)
        Vn = Vn - np.nanmean(Vn, axis=0, keepdims=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    utc = times[0] > 1e9
    tt = times.astype("datetime64[s]") if utc else (times - times[0]) / 60.0
    f = freqs[sel]
    fig, axs = plt.subplots(len(a.pairs), 2, figsize=(14, 3.6 * len(a.pairs)), sharex=True, sharey=True, squeeze=False)
    for r, pr in enumerate(a.pairs):
        i, j = [ants.index(int(x)) for x in pr.split(",")]
        for col, (X, lab) in enumerate([(Vc, "Before nulling"), (Vn, f"Nulled ({', '.join(a.null_bands)} MHz)")]):
            im = axs[r, col].pcolormesh(tt, f, np.angle(X[:, sel, i, j]).T, cmap="twilight",
                                        vmin=-np.pi, vmax=np.pi, shading="auto")
            axs[r, col].set_title(f"{lab}: inputs {ants[i]}-{ants[j]}", loc="left", fontsize=10)
        axs[r, 0].set_ylabel("Frequency (MHz)")
    for ax in axs[-1]:
        ax.set_xlabel("Time (UTC)" if utc else "time [min]")
        if utc:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.colorbar(im, ax=axs, label="Phase (rad)" + (", time mean removed" if a.demean else ""))
    fig.savefig(a.out, dpi=130)
    plt.close(fig)
    if k_used:
        print(f"channels in null bands: {len(k_used)}, modes removed per channel (k = 0, 1, 2): "
              f"{np.bincount(k_used, minlength=3).tolist()}")
    print(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
