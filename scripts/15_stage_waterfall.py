#!/usr/bin/env python
"""Dynamic spectra and dedispersed series of a burst from the MVDR stage (Figs. 1-2 analogue).

Reads the stage output of a window with RFI (--rfi) and of its paired window without RFI
(--clean): npz from 13_compare_kotekan_mvdr.py (Kotekan) or 12_stage_reference.py, with
power (2, n_frames, n_chan, n_beam), planes DFT and MVDR. Three columns are drawn:
clean window with DFT weights (reference), RFI window with DFT weights, RFI window with the
MVDR weights of the stage.

Per channel the power is normalized with the off-local frames of the burst (as in 14):
z = (P - mean_off P) / std_off P. Top row: z(t, f) around the dispersion sweep (frames at
their times, so gaps of the sparse capture show as gaps). Bottom row: dedispersed series,
each channel shifted by its dispersion delay (bins of one frame) and the z values of the
usable channels summed and divided by sqrt(number of channels), i.e. S/N per dedispersed
bin; the clean DFT series is repeated as a dashed line in every panel. Channels whose
leading tail is cut by the simulator, or with too few off frames, are excluded from the
series and hatched in the waterfall.

Outputs in --out: stage_waterfall.png and stage_waterfall.npz (series, peak S/N, z).

Example:
  python scripts/15_stage_waterfall.py --rfi V3.npz --clean V3C.npz --window-dir WINDOW_DIR \\
      --out FIG_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.data_io import load_events_file, load_window_dir_meta
from charts_mvdr.transients import stage_frame_masks

FRAME_S = 1536 * 10e-6 / 3


def normalize(P, off, min_off):
    """z per channel from the off-local frames; NaN where there are too few."""
    z = np.full(P.shape, np.nan)
    for c in range(P.shape[1]):
        o = off[:, c]
        if o.sum() < min_off:
            continue
        s = P[o, c].std(ddof=1)
        if s > 0:
            z[:, c] = (P[:, c] - P[o, c].mean()) / s
    return z


def dedisperse(z, times, t_arr, usable, frame_s=FRAME_S, span_s=0.1):
    """Sum of z over usable channels in bins of dedispersed time t - (t_arr - t_arr_top)."""
    t_top = float(np.min(t_arr[usable])) if usable.any() else 0.0
    edges = np.arange(-span_s, span_s + frame_s, frame_s) + t_top
    acc = np.zeros(edges.size - 1)
    cnt = np.zeros(edges.size - 1)
    for c in np.flatnonzero(usable):
        td = times + frame_s / 2 - (t_arr[c] - t_top)
        idx = np.searchsorted(edges, td) - 1
        ok = (idx >= 0) & (idx < acc.size) & np.isfinite(z[:, c])
        np.add.at(acc, idx[ok], z[ok, c])
        np.add.at(cnt, idx[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = acc / np.sqrt(cnt)
    return 0.5 * (edges[:-1] + edges[1:]) - t_top, snr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rfi", required=True, type=Path)
    ap.add_argument("--clean", required=True, type=Path)
    ap.add_argument("--window-dir", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--local-s", type=float, default=0.05)
    ap.add_argument("--guard-frames", type=int, default=1)
    ap.add_argument("--min-off", type=int, default=4)
    ap.add_argument("--margin-s", type=float, default=0.05,
                    help="time shown before the first and after the last arrival")
    ap.add_argument("--vmax", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=Path("stage_waterfall"))
    args = ap.parse_args(argv)

    zr, zc = np.load(args.rfi), np.load(args.clean)
    if zr["power"].shape != zc["power"].shape:
        raise SystemExit("the two windows must have the same frames, channels and beams")
    freqs = np.asarray(zr["freqs_mhz"], dtype=float)
    meta = load_window_dir_meta(args.window_dir)
    events = load_events_file(args.window_dir)
    n_frames = zr["power"].shape[1]
    times, on, off, trunc, t_arr, burst = stage_frame_masks(
        meta, events, n_frames, freqs, args.n_sigma, args.local_s, args.guard_frames, FRAME_S)
    usable = ~trunc & (off.sum(0) >= args.min_off) & (on.sum(0) > 0)

    panels = [("clean window, DFT", zc["power"][0, :, :, args.beam]),
              ("RFI window, DFT", zr["power"][0, :, :, args.beam]),
              ("RFI window, MVDR (stage)", zr["power"][1, :, :, args.beam])]
    res = []
    for name, P in panels:
        z = normalize(P, off, args.min_off)
        tau, snr = dedisperse(z, times, t_arr, usable)
        res.append({"name": name, "z": z, "tau": tau, "snr": snr, "peak": float(np.nanmax(snr))})

    ref_peak = res[0]["peak"]
    print(f"burst DM {burst['dm']:g}, sigma {burst['width_s'] * 1e3:g} ms; {usable.sum()} of "
          f"{freqs.size} channels in the dedispersed series")
    for r in res:
        print(f"  {r['name']:<26} dedispersed peak S/N {r['peak']:7.1f}  "
              f"({10 * np.log10(r['peak'] / ref_peak):+.2f} dB vs clean DFT)")

    t0 = float(np.min(t_arr)) - args.margin_s
    t1 = float(np.max(t_arr)) + args.margin_s
    # regular grid of one frame per column; columns without a frame (sparse capture) stay NaN
    n_col = int(np.ceil((t1 - t0) / FRAME_S))
    col = np.floor((times - t0) / FRAME_S).astype(int)
    in_view = (col >= 0) & (col < n_col)
    edges_t = t0 + FRAME_S * np.arange(n_col + 1)
    df = np.median(np.diff(freqs)) if freqs.size > 1 else 0.3
    edges_f = np.concatenate([freqs - df / 2, [freqs[-1] + df / 2]])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="row",
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.4})
    for j, r in enumerate(res):
        ax = axes[0, j]
        grid = np.full((n_col, freqs.size), np.nan)
        grid[col[in_view]] = r["z"][in_view]
        im = ax.pcolormesh(edges_t, edges_f, grid.T, cmap="viridis", vmin=-args.vmax / 2,
                           vmax=args.vmax, shading="flat")
        bad = ~usable
        if bad.any():
            ax.barh(freqs[bad], t1 - t0, left=t0, height=df, color="none", edgecolor="0.6",
                    hatch="///", linewidth=0)
        ax.set_xlim(t0, t1)
        ax.set_title(r["name"], fontsize=10)
        ax.set_xlabel("time [s]")
        if j == 0:
            ax.set_ylabel("frequency [MHz]")
        ax2 = axes[1, j]
        ax2.plot(r["tau"] * 1e3, r["snr"], drawstyle="steps-mid", color="C0")
        if j > 0:
            ax2.plot(res[0]["tau"] * 1e3, res[0]["snr"], drawstyle="steps-mid", color="0.5",
                     ls="--", lw=0.8, label="clean DFT")
            ax2.legend(fontsize=8, loc="upper right")
        ax2.set_title(f"dedispersed peak S/N {r['peak']:.1f} "
                      f"({10 * np.log10(r['peak'] / ref_peak):+.2f} dB)", fontsize=9)
        ax2.set_xlabel("dedispersed time [ms]")
        if j == 0:
            ax2.set_ylabel("S/N")
    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, label="z (off-local frames)")
    fig.suptitle(f"Burst DM {burst['dm']:g}, beam {args.beam}, frames of "
                 f"{FRAME_S * 1e3:.2f} ms", fontsize=11)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "stage_waterfall.png", dpi=130)
    np.savez_compressed(args.out / "stage_waterfall.npz", usable=usable, times=times,
                        freqs_mhz=freqs,
                        **{f"{k}_{i}": r[k] for i, r in enumerate(res) for k in ("z", "tau", "snr")},
                        peaks=np.array([r["peak"] for r in res]),
                        names=np.array([r["name"] for r in res]))
    print(f"saved {args.out / 'stage_waterfall.png'}")


if __name__ == "__main__":
    main()
