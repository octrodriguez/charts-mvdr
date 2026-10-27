#!/usr/bin/env python3
"""
16_rfi_census.py — RFI census of real autocorrelations (complement of Fig. 3).

Two subcommands, so that the heavy I/O runs where the data live and the analysis
runs anywhere:

extract  Reads GPU correlation frames with charts-utils (logical antenna order,
         V_ab = x_a conj(x_b)), keeps every --stride-th frame inside an optional
         time window, and saves the full small cube of the selected antennas:
         vis (M, M, n_freq, n_time) complex64, times, freqs, ants. Frames are
         read in chunks with progress lines, so it can run under nohup.

census   From that cube: per antenna, spectral peaks over a running median
         (charts_mvdr.rfi_census), occupancy over time, classes
         persistent / intermittent / variable / clean, where "variable" comes
         from charts_utils.cal.algorithms.mad_rfi_zapper applied to the
         autocorrelations normalized per channel. With --fringe (npz of
         11_fringe_coherence.py) the array class of each channel is crossed with
         the fringe class (majority over pairs). Outputs rfi_census.npz and
         rfi_census.png in --out.

Usage:
    python 16_rfi_census.py extract --input <frame_dir> --antennas 0,4,5,6 \\
        --stride 4 --out <vis.npz>
    python 16_rfi_census.py census --vis <vis.npz> --fringe <fringe_coherence.npz> \\
        --out <out_dir>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from charts_mvdr import rfi_census as rc

FRINGE_COLORS = {"source": "tab:green", "conjugate": "tab:olive", "stationary": "tab:red",
                 "ambiguous": "tab:gray", "none": "white"}
CLASS_COLORS = {"clean": "white", "variable": "tab:purple", "intermittent": "tab:orange",
                "persistent": "tab:red"}


def extract_cube(groups, times, ants, n_freq, loader, stride=1, chunk=200, log=print):
    """Load every `stride`-th group with `loader(groups, ants, n_freq)` in chunks.

    loader returns (M, M, n_freq, len(groups)) complex. Returns (vis, times).
    """
    sel = np.arange(0, len(groups), stride)
    out = np.zeros((len(ants), len(ants), n_freq, sel.size), dtype=np.complex64)
    t0 = time.time()
    for s in range(0, sel.size, chunk):
        idx = sel[s:s + chunk]
        out[..., s:s + idx.size] = loader([groups[i] for i in idx], ants, n_freq)
        log(f"  {s + idx.size}/{sel.size} frames ({time.time() - t0:.0f} s)")
    return out, np.asarray(times, dtype=float)[sel]


def cmd_extract(args) -> int:
    from charts_constants import FREQ
    from charts_utils.cal.coordinates import parse_time_arg
    from charts_utils.core.gpu_frames import discover_gpu_corr_frame_groups, read_gpu_frame_timestamps
    from charts_utils.core.io import load_gpu_triangular_series

    ants = [int(a) for a in args.antennas.split(",")]
    freqs = np.asarray(FREQ, dtype=float)
    groups = discover_gpu_corr_frame_groups(args.input, skip_invalid_frames=True)
    ts = read_gpu_frame_timestamps(groups)
    if ts is None:
        print("error: frames carry no valid FPGA timestamps", file=sys.stderr)
        return 1
    ts = np.asarray(ts, dtype=float)
    keep = np.ones(ts.size, dtype=bool)
    if args.time_start:
        keep &= ts >= parse_time_arg(args.time_start)
    if args.time_stop:
        keep &= ts <= parse_time_arg(args.time_stop)
    idx = np.where(keep)[0]
    if idx.size == 0:
        print("error: no frames in the window", file=sys.stderr)
        return 1
    print(f"{len(groups)} frames found, {idx.size} in the window, stride {args.stride}", flush=True)

    def loader(g, a, nf):
        return load_gpu_triangular_series(g, a, nf, args.num_antennas)

    vis, tsel = extract_cube([groups[i] for i in idx], ts[idx], ants, len(freqs), loader,
                             stride=args.stride, chunk=args.chunk,
                             log=lambda m: print(m, flush=True))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, vis=vis, times=tsel, freqs=freqs, ants=np.array(ants),
             stride=args.stride, input=os.path.abspath(args.input))
    print(f"saved {args.out}: vis {vis.shape}", flush=True)
    return 0


def variability_mask(auto_norm, valid, sigma):
    """Channels flagged by charts-utils mad_rfi_zapper on per-channel normalized autos."""
    from charts_utils.cal.algorithms import mad_rfi_zapper

    a = auto_norm[:, valid]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = a / np.nanmedian(a, axis=1, keepdims=True)
    mask = np.zeros(auto_norm.shape[0], dtype=bool)
    mask[mad_rfi_zapper(a, threshold_sigma=sigma)] = True
    return mask


def census(vis, freqs, ants, window, k, persistent, intermittent, var_sigma=None):
    """Per-antenna census. Returns dict with occ, cls (M, F), valid (M, T), median spectra."""
    m = len(ants)
    occ = np.full((m, len(freqs)), np.nan)
    cls = np.zeros((m, len(freqs)), dtype=int)
    valid = np.zeros((m, vis.shape[-1]), dtype=bool)
    spec = np.full((m, len(freqs)), np.nan)
    for i in range(m):
        a_norm, valid[i] = rc.normalize_frames(vis[i, i].real)
        _, flags = rc.spectral_peaks(a_norm, window=window, k=k)
        occ[i] = rc.occupancy(flags, valid[i])
        var = variability_mask(a_norm, valid[i], var_sigma) if var_sigma is not None else None
        cls[i] = rc.classify(occ[i], var, persistent=persistent, intermittent=intermittent)
        spec[i] = np.nanmedian(a_norm[:, valid[i]], axis=1)
    return {"occ": occ, "cls": cls, "valid": valid, "spec": spec, "array_cls": rc.array_class(cls)}


def plot(path, freqs, ants, res, fringe_major=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    m = len(ants)
    fig, axes = plt.subplots(m + 1, 1, figsize=(11, 1.9 * m + 1.6), sharex=True,
                             gridspec_kw={"height_ratios": [3] * m + [1]})
    for i in range(m):
        ax = axes[i]
        ax.plot(freqs, 10 * np.log10(res["spec"][i]), lw=0.7, color="k")
        for c, name in enumerate(rc.CLASSES[1:], start=1):
            sel = res["cls"][i] == c
            ax.scatter(freqs[sel], 10 * np.log10(res["spec"][i][sel]), s=6,
                       color=CLASS_COLORS[name], label=name if i == 0 else None, zorder=3)
        lost = np.mean(res["cls"][i] >= 2)
        ax.set_ylabel(f"ant {ants[i]}\n[dB]")
        ax.set_title(f"antenna {ants[i]}: band fraction with peaks (persistent + intermittent) "
                     f"{lost:.1%}, persistent {np.mean(res['cls'][i] == 3):.1%}",
                     fontsize=8, loc="left")
    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    ax = axes[-1]
    rows = [np.array([rc.CLASSES.index(rc.CLASSES[c]) for c in res["array_cls"]])]
    cmap = ListedColormap([CLASS_COLORS[c] for c in rc.CLASSES])
    df = np.median(np.diff(freqs))
    ext = [freqs[0] - df / 2, freqs[-1] + df / 2]
    ax.imshow(rows[0][None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(rc.CLASSES) - 1,
              extent=ext + [1, 2], interpolation="nearest")
    yt, yl = [1.5], ["array"]
    if fringe_major is not None:
        names = list(FRINGE_COLORS)
        codes = np.array([names.index(x) if x in names else names.index("none") for x in fringe_major])
        ax.imshow(codes[None, :], aspect="auto", cmap=ListedColormap(list(FRINGE_COLORS.values())),
                  vmin=0, vmax=len(names) - 1, extent=ext + [0, 1], interpolation="nearest")
        yt, yl = [0.5, 1.5], ["fringe", "array"]
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=v, label=k) for k, v in FRINGE_COLORS.items() if k != "none"],
                  fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.9))
    ax.set_ylim(0 if fringe_major is not None else 1, 2)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_xlabel("Frequency [MHz]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_census(args) -> int:
    z = np.load(args.vis)
    vis, freqs, ants = z["vis"], z["freqs"], [int(a) for a in z["ants"]]
    res = census(vis, freqs, ants, args.window, args.k, args.persistent, args.intermittent,
                 None if args.no_variability else args.var_sigma)
    print(f"{vis.shape[-1]} frames; valid per antenna: "
          + ", ".join(f"{a}: {int(v.sum())}" for a, v in zip(ants, res["valid"])))
    print("band fraction per antenna (persistent / intermittent / variable):")
    for a, c in zip(ants, res["cls"]):
        print(f"  ant {a}: {np.mean(c == 3):.3f} / {np.mean(c == 2):.3f} / {np.mean(c == 1):.3f}")

    fringe_major, table, names = None, None, None
    if args.fringe:
        fz = np.load(args.fringe, allow_pickle=True)
        if len(fz["freqs"]) != len(freqs):
            print(f"error: fringe file has {len(fz['freqs'])} channels, census {len(freqs)}", file=sys.stderr)
            return 1
        fringe_major = rc.fringe_majority(fz["labels"])
        names = list(FRINGE_COLORS)
        table = rc.cross_counts(res["array_cls"], fringe_major, names)
        print("\narray class (rows) x fringe class (columns), channels:")
        print(" " * 14 + "".join(f"{n:>12}" for n in names))
        for i, c in enumerate(rc.CLASSES):
            print(f"{c:>14}" + "".join(f"{v:>12d}" for v in table[i]))

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "rfi_census.npz"), freqs=freqs, ants=np.array(ants),
             classes=np.array(rc.CLASSES), occ=res["occ"], cls=res["cls"], valid=res["valid"],
             spec=res["spec"], array_cls=res["array_cls"],
             fringe_major=fringe_major if fringe_major is not None else np.array([]),
             fringe_names=np.array(names or []), table=table if table is not None else np.array([]),
             params=np.array([args.window, args.k, args.persistent, args.intermittent]))
    plot(os.path.join(args.out, "rfi_census.png"), freqs, ants, res, fringe_major)
    print(f"\nsaved {args.out}/rfi_census.npz and rfi_census.png")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RFI census of real autocorrelations.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="Reduced visibility cube from GPU frames.")
    e.add_argument("--input", required=True, help="Directory with GPU correlation frames.")
    e.add_argument("--antennas", required=True, help="Comma-separated logical antenna indices.")
    e.add_argument("--num-antennas", type=int, default=None, help="Physical antenna count (inferred if omitted).")
    e.add_argument("--stride", type=int, default=4, help="Keep every n-th frame.")
    e.add_argument("--chunk", type=int, default=200, help="Frames per read chunk (progress line each).")
    e.add_argument("--time-start", default=None, help="Unix seconds or ISO UTC.")
    e.add_argument("--time-stop", default=None, help="Unix seconds or ISO UTC.")
    e.add_argument("--out", required=True, help="Output .npz.")
    c = sub.add_parser("census", help="Census from a cube saved by extract.")
    c.add_argument("--vis", required=True, help=".npz from extract.")
    c.add_argument("--fringe", default=None, help="fringe_coherence.npz from 11_fringe_coherence.py.")
    c.add_argument("--window", type=int, default=31, help="Running-median window [channels], odd.")
    c.add_argument("--k", type=float, default=5.0, help="Peak threshold in robust sigmas.")
    c.add_argument("--persistent", type=float, default=0.5, help="Occupancy for persistent.")
    c.add_argument("--intermittent", type=float, default=0.05, help="Occupancy for intermittent.")
    c.add_argument("--var-sigma", type=float, default=3.0, help="threshold_sigma of mad_rfi_zapper.")
    c.add_argument("--no-variability", action="store_true", help="Skip the mad_rfi_zapper criterion.")
    c.add_argument("--out", required=True, help="Output directory.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_extract(args) if args.cmd == "extract" else cmd_census(args)


if __name__ == "__main__":
    sys.exit(main())
