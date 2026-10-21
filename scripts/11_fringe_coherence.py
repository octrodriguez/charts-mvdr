#!/usr/bin/env python3
"""
11_fringe_coherence.py — Coherence of real visibilities with a source's fringe model.

For each antenna pair, compares the raw visibilities of GPU correlation frames
with the geometric phase of a tracked source (default: the Sun) through the
coherence c(B) of charts_mvdr.fringes, which ignores the constant instrumental
phase of each channel and so needs no gain calibration. Two outputs:

1. Baseline fit per band: the baseline vector (east, north) that maximizes the
   channel-averaged coherence, compared with the layout (positions file),
   its negative (conjugated data) and zero (stationary signal).
2. Channel classification over the whole band: source / conjugate /
   stationary / ambiguous / none, saved to an .npz for an RFI census.

Frames are read with charts-utils (logical antenna order, V_ab = x_a conj(x_b)).

Usage:
    python 11_fringe_coherence.py --input <frame_dir> --antennas 0,4,5,6 \\
        --positions <positions.npy> --time-start 2026-01-01T12:00:00 \\
        --time-stop 2026-01-01T18:00:00 --out <out_dir> \\
        --bands 300-410 431-500 --exclude 360-381.5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from charts_constants import CHARTS_ANTENNA_REF_LOCATION, FREQ
from charts_utils.cal.coordinates import CHARTSArray, geometric_visibility_phase, parse_time_arg
from charts_utils.core.gpu_frames import discover_gpu_corr_frame_groups, read_gpu_frame_timestamps
from charts_utils.core.io import load_gpu_triangular_series

from charts_mvdr import fringes


def parse_ranges(items):
    out = []
    for item in items or []:
        lo, hi = (float(x) for x in item.split("-"))
        if hi <= lo:
            raise ValueError(f"empty range {item!r}")
        out.append((lo, hi))
    return out


def band_channels(freqs, lo, hi, exclude, step):
    m = (freqs >= lo) & (freqs < hi)
    for a, b in exclude:
        m &= ~((freqs >= a) & (freqs < b))
    return np.where(m)[0][::step]


def bin_time(vis, t, n):
    """Average n consecutive integrations: vis (..., time) -> (..., time // n)."""
    nt = (vis.shape[-1] // n) * n
    v = vis[..., :nt].reshape(vis.shape[:-1] + (nt // n, n)).mean(-1)
    return v, t[:nt].reshape(-1, n).mean(1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Directory with GPU correlation frames.")
    p.add_argument("--antennas", required=True, help="Comma-separated logical antenna indices.")
    p.add_argument("--positions", required=True,
                   help="Antenna positions .npy, shape (n_antennas, 3), ENU metres, logical order.")
    p.add_argument("--num-antennas", type=int, default=None, help="Physical antenna count (inferred if omitted).")
    p.add_argument("--time-start", required=True, help="Unix seconds or ISO UTC.")
    p.add_argument("--time-stop", required=True, help="Unix seconds or ISO UTC.")
    p.add_argument("--target", default="sun", help="Source name known to charts-utils.")
    p.add_argument("--bin-frames", type=int, default=12, help="Integrations averaged per time bin.")
    p.add_argument("--bands", nargs="+", default=["300-410", "431-501.6"],
                   help="MHz ranges for the baseline fit.")
    p.add_argument("--exclude", nargs="*", default=[], help="MHz ranges left out of the baseline fit.")
    p.add_argument("--chan-step", type=int, default=2, help="Use every n-th channel in the baseline fit.")
    p.add_argument("--extent", type=float, nargs=2, default=[6.0, 3.0],
                   help="Half-width of the baseline search box, east and north [m].")
    p.add_argument("--min-coherence", type=float, default=0.3)
    p.add_argument("--max-overlap", type=float, default=0.5)
    p.add_argument("--out", required=True, help="Output directory.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    ants = [int(a) for a in args.antennas.split(",")]
    freqs = np.asarray(FREQ, dtype=float)
    exclude = parse_ranges(args.exclude)
    bands = parse_ranges(args.bands)

    groups = discover_gpu_corr_frame_groups(args.input, skip_invalid_frames=True)
    ts = read_gpu_frame_timestamps(groups)
    if ts is None:
        print("error: frames carry no valid FPGA timestamps", file=sys.stderr)
        return 1
    t0, t1 = parse_time_arg(args.time_start), parse_time_arg(args.time_stop)
    keep = np.where((ts >= t0) & (ts <= t1))[0]
    if keep.size < 2 * args.bin_frames:
        print(f"error: only {keep.size} frames in the window", file=sys.stderr)
        return 1
    print(f"Loading {keep.size}/{len(groups)} frames, antennas {ants} ...", flush=True)
    cube = load_gpu_triangular_series([groups[i] for i in keep], ants, len(freqs), args.num_antennas)
    vis, tb = bin_time(cube, np.asarray(ts)[keep], args.bin_frames)      # (M, M, freq, tbin)
    print(f"{vis.shape[-1]} time bins of {args.bin_frames} integrations", flush=True)

    pos_all = np.load(args.positions)
    if pos_all.ndim != 2 or pos_all.shape[1] != 3 or pos_all.shape[0] <= max(ants):
        print(f"error: positions must have shape (n >= {max(ants) + 1}, 3); got {pos_all.shape}", file=sys.stderr)
        return 1
    pos = pos_all[ants]
    lat, lon, alt = CHARTS_ANTENNA_REF_LOCATION
    unit = CHARTSArray(lat=lat, lon=lon, alt=alt,
                       antenna_positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float))
    phase_unit = geometric_visibility_phase(tb, freqs, unit, [1, 2], [0, 0], src_name=args.target)
    phase_unit = np.asarray(phase_unit, dtype=float)                    # (freq, 2, tbin)

    pairs = [(i, j) for i in range(len(ants)) for j in range(i + 1, len(ants))]
    layout = np.array([pos[i, :2] - pos[j, :2] for i, j in pairs])
    c_all = np.full((len(pairs), 3, len(freqs)), np.nan)
    labels = np.empty((len(pairs), len(freqs)), dtype="<U10")
    fits = np.full((len(pairs), len(bands), 3), np.nan)                  # B_e, B_n, c

    for b, (lo, hi) in enumerate(bands):
        ch = band_channels(freqs, lo, hi, exclude, args.chan_step)
        print(f"\n== band {lo:g}-{hi:g} MHz: {len(ch)} channels in the fit")
        for p, (i, j) in enumerate(pairs):
            v = vis[i, j][ch].T                                          # (tbin, chan)
            best, c = fringes.fit_baseline(v, phase_unit[ch], extent=args.extent)
            fits[p, b] = (best[0], best[1], c)
            c3 = np.nanmean(fringes.coherence(v, phase_unit[ch], [layout[p], -layout[p], (0.0, 0.0)]), axis=1)
            print(f"   ({ants[i]},{ants[j]}): best B = ({best[0]:+.2f}, {best[1]:+.2f}) m, c = {c:.3f} | "
                  f"layout ({layout[p, 0]:+.2f}, {layout[p, 1]:+.2f}): c = {c3[0]:.3f}, "
                  f"-layout c = {c3[1]:.3f}, B=0 c = {c3[2]:.3f}")

    print("\n== channel classification (all channels)")
    for p, (i, j) in enumerate(pairs):
        lab, c = fringes.classify_channels(vis[i, j].T, phase_unit, layout[p],
                                           min_coherence=args.min_coherence, max_overlap=args.max_overlap)
        labels[p], c_all[p] = lab, c
        counts = {k: int((lab == k).sum()) for k in fringes.LABELS}
        print(f"   ({ants[i]},{ants[j]}): " + ", ".join(f"{k} {n}" for k, n in counts.items()))

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "fringe_coherence.npz")
    np.savez(out, freqs=freqs, ants=ants, pairs=np.array(pairs), layout=layout, coherence=c_all,
             labels=labels, bands=np.array(bands), fits=fits, times=tb, target=args.target)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
