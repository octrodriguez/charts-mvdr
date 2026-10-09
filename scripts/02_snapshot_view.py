#!/usr/bin/env python3
"""
02_snapshot_view.py — Visualize single-snapshot correlation datasets.

Designed for the Fernando snapshot dataset:
    /scratch/octo/charts-simulations/16_09_2026Fernando_corr/
    Files: corr_<source>_<condition>_0000000.bin

Verified byte-level format (Astron blocked, prefixed header):
    [4 B]         uint32 LE metadata_size
    [49,848 B]    chordMetadata
    [5,677,056 B] payload
    Payload: 336 freq x 528 baselines x 2x2 pols x int32 (I, Q)
    Load with: load_gpu_visibilities(path, n_freq=336, num_antennas=64,
                                     frame_format="prefixed")
    Returns: (336, 2080) complex64

Three view modes:
    baseline : |V(f)| and arg(V(f)) for one antenna pair
    autos    : all 64 autocorrelations vs frequency
    heatmap  : 2D |V| over (baseline, frequency)

Usage:
    python 02_snapshot_view.py --input <dir> --source sun --mode all
    python 02_snapshot_view.py --input <dir> --source frb --mode baseline \
        --baseline "0,4" --compare
    python 02_snapshot_view.py --input <dir> --source frb --mode autos \
        --condition with_noise

IMPORTANT: the snapshots contain 336 channels = HALF of the 672-channel
CHARTS band. Unless Fernando confirms otherwise, the default assumption is
the LOWER half (300-400.8 MHz). Override --freq-start if the file contains
the upper half (400.8-501.6 MHz).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_utils.core.io import load_gpu_visibilities
from charts_utils.core.kotekan import KOTEKAN_GPU_N_FREQ
from charts_utils.core.visibilities import visibility_product_index


# ---------------------------------------------------------------------------
# Constants from the F-engine overview and events.json
# ---------------------------------------------------------------------------

BAND_START_MHZ = 300.0
CHANNEL_WIDTH_MHZ = 0.3          # from F-engine overview (channel width 300 kHz)
DEFAULT_RFI_CHANNELS = [94, 133, 147, 450]


def global_channel_to_mhz(ch: int) -> float:
    """Absolute frequency (MHz) of a global CHARTS channel index (0-671)."""
    return BAND_START_MHZ + ch * CHANNEL_WIDTH_MHZ


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize Fernando snapshot correlations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Directory with corr_<source>_<condition>_*.bin files.")
    p.add_argument("--source", required=True,
                   help="Source name (e.g. sun, frb, pictor_a).")
    p.add_argument("--mode", default="all",
                   choices=["baseline", "autos", "heatmap", "all"])
    p.add_argument("--baseline", default="0,4",
                   help="Baseline for baseline mode, format 'a,b'.")
    p.add_argument("--n-freq", type=int, default=336,
                   help="Channels per snapshot (NOT 672).")
    p.add_argument("--num-antennas", type=int, default=64)
    p.add_argument("--frame-format", default="prefixed",
                   choices=["prefixed", "headerless", "auto"])
    p.add_argument("--freq-start", type=float, default=BAND_START_MHZ,
                   help="Frequency of channel 0 in the snapshot (MHz). "
                        "Default assumes the lower half of the band.")
    p.add_argument("--compare", action="store_true",
                   help="Plot all conditions of the source side by side. "
                        "Without this flag, only one condition is plotted "
                        "(see --condition).")
    p.add_argument("--condition", default=None,
                   help="Single condition to plot when --compare is not set "
                        "(e.g. no_noise, with_noise, saturated). "
                        "Default: with_noise if present, else no_noise, "
                        "else alphabetically first.")
    p.add_argument("--heatmap-block", type=int, default=10,
                   help="Baseline aggregation block for the heatmap mode.")
    p.add_argument("--output", default="./outputs/snapshots")
    p.add_argument("--dpi", type=int, default=150)
    return p


# ---------------------------------------------------------------------------
# File discovery and helpers
# ---------------------------------------------------------------------------

def find_source_files(directory: str, source: str) -> dict[str, str]:
    """Return {condition: path} for every corr_<source>_<condition>_*.bin."""
    pattern = re.compile(rf"^corr_{re.escape(source)}_(.+?)_\d+\.bin$")
    result: dict[str, str] = {}
    for p in sorted(Path(directory).glob(f"corr_{source}_*.bin")):
        m = pattern.match(p.name)
        if m:
            result[m.group(1)] = str(p)
    return result


def load_rfi_channels(directory: str) -> list[int]:
    """Load RFI channel indices from any *events.json in the directory."""
    for p in sorted(Path(directory).glob("*events.json")):
        try:
            with open(p) as f:
                events = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(events, list):
            continue
        chans: set[int] = set()
        for ev in events:
            if not isinstance(ev, dict):
                continue
            if str(ev.get("event_type", "")).startswith("rfi"):
                channels = ev.get("channels")
                if channels:
                    chans.update(int(c) for c in channels)
        if chans:
            print(f"Loaded {len(chans)} RFI channels from {p.name}: {sorted(chans)}")
            return sorted(chans)
    print(f"No events.json found; using default RFI channels "
          f"{DEFAULT_RFI_CHANNELS}.")
    return list(DEFAULT_RFI_CHANNELS)


def parse_baseline(spec: str, n_ant: int) -> tuple[int, int]:
    try:
        a, b = [int(x.strip()) for x in spec.split(",")]
    except ValueError:
        raise ValueError(f"--baseline must be 'a,b', got '{spec}'.")
    if not (0 <= a < n_ant and 0 <= b < n_ant):
        raise ValueError(f"Antennas must be in 0..{n_ant - 1}.")
    return a, b


def auto_indices(n_ant: int) -> list[int]:
    """Flat indices of autocorrelations in the upper-triangle ordering."""
    return [k * n_ant - k * (k - 1) // 2 for k in range(n_ant)]


def frequency_axis(n_freq: int, freq_start: float) -> np.ndarray:
    return freq_start + np.arange(n_freq) * CHANNEL_WIDTH_MHZ


def annotate_rfi(ax, freq: np.ndarray, rfi_channels: list[int]) -> None:
    """Draw vertical dashed lines at RFI channel frequencies within range."""
    fmin, fmax = float(freq[0]), float(freq[-1])
    for ch in rfi_channels:
        f_ch = global_channel_to_mhz(ch)
        if fmin <= f_ch <= fmax:
            ax.axvline(f_ch, color="red", alpha=0.35, lw=0.8, ls="--")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_baseline(fig, spec_mag, spec_phase,
                  vis, freq, a, b, n_ant, rfi_channels, label):
    """Two-row panel: |V| and phase for one baseline."""
    flat, conj = visibility_product_index(a, b, n_ant)
    v = vis[:, flat]
    if conj:
        v = np.conj(v)

    ax_mag = fig.add_subplot(spec_mag)
    ax_mag.plot(freq, np.abs(v), lw=1.0, color="C0")
    ax_mag.set_ylabel(f"|V({a},{b})|")
    ax_mag.set_title(label)
    ax_mag.grid(True, alpha=0.3)
    v_abs = np.abs(v)
    finite = v_abs[np.isfinite(v_abs) & (v_abs > 0)]
    if finite.size > 0 and finite.max() / finite.min() > 100:
        ax_mag.set_yscale("log")
    annotate_rfi(ax_mag, freq, rfi_channels)

    ax_ph = fig.add_subplot(spec_phase, sharex=ax_mag)
    ax_ph.plot(freq, np.angle(v), lw=1.0, color="C1")
    ax_ph.set_ylabel(f"arg V({a},{b}) [rad]")
    ax_ph.set_xlabel("Frequency (MHz)")
    ax_ph.set_ylim(-np.pi, np.pi)
    ax_ph.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_ph.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax_ph.grid(True, alpha=0.3)
    annotate_rfi(ax_ph, freq, rfi_channels)


def draw_autos(fig, spec_row, vis, freq, n_ant, rfi_channels, label):
    """One-row panel: all 64 autocorrelations vs frequency."""
    auto = np.abs(vis[:, auto_indices(n_ant)])
    ax = fig.add_subplot(spec_row)
    for k in range(n_ant):
        ax.plot(freq, auto[:, k], lw=0.4, alpha=0.5)
    ax.plot(freq, np.median(auto, axis=1), color="black", lw=2.0,
            label="median across antennas")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|Auto| [arb.]")
    ax.set_title(f"{label} — {n_ant} autocorrelations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    annotate_rfi(ax, freq, rfi_channels)


def draw_heatmap(fig, spec_row, vis, freq, n_ant, rfi_channels, label, block=10):
    """One-row panel: |V| heatmap over (baseline, frequency)."""
    mag = np.abs(vis)  # (n_freq, n_prod)
    n_freq, n_prod = mag.shape

    if block > 1:
        n_blocks = n_prod // block
        if n_blocks >= 1:
            mag = mag[:, :n_blocks * block].reshape(
                n_freq, n_blocks, block
            ).mean(axis=2)
            n_prod_plot = n_blocks
            ylabel = f"Baseline block ({block} each)"
        else:
            n_prod_plot = n_prod
            ylabel = "Baseline index"
    else:
        n_prod_plot = n_prod
        ylabel = "Baseline index"

    ax = fig.add_subplot(spec_row)
    vmin = float(np.percentile(mag, 5))
    vmax = float(np.percentile(mag, 95))
    im = ax.imshow(
        mag.T,
        aspect="auto",
        origin="lower",
        extent=[freq[0], freq[-1], 0, n_prod_plot],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{label} — |V| heatmap ({n_prod} products)")
    annotate_rfi(ax, freq, rfi_channels)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="|V| [arb.]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    sources = find_source_files(args.input, args.source)
    if not sources:
        print(f"Error: no files matching 'corr_{args.source}_*.bin' "
              f"in {args.input}.", file=sys.stderr)
        sys.exit(1)

    print(f"=== SNAPSHOT VIEW: {args.source} ===")
    print(f"Directory : {args.input}")
    print(f"Conditions: {sorted(sources.keys())}")

    if args.n_freq != KOTEKAN_GPU_N_FREQ // 2:
        print(f"WARNING: --n-freq={args.n_freq} but the expected snapshot "
              f"size is {KOTEKAN_GPU_N_FREQ // 2}. Proceeding anyway.")

    # Decide which conditions to plot
    if args.compare:
        if args.condition is not None:
            print("Note: --condition ignored when --compare is set.")
        conditions_to_plot = sorted(sources.keys())
    else:
        if args.condition is not None:
            if args.condition not in sources:
                print(f"Error: condition '{args.condition}' not available. "
                      f"Choices: {sorted(sources.keys())}", file=sys.stderr)
                sys.exit(1)
            conditions_to_plot = [args.condition]
        else:
            if "with_noise" in sources:
                conditions_to_plot = ["with_noise"]
            elif "no_noise" in sources:
                conditions_to_plot = ["no_noise"]
            else:
                conditions_to_plot = [sorted(sources.keys())[0]]
            print(f"Auto-selected condition: {conditions_to_plot[0]} "
                  f"(use --condition to override, --compare for all)")

    try:
        a, b = parse_baseline(args.baseline, args.num_antennas)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    rfi_channels = load_rfi_channels(args.input)

    freq = frequency_axis(args.n_freq, args.freq_start)
    print(f"Frequency axis: {freq[0]:.4f} to {freq[-1]:.4f} MHz "
          f"({len(freq)} channels, {CHANNEL_WIDTH_MHZ} MHz each)")
    print(f"NOTE: only HALF the CHARTS band is present. If this file is the "
          f"upper half, pass --freq-start 400.8.")

    os.makedirs(args.output, exist_ok=True)

    vis_by_cond: dict[str, np.ndarray] = {}

    print("\n=== LOADING ===")
    for cond in conditions_to_plot:
        path = sources[cond]
        try:
            vis = load_gpu_visibilities(
                path,
                n_freq=args.n_freq,
                num_antennas=args.num_antennas,
                frame_format=args.frame_format,
            )
        except Exception as exc:
            print(f"  [{cond}] FAILED: {exc}")
            continue
        vis_by_cond[cond] = vis
        print(f"  [{cond}] shape={vis.shape}, dtype={vis.dtype}, "
              f"file={Path(path).name}")

    if not vis_by_cond:
        print("No conditions loaded successfully. Aborting.", file=sys.stderr)
        sys.exit(1)

    modes = ["baseline", "autos", "heatmap"] if args.mode == "all" else [args.mode]

    for mode in modes:
        n_cols = len(vis_by_cond)
        fig_width = 5.0 * n_cols
        fig_height = 6.0 if mode == "baseline" else 4.5

        fig = plt.figure(figsize=(fig_width, fig_height),
                         constrained_layout=True)

        gs = fig.add_gridspec(
            2 if mode == "baseline" else 1,
            n_cols,
        )

        for col, (cond, vis) in enumerate(vis_by_cond.items()):
            label = f"{args.source} [{cond}]"
            if mode == "baseline":
                draw_baseline(
                    fig,
                    gs[0, col],          # spec para |V|
                    gs[1, col],          # spec para fase
                    vis, freq, a, b,
                    args.num_antennas, rfi_channels, label,
                )
            elif mode == "autos":
                draw_autos(
                    fig, gs[0, col],
                    vis, freq, args.num_antennas, rfi_channels, label,
                )
            elif mode == "heatmap":
                draw_heatmap(
                    fig, gs[0, col],
                    vis, freq, args.num_antennas, rfi_channels, label,
                    block=args.heatmap_block,
                )

        fig.suptitle(f"{args.source} — {mode}", fontsize=13, fontweight="bold")

        out_name = f"{args.source}_{mode}"
        if args.compare:
            out_name += "_compare"
        elif len(conditions_to_plot) == 1:
            out_name += f"_{conditions_to_plot[0]}"
        out_path = os.path.join(args.output, out_name + ".png")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()