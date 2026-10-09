#!/usr/bin/env python3
"""
01_inspect.py — Dataset diagnostic tool for CHARTS correlation frames.

Robustly handles:
    - Single-block and split-frequency datasets (corr_freq_0/1)
    - Files with or without FPGA metadata prefixes
    - Headerless Astron payloads
    - Any file-naming pattern passed via --pattern

Does NOT assume a cadence, antenna count, or file layout.

Usage:
    python 01_inspect.py --input <dir>
    python 01_inspect.py --input <dir> --pattern "win03UTC_64ant_*.bin"
    python 01_inspect.py --input <dir> --recursive
    python 01_inspect.py --input <dir> --summary
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

from charts_utils.core.io import (
    gpu_corr_to_visibilities,
    infer_gpu_corr_num_antennas,
    load_gpu_corr_frame,
    load_gpu_visibilities,
)
from charts_utils.core.gpu_frames import (
    discover_gpu_corr_frame_groups,
    read_gpu_corr_frame_header,
)
from charts_utils.core.kotekan import KOTEKAN_GPU_N_FREQ

try:
    from charts_utils.core.gpu_frames import gpu_corr_frame_index
except ImportError:
    _FRAME_RE = re.compile(r".+_(\d+)\.bin$")

    def gpu_corr_frame_index(path: str) -> int:
        m = _FRAME_RE.match(os.path.basename(path))
        if m is None:
            raise ValueError(f"Cannot parse frame index from '{path}'.")
        return int(m.group(1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect a CHARTS GPU correlation dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True)
    p.add_argument("--pattern", default="*.bin")
    p.add_argument("--n-freq", type=int, default=KOTEKAN_GPU_N_FREQ)
    p.add_argument("--max-inspect", type=int, default=5)
    p.add_argument("--recursive", action="store_true",
                   help="Also look for .bin files in subdirectories.")
    p.add_argument("--summary", action="store_true",
                   help="Print a compact summary only; skip verbose dumps.")
    return p


def try_read_header(path: str):
    try:
        return read_gpu_corr_frame_header(path)
    except (ValueError, OSError, struct.error):
        return None


def list_files(directory: str, pattern: str, recursive: bool) -> list[str]:
    root = Path(directory)
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(str(p) for p in iterator if p.is_file())


def deep_find(obj, key):
    """Recursive search for `key` anywhere in a nested dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = deep_find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = deep_find(item, key)
            if r is not None:
                return r
    return None


def summarise_events_json(data) -> None:
    """If `data` looks like a list of events with event_id/event_type, print a table."""
    if not isinstance(data, list) or not data:
        return
    first = data[0]
    if not isinstance(first, dict) or "event_type" not in first:
        return

    print("\n  --- Event table ---")
    header = f"  {'event_id':<22} {'type':<14} {'t_start':>10} {'dur':>8} {'DM':>8} {'amp':>8}  channels"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ev in data:
        if not isinstance(ev, dict):
            continue
        ch = ev.get("channels")
        ch_str = "-" if ch is None else ",".join(str(c) for c in ch[:4]) + (
            "..." if isinstance(ch, list) and len(ch) > 4 else ""
        )
        print(
            f"  {str(ev.get('event_id', '?')):<22} "
            f"{str(ev.get('event_type', '?')):<14} "
            f"{float(ev.get('t_start_s', 0.0)):>10.3f} "
            f"{float(ev.get('duration_s', 0.0)):>8.3f} "
            f"{float(ev.get('dm', 0.0)):>8.2f} "
            f"{float(ev.get('nominal_amp', 0.0)):>8.2f}  "
            f"{ch_str}"
        )


def inspect_metadata_files(directory: str, recursive: bool, summary: bool) -> dict:
    """Print JSON/HDF5 metadata and return extracted key fields."""
    print("\n=== METADATA FILES ===")
    extracted = {"signal_antennas": None, "position_overrides": None}

    root = Path(directory)
    iterator = root.rglob("*") if recursive else root.iterdir()
    candidates = sorted(p for p in iterator if p.is_file())

    if not candidates:
        print("  (no files found at this level)")
        return extracted

    for path in candidates:
        suffix = path.suffix.lower()

        if suffix == ".json":
            print(f"\n[JSON] {path.name}  ({path.stat().st_size} bytes)")
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  Failed to parse: {exc}")
                continue

            for key in ("signal_antennas", "position_overrides"):
                val = deep_find(data, key)
                if val is not None:
                    extracted[key] = val
                    print(f"  -> {key}: {val}")

            summarise_events_json(data)

            if not summary:
                pretty = json.dumps(data, indent=2)
                if len(pretty) < 3000:
                    print(pretty)
                else:
                    print(pretty[:3000] + "\n  ... (truncated)")

        elif suffix in (".h5", ".hdf5"):
            print(f"\n[HDF5] {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
            try:
                import h5py
                with h5py.File(path, "r") as f:
                    if summary:
                        print(f"  Root keys: {list(f.keys())}")
                    else:
                        print("  Root keys:")
                        for k in f.keys():
                            obj = f[k]
                            if hasattr(obj, "shape"):
                                print(f"    {k}: shape={obj.shape}, dtype={obj.dtype}")
                            else:
                                print(f"    {k}: group with {len(obj.keys())} children")
                    print("  Root attributes:")
                    for k, v in f.attrs.items():
                        v_str = str(v)
                        if len(v_str) > 200:
                            v_str = v_str[:200] + "..."
                        print(f"    {k}: {v_str}")
            except (OSError, KeyError) as exc:
                print(f"  Failed to open HDF5: {exc}")

    return extracted


def infer_n_ant_from_n_prod(n_prod: int) -> tuple[int, bool]:
    """Return (n_ant, is_triangular). If not triangular, caller is warned."""
    disc = 1 + 8 * n_prod
    sqrt_disc = np.sqrt(disc)
    is_tri = np.isclose(sqrt_disc, round(sqrt_disc))
    n_ant = int((sqrt_disc - 1) / 2)
    return n_ant, bool(is_tri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if not os.path.isdir(args.input):
        print(f"Error: '{args.input}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"=== DATASET: {args.input} ===")

    # ------------------------------------------------------------------
    # 1. Discover frames
    # ------------------------------------------------------------------
    print("\n=== FRAME DISCOVERY ===")
    print(f"Pattern: {args.pattern} (recursive={args.recursive})")

    files = list_files(args.input, args.pattern, args.recursive)
    print(f"Files matched: {len(files)}")
    if not files:
        sys.exit(1)
    print(f"First : {[Path(f).name for f in files[:2]]}")
    print(f"Last  : {[Path(f).name for f in files[-2:]]}")

    groups = None
    try:
        groups = discover_gpu_corr_frame_groups(
            args.input,
            pattern=args.pattern,
            allow_incomplete_last=True,
        )
        print(f"discover_gpu_corr_frame_groups: {len(groups)} groups")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        print(f"discover_gpu_corr_frame_groups failed ({exc}); using file list.")

    # ------------------------------------------------------------------
    # 2. Frequency-block structure
    # ------------------------------------------------------------------
    print("\n=== FREQUENCY STRUCTURE ===")
    split_blocks = any("corr_freq_" in Path(f).name for f in files)

    if groups and len(groups) > 0 and len(groups[0].frequency_blocks) > 1:
        n_blocks = len(groups[0].frequency_blocks)
        n_freq_per_block = args.n_freq // n_blocks
        print(f"Split into {n_blocks} frequency blocks per frame.")
        print(f"Unique frames: {len(groups)}")
        print(f"Frame indices: {groups[0].frame_index} .. {groups[-1].frame_index}")
    else:
        n_blocks = 1
        n_freq_per_block = args.n_freq
        if split_blocks:
            blocks = set()
            for f in files:
                m = re.search(r"corr_freq_(\d+)_", Path(f).name)
                if m:
                    blocks.add(int(m.group(1)))
            print(f"Names contain corr_freq_ but grouping did not pair. "
                  f"Blocks detected: {sorted(blocks)}")
        else:
            print("Single-block dataset.")

    # ------------------------------------------------------------------
    # 3. FPGA headers (iterate over groups to avoid block duplicates)
    # ------------------------------------------------------------------
    print(f"\n=== FPGA HEADERS (first {args.max_inspect} frames) ===")
    if groups:
        iter_targets = [(f"frame {g.frame_index} (blocks={g.frequency_blocks})",
                         g.paths[0]) for g in groups[: args.max_inspect]]
    else:
        iter_targets = [(Path(f).name, f) for f in files[: args.max_inspect]]

    for label, path in iter_targets:
        hdr = try_read_header(path)
        if hdr is None:
            print(f"  {label}: headerless / no valid metadata prefix")
        else:
            print(f"  {label}: metadata_size={hdr.metadata_size}, "
                  f"fpga_seq={hdr.fpga_seq_num}, time0_fpga={hdr.time0_fpga} us")

    # ------------------------------------------------------------------
    # 4. Cadence inference
    # ------------------------------------------------------------------
    print("\n=== CADENCE INFERENCE ===")
    cadence_s = None
    if groups is None or len(groups) < 2:
        print("Need at least two groups to infer cadence.")
    else:
        hdr0 = try_read_header(groups[0].paths[0])
        hdr1 = try_read_header(groups[1].paths[0])
        if hdr0 is not None and hdr1 is not None:
            dt_us = hdr1.time0_fpga - hdr0.time0_fpga
            cadence_s = dt_us / 1e6
            print(f"Frame {groups[0].frame_index} -> {groups[1].frame_index}: "
                  f"delta = {dt_us} us = {cadence_s:.6f} s")
        else:
            print("No FPGA headers on the first two groups.")
            try:
                dt = (os.path.getmtime(groups[1].paths[0])
                      - os.path.getmtime(groups[0].paths[0]))
                cadence_s = dt
                print(f"Fallback to mtime delta: {dt:.4f} s  "
                      f"(NOTE: unreliable on GPFS after rsync/cp)")
            except OSError as exc:
                print(f"  mtime fallback failed: {exc}")

    # ------------------------------------------------------------------
    # 5. Antenna count inference
    # ------------------------------------------------------------------
    print("\n=== ANTENNA COUNT INFERENCE ===")
    n_ant = None
    sample_path = groups[0].paths[0] if groups else files[0]
    try:
        n_ant = infer_gpu_corr_num_antennas(sample_path, n_freq_per_block)
        print(f"Inferred physical antennas from payload "
              f"(n_freq_per_file={n_freq_per_block}): {n_ant}")
    except (ValueError, OSError) as exc:
        print(f"Could not infer antenna count: {exc}")
        n_ant = 64
        print(f"Falling back to explicit n_ant={n_ant}")

    # ------------------------------------------------------------------
    # 6. Frame load test
    # ------------------------------------------------------------------
    print("\n=== FRAME LOAD TEST (first frame) ===")
    n_ant_full = None
    per_ant_median = None
    try:
        if groups and len(groups[0].paths) > 1:
            vis = load_gpu_visibilities(groups[0], args.n_freq, n_ant)
            print(f"Visibilities (concatenated blocks): "
                  f"shape={vis.shape}, dtype={vis.dtype}")
        else:
            corr = load_gpu_corr_frame(sample_path, n_freq_per_block,
                                        num_antennas=n_ant)
            print(f"Raw corr tensor: shape={corr.shape}, dtype={corr.dtype}")
            vis = gpu_corr_to_visibilities(corr)
            print(f"Visibilities: shape={vis.shape}, dtype={vis.dtype}")

        if vis.ndim != 2:
            raise ValueError(
                f"Expected 2D visibilities, got shape {vis.shape}. "
                f"The API may have changed or polarizations are stacked."
            )

        n_prod = vis.shape[1]
        n_ant_full, is_tri = infer_n_ant_from_n_prod(n_prod)
        if not is_tri:
            print(f"WARNING: n_prod={n_prod} is not a triangular number. "
                  f"Inferred n_ant={n_ant_full} may be wrong "
                  f"(multiple polarizations?).")

        auto_indices = [k * n_ant_full - k * (k - 1) // 2
                        for k in range(n_ant_full)]
        autos = np.abs(vis[:, auto_indices])
        per_ant_median = np.median(autos, axis=0)
        global_median = float(np.median(per_ant_median))
        threshold = 0.1 * global_median

        print(f"\nAutocorrelation summary (n_ant={n_ant_full}):")
        print(f"  Global median: {global_median:.4e}")

        if args.summary:
            n_low = int(np.sum(per_ant_median < threshold))
            print(f"  Antennas below 10% of median: {n_low} / {n_ant_full}")
        else:
            print(f"  Per-antenna medians (threshold < {threshold:.4e}):")
            for a in range(n_ant_full):
                flag = "  <-- LOW" if per_ant_median[a] < threshold else ""
                print(f"    ant {a:3d}: {per_ant_median[a]:.4e}{flag}")

    except (ValueError, OSError) as exc:
        print(f"Failed to load first frame: {exc}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # 7. Metadata extraction
    # ------------------------------------------------------------------
    extracted = inspect_metadata_files(args.input, args.recursive, args.summary)

    # ------------------------------------------------------------------
    # 8. Cross-check + duration report
    # ------------------------------------------------------------------
    print("\n=== METADATA CROSS-CHECK ===")
    print(f"Physical antennas (from payload): {n_ant}")
    if extracted["signal_antennas"] is not None:
        sa = extracted["signal_antennas"]
        if isinstance(sa, list):
            print(f"signal_antennas (from JSON): {len(sa)} listed -> {sa}")
        else:
            print(f"signal_antennas (from JSON): {sa}")
    if extracted["position_overrides"] is not None:
        po = extracted["position_overrides"]
        if isinstance(po, list):
            print(f"position_overrides (from JSON): {len(po)} entries")
            for ov in po[:3]:
                print(f"  {ov}")
        else:
            print(f"position_overrides (from JSON): {po}")

    if groups and cadence_s is not None:
        total_s = len(groups) * cadence_s
        print("\n=== DATASET DURATION ===")
        print(f"  Frames  : {len(groups)}")
        print(f"  Cadence : {cadence_s:.6f} s")
        print(f"  Total   : {total_s:.2f} s = {total_s / 60:.2f} min "
              f"= {total_s / 3600:.2f} h")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()