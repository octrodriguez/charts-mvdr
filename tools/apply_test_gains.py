#!/usr/bin/env python
"""Apply known complex antenna gains to a simulated baseband window (calibration test).

Writes, under OUT_ROOT:
  clean/   the first --num-frames frames of the window (symlinks) with its metadata
  gained/  the same frames with every antenna multiplied by a gain phase q in {1, j, -1, -j}
           (random per channel and antenna), plus the metadata
  gains_test.npy / .json  the gains in the charts-utils format (complex64, (n_rows, n_ant),
           convention v_obs = g v, rows = channels from 300 MHz): g = A q for the window
           channels and 1 elsewhere; sidecar as in the gains format spec.

Quarter-turn phases keep the 4-bit samples exact, except a component of -8 that would
become +8, which is saturated to +7 (counted and reported; it limits how exactly a
calibrated beam can match the clean one). The amplitude A (--amplitude) is put in the
file only: a beamformer that applies 1 / g to the gained window must reproduce the clean
beam scaled by exactly 1 / A, which checks the amplitude (divide, not multiply) as well as
the phase (1 / g, not g or 1 / g*).

Antennas are logical (the order of the gains file); element e of a frame holds antenna
N - 1 - e with --antenna-order descending (the cudaDirectBeamTracker default) or e with
ascending. A beamformer must use the same mapping for the test to pass.

Example:
  python tools/apply_test_gains.py --window-dir WINDOW_DIR --out-root OUT --num-frames 20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import FRAME_NAME_RE


def rotate_quarter(re: np.ndarray, im: np.ndarray, k: np.ndarray):
    """(re, im) * j^k for integer arrays and k in {0, 1, 2, 3} (broadcast), saturating +8."""
    out_re = np.where(k == 0, re, np.where(k == 1, -im, np.where(k == 2, -re, im)))
    out_im = np.where(k == 0, im, np.where(k == 1, re, np.where(k == 2, -im, -re)))
    sat = int(np.count_nonzero(out_re > 7) + np.count_nonzero(out_im > 7))
    return np.minimum(out_re, 7), np.minimum(out_im, 7), sat


def unpack(raw: np.ndarray):
    re = (raw & 0x0F).astype(np.int16)
    im = (raw >> 4).astype(np.int16)
    re[re > 7] -= 16
    im[im > 7] -= 16
    return re, im


def pack(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    return ((re & 0x0F) | ((im & 0x0F) << 4)).astype(np.uint8)


def window_frames(window_dir: Path):
    frames = {}
    for p in window_dir.glob("*.bin"):
        m = FRAME_NAME_RE.match(p.name)
        if m:
            frames.setdefault(m["window"], []).append((int(m["index"]), p))
    if len(frames) != 1:
        raise SystemExit(f"expected one window in {window_dir}, found {sorted(frames)}")
    name, items = next(iter(frames.items()))
    return name, [p for _, p in sorted(items)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-dir", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--num-ant", type=int, default=64)
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--first-channel", type=int, default=0,
                    help="row of the gains file for channel 0 of the window")
    ap.add_argument("--file-rows", type=int, default=672, help="rows of the gains file")
    ap.add_argument("--amplitude", type=float, default=2.0)
    ap.add_argument("--antenna-order", choices=("descending", "ascending"),
                    default="descending")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    name, frames = window_frames(args.window_dir)
    frames = frames[:args.num_frames]
    nf, na = args.num_freq, args.num_ant
    rng = np.random.default_rng(args.seed)
    k_ant = rng.integers(0, 4, size=(nf, na))                     # per channel, logical antenna
    elem_to_ant = (na - 1 - np.arange(na)) if args.antenna_order == "descending" \
        else np.arange(na)
    k_elem = k_ant[:, elem_to_ant]                                 # per channel, element

    clean, gained = args.out_root / "clean", args.out_root / "gained"
    for d in (clean, gained):
        d.mkdir(parents=True, exist_ok=True)
        for extra in list(args.window_dir.glob("*_meta.h5")) + list(
                args.window_dir.glob("*events*.json")):
            link = d / extra.name
            if not link.exists():
                os.symlink(extra.resolve(), link)

    n_sat, n_bytes = 0, args.samples * nf * na
    for p in frames:
        link = clean / p.name
        if not link.exists():
            os.symlink(p.resolve(), link)
        raw = np.fromfile(p, dtype=np.uint8)
        meta_size = int(raw[:4].view("<u4")[0])
        head, data = raw[:4 + meta_size], raw[4 + meta_size:]
        if data.size != n_bytes:
            raise SystemExit(f"{p}: {data.size} data bytes, expected {n_bytes}")
        re, im = unpack(data.reshape(args.samples, nf, na))
        re, im, sat = rotate_quarter(re, im, k_elem[None, :, :])
        n_sat += sat
        np.concatenate([head, pack(re, im).ravel()]).tofile(gained / p.name)

    g = np.ones((args.file_rows, na), dtype=np.complex64)
    rows = slice(args.first_channel, args.first_channel + nf)
    g[rows] = (args.amplitude * (1j ** k_ant)).astype(np.complex64)
    gfile = args.out_root / "gains_test.npy"
    np.save(gfile, g)
    side = {
        "created_utc": "test", "valid_from_utc": "test", "valid_to_utc": "test",
        "calibrator": "synthetic quarter-turn gains (tools/apply_test_gains.py)",
        "ref_ant": None, "antennas": list(range(na)), "positions_file": None,
        "fengine_gains": None, "window": str(args.window_dir), "frames": len(frames),
        "amplitude": args.amplitude, "seed": args.seed, "antenna_order": args.antenna_order,
        "first_channel": args.first_channel, "saturated_components": n_sat,
        "saturated_fraction": n_sat / (2.0 * n_bytes * len(frames)),
    }
    gfile.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(f"window {name}: {len(frames)} frames -> {clean} and {gained}; gains {gfile} "
          f"(A = {args.amplitude:g}, {args.antenna_order}); saturated components {n_sat} "
          f"({side['saturated_fraction']:.2e})")


if __name__ == "__main__":
    main()
