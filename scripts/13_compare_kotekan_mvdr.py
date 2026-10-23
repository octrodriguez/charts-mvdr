#!/usr/bin/env python
"""Compare the Kotekan chartsMVDRBeams output with the NumPy reference on the same frames.

RUN_DIR is the output directory of jobs/mvdr_kotekan.sh: RUN_DIR/corr holds the
correlation frames of the run (corr_%07d.bin) and RUN_DIR/mvdr the stage output
(mvdr_power_%07d.bin, mvdr_weights_%07d.bin). The reference
(charts_mvdr.stage_reference.run_stage) is run on RUN_DIR/corr with the same parameters
(they must match the ones in the Kotekan config) and compared with a floating-point
tolerance: the stage computes in double and stores float32, so agreement is expected at
~1e-7, not bit for bit.

Checks: update frames and frames per window identical; NaN pattern of the beam power
identical; max relative error of the beam power, of the weights (||dw|| / ||w|| per channel
and beam) and of the WNG below --tol. Prints "MVDR STAGE MATCH" or "MVDR STAGE MISMATCH".
If the stage produced fewer frames than the window (Kotekan stopped early), only the common
frames are compared and the count is reported.

Optionally saves the stage output in the npz format of 12_stage_reference.py (--out), so the
same analysis tools apply to both.

Example:
  python scripts/13_compare_kotekan_mvdr.py --run-dir RUN_DIR --window-dir WINDOW_DIR \\
      --beams 0.4677 0.4801 -0.10 0.25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import load_window_dir_meta
from charts_mvdr.stage_reference import (StageConfig, load_correlation_stack, load_gains,
                                         read_stage_power, read_stage_weights, run_stage)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--window-dir", required=True, type=Path)
    ap.add_argument("--beams", required=True, type=float, nargs="+")
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--num-ant", type=int, default=64)
    ap.add_argument("--samples-per-frame", type=int, default=1536)
    ap.add_argument("--half-window", type=int, default=40)
    ap.add_argument("--guard", type=int, default=1)
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--update-every", type=int, default=1)
    ap.add_argument("--loading", type=float, default=10.0)
    ap.add_argument("--loading-ref", default="noise", choices=("noise", "trace", "absolute"))
    ap.add_argument("--antenna-order", choices=("descending", "ascending"),
                    default="descending",
                    help="logical antenna of correlator element e for the gains file")
    ap.add_argument("--gains", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", type=Path, default=None, help="save the stage output as npz")
    args = ap.parse_args(argv)
    if len(args.beams) % 2:
        ap.error("--beams needs pairs l m")
    return args


def main(argv=None):
    args = parse_args(argv)
    beams = np.asarray(args.beams, dtype=float).reshape(-1, 2)
    nb, nf, na = beams.shape[0], args.num_freq, args.num_ant
    mvdr_dir, corr_dir = args.run_dir / "mvdr", args.run_dir / "corr"
    P = read_stage_power(mvdr_dir, nf, nb)
    W = read_stage_weights(mvdr_dir, nf, nb, na)

    meta = load_window_dir_meta(args.window_dir)
    freqs_mhz = np.asarray(meta["frequencies_mhz"], dtype=float)[:nf]
    cfg = StageConfig(beams=beams, positions_x_m=meta["antenna_pos_x_m"],
                      positions_y_m=meta["antenna_pos_y_m"],
                      samples_per_frame=args.samples_per_frame,
                      half_window=args.half_window, guard=args.guard,
                      min_frames=args.min_frames, update_every=args.update_every,
                      loading=args.loading, loading_ref=args.loading_ref,
                      antenna_order=args.antenna_order)
    gains = load_gains(args.gains, n_ant=na) if args.gains is not None else None

    ref_p, ref_w, ref_wng = [], [], []
    upd = nwin = None
    for i0 in range(0, nf, args.chunk):
        ch = np.arange(i0, min(nf, i0 + args.chunk))
        _, S = load_correlation_stack(corr_dir, nf, na, ch)
        out = run_stage(S, freqs_mhz[ch] * 1e6, cfg,
                        gains=None if gains is None else gains[ch])
        ref_p.append(out.power)
        ref_w.append(np.asarray(out.weights))
        ref_wng.append(np.asarray(out.wng))
        upd, nwin = out.update_frames, out.n_window
    ref_p = np.concatenate(ref_p, axis=2)
    ref_w = np.concatenate(ref_w, axis=1)
    ref_wng = np.concatenate(ref_wng, axis=1)

    n_ref, n_st = ref_p.shape[1], P.shape[1]
    n = min(n_ref, n_st)
    print(f"frames: reference {n_ref}, stage {n_st}; comparing {n}")
    ok = True
    upd_common = [t for t in upd if t < n]
    st_upd = [int(t) for t in W["update_frames"] if t < n]
    if st_upd != upd_common:
        print(f"update frames differ: stage {st_upd[:10]}..., reference {upd_common[:10]}...")
        ok = False
    else:
        k = len(upd_common)
        if list(W["n_window"][:k]) != [int(nwin[t]) for t in upd_common]:
            print("frames per window differ")
            ok = False
        dw = np.linalg.norm(W["weights"][:k] - ref_w[:k], axis=-1)
        nw = np.linalg.norm(ref_w[:k], axis=-1)
        rel_w = float(np.max(dw[nw > 0] / nw[nw > 0])) if k else 0.0
        rel_wng = float(np.max(np.abs(W["wng"][:k] - ref_wng[:k]) / ref_wng[:k])) if k else 0.0
        print(f"weights: {k} updates, max rel {rel_w:.2e}; WNG max rel {rel_wng:.2e}")
        ok &= rel_w < args.tol and rel_wng < args.tol
    a, b = P[:, :n], ref_p[:, :n]
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        print(f"NaN pattern differs: stage {int(np.isnan(a).sum())}, "
              f"reference {int(np.isnan(b).sum())}")
        ok = False
    fin = np.isfinite(b) & (b != 0)
    rel_p = np.abs(a[fin] - b[fin]) / np.abs(b[fin])
    for name, plane in (("DFT", 0), ("MVDR", 1)):
        m = fin[plane]
        r = np.abs(a[plane][m] - b[plane][m]) / np.abs(b[plane][m])
        print(f"power {name}: max rel {r.max():.2e}, median {np.median(r):.2e}")
    ok &= bool(rel_p.max() < args.tol)
    print(f"MVDR STAGE {'MATCH' if ok else 'MISMATCH'} (tol {args.tol:g})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        np.savez_compressed(args.out, power=P, n_window=np.asarray(nwin)[:n_st],
                            channels=np.arange(nf), freqs_mhz=freqs_mhz, beams=beams,
                            update_frames=W["update_frames"], weights=W["weights"],
                            wng=W["wng"], params=json.dumps(params), source="kotekan")
        print(f"saved {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
