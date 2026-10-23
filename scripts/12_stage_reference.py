#!/usr/bin/env python
"""Run the NumPy reference of the Kotekan MVDR stage on a correlation window.

Reads the cudaCorrelatorAstron dumps of CORR_DIR (one frame per dump, in dump
order), the antenna positions and channel frequencies from the window metadata
(<window>_meta.h5 in WINDOW_DIR) and optional complex gains (charts-utils .npy,
(n_freq, n_ant)), and applies charts_mvdr.stage_reference.run_stage: blind local
covariance window, DFT and MVDR weights, beam power per frame, channel and beam.

Output (npz): power (2, n_frames, n_chan, n_beam) [dft, mvdr], n_window,
dump_index, channels, freqs_mhz, beams, update_frames, weights
(n_update, n_chan, n_beam, n_ant), wng (n_update, n_chan, n_beam) and the
parameters. This is what the Kotekan stage output is compared against, and it
already gives frame-resolution dynamic spectra per beam.

Channels are processed in chunks (--chunk) to bound memory: a full window of
154 frames x 336 channels x 64^2 is 3.4 GB in complex128.

Example:
  python scripts/12_stage_reference.py --corr-dir CORR_DIR --window-dir WINDOW_DIR \\
      --beams -0.10 0.25 0.4677 0.4801 --out OUT.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import load_window_dir_meta
from charts_mvdr.stage_reference import (StageConfig, load_correlation_stack, load_gains,
                                         run_stage)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corr-dir", required=True, type=Path)
    ap.add_argument("--window-dir", required=True, type=Path)
    ap.add_argument("--beams", required=True, type=float, nargs="+",
                    help="look directions as l1 m1 l2 m2 ...")
    ap.add_argument("--channels", type=int, nargs="*", default=None)
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--num-ant", type=int, default=64)
    ap.add_argument("--samples-per-frame", type=int, default=1536)
    ap.add_argument("--half-window", type=int, default=40)
    ap.add_argument("--guard", type=int, default=1)
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--update-every", type=int, default=1)
    ap.add_argument("--loading", type=float, default=10.0)
    ap.add_argument("--loading-ref", default="noise", choices=("noise", "trace", "absolute"))
    ap.add_argument("--gains", type=Path, default=None,
                    help="complex (n_freq, n_ant) .npy, charts-utils convention v_obs = g v")
    ap.add_argument("--chunk", type=int, default=48, help="channels per pass")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if len(args.beams) % 2:
        ap.error("--beams needs pairs l m")
    return args


def main(argv=None):
    args = parse_args(argv)
    meta = load_window_dir_meta(args.window_dir)
    freqs_mhz = np.asarray(meta["frequencies_mhz"], dtype=float)
    channels = (np.arange(args.num_freq) if args.channels is None
                else np.asarray(args.channels, dtype=int))
    beams = np.asarray(args.beams, dtype=float).reshape(-1, 2)
    cfg = StageConfig(beams=beams, positions_x_m=meta["antenna_pos_x_m"],
                      positions_y_m=meta["antenna_pos_y_m"],
                      samples_per_frame=args.samples_per_frame,
                      half_window=args.half_window, guard=args.guard,
                      min_frames=args.min_frames, update_every=args.update_every,
                      loading=args.loading, loading_ref=args.loading_ref)
    gains_all = (load_gains(args.gains, n_ant=args.num_ant) if args.gains is not None
                 else None)

    powers, weights, wngs = [], [], []
    for i0 in range(0, channels.size, args.chunk):
        ch = channels[i0:i0 + args.chunk]
        dump_idx, S = load_correlation_stack(args.corr_dir, args.num_freq, args.num_ant, ch)
        g = None if gains_all is None else gains_all[ch]
        out = run_stage(S, freqs_mhz[ch] * 1e6, cfg, gains=g)
        powers.append(out.power)
        weights.append(np.asarray(out.weights, dtype=np.complex64))
        wngs.append(np.asarray(out.wng, dtype=np.float32))
        print(f"channels {ch[0]}-{ch[-1]}: {S.shape[0]} frames, "
              f"{len(out.update_frames)} updates", flush=True)

    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        power=np.concatenate(powers, axis=2).astype(np.float32),
        n_window=out.n_window, dump_index=np.asarray(dump_idx), channels=channels,
        freqs_mhz=freqs_mhz[channels], beams=beams,
        update_frames=np.asarray(out.update_frames),
        weights=np.concatenate(weights, axis=1), wng=np.concatenate(wngs, axis=1),
        params=json.dumps(params),
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
