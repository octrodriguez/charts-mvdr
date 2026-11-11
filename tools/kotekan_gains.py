#!/usr/bin/env python
"""Full-width gains file for the Kotekan consumers from a calibration solution.

cal_pipeline_gpu (charts-utils) writes one column per calibrated antenna, (n_freq, n_cols):
a .npz gain file (antennas and format version inside) or, before gains format 2, a bare
.npy (--gain-antennas and --gains-format 1). The Kotekan loaders (chartsMvdr::load_gains_npy, used by
chartsMVDRBeams and cudaDirectBeamTracker) expect one column per correlator input,
(n_freq, n_inputs); a file with fewer columns is only accepted with an explicit antenna
subset, and chartsMVDRBeams passes none (a 4-column file is mapped onto a fixed antenna
set of an earlier configuration). This tool writes the agreed full-width layout: column
j of the solution at logical input --gain-antennas[j], NaN at every other input, so the
consumers drop the inputs without a solution.

With --interp-gains the channels without a solution (usually the RFI channels flagged in
the calibration) are filled as in 17_real_nulling.py (charts_mvdr.tracking.
interpolate_gains: amplitude and unwrapped phase linear in channel, edges per
--gain-edge), so that a beamformer can still form beams there; otherwise those channels
stay NaN and the consumers drop every antenna in them.

The width is --num-inputs, or array.n_antennas of a run metadata file (--run-metadata),
i.e. the number of signal inputs of the correlator: one column per input, whatever the
number of antennas or polarizations behind them. The Kotekan loaders conjugate what they
read to obtain the voltage-domain gains, so the output always holds the conjugate of the
voltage gains (gains format version 1 of charts_mvdr.real_io), whatever the version of the
input: a format-2 gain file is conjugated here. The sidecar records format_version 1.

Writes OUT (complex64) and OUT with suffix .json: the sidecar of the gains format, with
the keys of the source sidecar when one exists next to --gains, plus notes on the
expansion and the interpolated channels.

Example:
  python tools/kotekan_gains.py --gains gains_sun_<start>_<stop>.npz \\
      --run-metadata run_metadata.json --interp-gains --out gains_full.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from charts_mvdr import real_io, tracking


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    real_io.add_gains_arguments(ap)
    width = ap.add_mutually_exclusive_group(required=True)
    width.add_argument("--num-inputs", type=int, help="Correlator signal inputs (file width).")
    width.add_argument("--run-metadata", type=Path,
                       help="run_metadata_*.json of the run: width = array.n_antennas.")
    ap.add_argument("--interp-gains", action="store_true",
                    help="Fill channels without a solution (see tracking.interpolate_gains).")
    ap.add_argument("--gain-edge", default="linear", choices=list(tracking.EDGE_MODES))
    ap.add_argument("--out", required=True, type=Path)
    return ap.parse_args(argv)


def num_inputs_from_metadata(path: Path) -> int:
    meta = json.loads(Path(path).read_text())
    try:
        return int(meta["array"]["n_antennas"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"error: {path} has no array.n_antennas") from None


def main(argv=None) -> int:
    args = parse_args(argv)
    args.gains = Path(args.gains)
    n_inputs = args.num_inputs if args.num_inputs else num_inputs_from_metadata(args.run_metadata)
    try:
        g, cols, file_interp = real_io.gains_from_args(args)
        version = real_io.read_gains(args.gains, fmt=args.gains_format, key=args.gains_key).convention
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    solved = np.isfinite(g).all(axis=1) & ~file_interp.any(axis=1)

    interp = np.zeros(g.shape, dtype=bool)
    if args.interp_gains:
        # Channel index as the abscissa: uniform channels, so the interpolation and the edge
        # delay fit are the same as with frequencies (linear change of variable).
        g, interp = tracking.interpolate_gains(g, np.arange(g.shape[0], dtype=float),
                                               edge=args.gain_edge)
    try:
        full = tracking.expand_gains(g, cols, n_inputs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, full.astype(np.complex64))

    src_sidecar = args.gains.with_suffix(".json")
    meta = {}
    if src_sidecar.exists():
        doc = json.loads(src_sidecar.read_text())
        meta = {k: doc.get(k) for k in tracking.SIDECAR_KEYS if k in doc}
    meta["antennas"] = cols
    meta["format_version"] = real_io.INTERNAL_CONVENTION
    note = (f"expanded by tools/kotekan_gains.py from {args.gains.name} "
            f"(gains format {version}, {g.shape[1]} columns) to {n_inputs} inputs, NaN outside "
            f"{cols}; holds the conjugate of the voltage gains, as the Kotekan loaders expect; "
            f"{int(solved.sum())} of {g.shape[0]} channels solved in all antennas")
    if file_interp.any():
        note += f"; {int(file_interp.any(axis=1).sum())} channels interpolated by the calibration"
    if args.interp_gains:
        note += (f"; {int(interp.any(axis=1).sum())} channels interpolated here "
                 f"(gain edge {args.gain_edge})")
    meta["notes"] = note
    side = tracking.write_sidecar(args.out, **meta)
    print(f"saved {args.out} {full.shape} complex64 and {side.name}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
