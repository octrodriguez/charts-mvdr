#!/usr/bin/env python
"""Compare formed beams of two tracker runs (rawFileWrite dumps of complex float32 frames).

Fits one complex scale between the runs, alpha = <ref, test> / <ref, ref> over all samples,
and the residual ||test - alpha ref|| / ||test||. Used for the calibration test of
tools/apply_test_gains.py: clean window without gains (ref) vs gained window with the gains
file (test) must give alpha = 1 / A and a residual set only by the saturated samples; the
gained window without gains decorrelates (residual near 1).

With --expected-scale and --tol-scale / --tol-residual it prints "TRACKER CALIBRATION MATCH"
or "TRACKER CALIBRATION MISMATCH" and exits 0 or 1. The layout inside a frame does not
matter: both runs use the same config apart from the gains.

Example:
  python tools/compare_tracker_beams.py --ref REF_DIR --test TEST_DIR --name beams \\
      --expected-scale 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import numbered_files, read_rawfile_payload


def compare(ref_dir, test_dir, name: str):
    """(alpha, residual, n_frames) over the frames present in both directories."""
    ref = numbered_files(ref_dir, name)
    test = numbered_files(test_dir, name)
    n = min(len(ref), len(test))
    if n == 0:
        raise SystemExit(f"no {name}_*.bin in both {ref_dir} and {test_dir}")
    rr = rt = tt = 0.0
    for a, b in zip(ref[:n], test[:n]):
        x = np.frombuffer(read_rawfile_payload(a), "<c8").astype(np.complex128)
        y = np.frombuffer(read_rawfile_payload(b), "<c8").astype(np.complex128)
        if x.size != y.size:
            raise SystemExit(f"{a.name}: different frame sizes")
        rr += np.vdot(x, x).real
        rt += np.vdot(x, y)
        tt += np.vdot(y, y).real
    alpha = rt / rr
    resid2 = max(tt - abs(rt) ** 2 / rr, 0.0)       # ||y - alpha x||^2
    return alpha, float(np.sqrt(resid2 / tt)), n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--name", default="beams", help="rawFileWrite file_name")
    ap.add_argument("--expected-scale", type=complex, default=None)
    ap.add_argument("--tol-scale", type=float, default=1e-3)
    ap.add_argument("--tol-residual", type=float, default=1e-2)
    args = ap.parse_args(argv)

    alpha, resid, n = compare(args.ref, args.test, args.name)
    print(f"{n} frames: alpha = {alpha.real:+.6f}{alpha.imag:+.6f}j, |alpha| = {abs(alpha):.6f}, "
          f"residual = {resid:.2e}")
    if args.expected_scale is None:
        return 0
    ok = abs(alpha - args.expected_scale) < args.tol_scale * abs(args.expected_scale) \
        and resid < args.tol_residual
    print(f"TRACKER CALIBRATION {'MATCH' if ok else 'MISMATCH'} (expected alpha "
          f"{args.expected_scale.real:g}, tol {args.tol_scale:g}; residual tol "
          f"{args.tol_residual:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
