#!/usr/bin/env python
"""Compare the Kotekan GPU correlator (cudaCorrelatorAstron) with np_correlator.

For each baseband frame <window>_%07d.bin and the matching dump corr_%07d.bin,
R_k = read_astron_correlation(...) (np_correlator convention) is compared with
R_np = sum_t v_i conj(v_j). Products of 4-bit integers summed in int32 are exact,
so agreement must be exact. Columns:

  exact      max |R_k - R_np|
  model      max |R_k - R_model|, where R_model reproduces the kernel's 4-bit
             negation overflow (-(-8) -> -8 in the imaginary part)
  bad        number of entries with R_k != R_np
  bad_ch     channels with mismatches
  n(-8)      number of -8 components in the frame

Verdicts: EXACT MATCH; MATCH EXCEPT -8 (all mismatches explained by the overflow,
i.e. data outside the F-engine range [-7, 7]); MISMATCH.

Example:
  python scripts/04_compare_correlators.py \\
      --baseband-dir $SCRATCH/data/sim_test/quick_day_gsun \\
      --corr-dir $SCRATCH/data/kotekan_corr/quick_day_gsun_<jobid>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import read_astron_correlation, read_baseband_frame
from charts_mvdr.np_correlator import correlate

FRAME_RE = re.compile(r"^(?P<window>.+)_(?P<index>\d{7})\.bin$")


def frames_by_index(directory: Path, prefix: str | None = None) -> dict[int, Path]:
    out = {}
    for p in sorted(directory.glob("*.bin")):
        m = FRAME_RE.match(p.name)
        if m and (prefix is None or m["window"] == prefix):
            out[int(m["index"])] = p
    return out


def kernel_model(V: np.ndarray) -> np.ndarray:
    """R as produced by the kernel, including the 4-bit negation overflow.

    The kernel stores S = sum conj(v_i) v_j with Im(S) = sum Re_i Im_j + Im_i (-Re_j),
    the negation done in 4 bits (-(-8) = -8); the reader returns conj(S).
    """
    re, im = V.real, V.imag
    wneg = np.where(re == -8, -8.0, -re)
    ein = lambda a, b: np.einsum("tfi,tfj->fij", a, b, optimize=True)
    S = ein(re, re) + ein(im, im) + 1j * (ein(re, im) + ein(im, wneg))
    return np.conj(S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseband-dir", type=Path, required=True)
    ap.add_argument("--corr-dir", type=Path, required=True)
    ap.add_argument("--window", default=None, help="frame prefix (default: directory name)")
    ap.add_argument("--corr-prefix", default="corr")
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--num-ant", type=int, default=64)
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    window = args.window or args.baseband_dir.name
    bb = frames_by_index(args.baseband_dir, window)
    cr = frames_by_index(args.corr_dir, args.corr_prefix)
    common = sorted(set(bb) & set(cr))[: args.max_frames]
    print(f"baseband frames: {len(bb)}  correlator dumps: {len(cr)}  compared: {len(common)}")
    if not common:
        print("No matching frame indices.", file=sys.stderr)
        return 2

    # Compare only the entries stored by the kernel (lower block triangle); the
    # reader fills the rest by Hermitian symmetry, which the overflow model breaks.
    idx = np.arange(args.num_ant) // 2
    stored = idx[:, None] >= idx[None, :]

    print(f"\n{'frame':>5} {'exact':>10} {'model':>10} {'bad':>7} {'n(-8)':>7}  bad_ch")
    rows = []
    for k in common:
        V = read_baseband_frame(bb[k], args.num_freq, args.num_ant, args.samples)
        V = V.astype(np.complex128)
        Rk = read_astron_correlation(cr[k], args.num_freq, args.num_ant)
        d_exact = np.where(stored, np.abs(Rk - correlate(V)), 0.0)
        has8 = ((V.real == -8) | (V.imag == -8)).any(axis=(0, 2))   # per channel
        n8 = int(np.sum(V.real == -8) + np.sum(V.imag == -8))
        # The overflow model only differs from R_np in channels with -8 samples
        d_model = d_exact.copy()
        if has8.any():
            ch = np.flatnonzero(has8)
            d_model[ch] = np.where(stored, np.abs(Rk[ch] - kernel_model(V[:, ch, :])), 0.0)
        d_model = d_model.max()
        bad = d_exact > 0
        bad_ch = np.flatnonzero(bad.any(axis=(1, 2)))
        rows.append((d_exact.max(), d_model, int(bad.sum()), n8))
        print(f"{k:5d} {d_exact.max():10.3g} {d_model:10.3g} {bad.sum():7d} {n8:7d}  "
              f"{bad_ch[:10].tolist()}")

    a = np.array(rows, dtype=float)
    if np.all(a[:, 0] == 0):
        verdict = "EXACT MATCH: Kotekan correlator == np_correlator"
    elif np.all(a[:, 1] == 0):
        verdict = ("MATCH EXCEPT -8: all mismatches come from samples at -8 (4-bit negation "
                   "overflow in the kernel); data in [-7, 7] are exact")
    else:
        verdict = "MISMATCH: differences not explained by the -8 overflow"
    print("\n" + verdict)
    return 0 if verdict.startswith(("EXACT", "MATCH")) else 1


if __name__ == "__main__":
    sys.exit(main())
