#!/usr/bin/env python
"""Diagnose the element layout of a cudaCorrelatorAstron dump against np_correlator.

Works on the raw 2x2 blocks (no reconstruction) and reports, for each hypothesis,
the max error separately for three groups of entries:
  off   blocks with ry > rx (all four entries)
  d_lo  diagonal blocks, entry [1, 0]   (element 2r+1, 2r)
  d_up  diagonal blocks, entry [0, 1]   (element 2r, 2r+1)
Hypotheses: block entry [a, b] equals R_np[ry*2+a, rx*2+b] or its transpose
R_np[ry*2+b, rx*2+a], each direct or conjugated, from the raw or -8-clipped voltages.
It also characterises d_up (zero? equal to d_lo? equal to conj(d_lo)?).

Usage:
  python tools/diagnose_astron_layout.py --frame <window>_0000000.bin --dump corr_0000000.bin
"""
import argparse
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import read_baseband_frame
from charts_mvdr.np_correlator import correlate


def raw_blocks(path, num_freq, num_ant, bs=2):
    path = Path(path)
    with open(path, "rb") as f:
        meta = int(np.fromfile(f, dtype="<u4", count=1)[0])
    n_rec = num_ant // bs
    nb = n_rec * (n_rec + 1) // 2
    raw = np.fromfile(path, dtype="<i4", count=num_freq * nb * bs * bs * 2, offset=4 + meta)
    raw = raw.reshape(num_freq, nb, bs, bs, 2).astype(np.float64)
    return raw[..., 0] + 1j * raw[..., 1], meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--num-ant", type=int, default=64)
    args = ap.parse_args()
    F, N = args.num_freq, args.num_ant

    V = read_baseband_frame(args.frame, F, N).astype(np.complex128)
    re, im = V.real, V.imag

    def remap(x, to):
        return np.where(x == -8, to, x)

    # Candidate interpretations of the int4 code -8 by the GPU kernel
    variants = {
        "raw":       V,
        "clip8":     remap(re, -7) + 1j * remap(im, -7),
        "neg8":      remap(re, 8) + 1j * remap(im, 8),
        "neg8_re":   remap(re, 8) + 1j * im,
        "neg8_im":   re + 1j * remap(im, 8),
        "zero8":     remap(re, 0) + 1j * remap(im, 0),
    }
    n8 = int(np.sum(re == -8) + np.sum(im == -8))
    print(f"samples with a -8 component: {n8} of {2 * V.size}")
    refs = {k: correlate(v) for k, v in variants.items()}
    B, meta = raw_blocks(args.dump, F, N)
    print(f"metadata bytes: {meta}; blocks: {B.shape}; |R| max: {np.abs(refs['raw']).max():.3g}")

    ry, rx = np.tril_indices(N // 2)
    off = ry > rx
    diag = ry == rx
    print(f"\n{'hypothesis':<28} {'off':>10} {'d_lo':>10} {'d_up':>10} {'exact%':>8}")
    for vname, R in refs.items():
        for tname, transpose in (("[a,b]", False), ("[b,a]", True)):
            for cname, f in (("direct", lambda x: x), ("conj", np.conj)):
                H = np.empty_like(B)
                for a in range(2):
                    for b in range(2):
                        i, j = (b, a) if transpose else (a, b)
                        H[:, :, a, b] = f(R[:, ry * 2 + i, rx * 2 + j])
                e = np.abs(B - H)
                print(f"{vname + ' ' + tname + ' ' + cname:<28} "
                      f"{e[:, off].max():10.3g} {e[:, diag, 1, 0].max():10.3g} "
                      f"{e[:, diag, 0, 1].max():10.3g} {100 * np.mean(e == 0):8.2f}")

    # Models of the stored block entry S[i, j] (no extra conjugation applied):
    # real part = sum(Re_i Re_j + Im_i Im_j) (no negation, exact);
    # imag part = sum(Re_i Im_j - Im_i Re_j) with one operand negated in 4-bit
    # arithmetic before the product, so that -(-8) wraps to -8.
    def wneg(x):
        return np.where(x == -8, -8.0, -x)

    def ein(a, b):
        return np.einsum("tfi,tfj->fij", a, b)

    real = ein(re, re) + ein(im, im)
    models = {
        "wrapA: Re_i Im_j + (-Im_i) Re_j": real + 1j * (ein(re, im) + ein(wneg(im), re)),
        "wrapB: Re_i Im_j + Im_i (-Re_j)": real + 1j * (ein(re, im) + ein(im, wneg(re))),
        "wrapC: (-Re_i) Im_j ... conj":    real - 1j * (ein(wneg(re), im) + ein(im, re)),
        "wrapD: Re_i (-Im_j) ... conj":    real - 1j * (ein(re, wneg(im)) + ein(im, re)),
    }
    print(f"\n{'4-bit negation model (stored S)':<36} {'off':>10} {'d_lo':>10} {'d_up':>10} {'exact%':>8}")
    for name, S in models.items():
        for tname, transpose in (("[a,b]", False), ("[b,a]", True)):
            H = np.empty_like(B)
            for a in range(2):
                for b in range(2):
                    i, j = (b, a) if transpose else (a, b)
                    H[:, :, a, b] = S[:, ry * 2 + i, rx * 2 + j]
            e = np.abs(B - H)
            print(f"{name[:28] + ' ' + tname:<36} {e[:, off].max():10.3g} "
                  f"{e[:, diag, 1, 0].max():10.3g} {e[:, diag, 0, 1].max():10.3g} "
                  f"{100 * np.mean(e == 0):8.2f}")

    # Where do the raw-conj mismatches sit?
    H = np.empty_like(B)
    for a in range(2):
        for b in range(2):
            H[:, :, a, b] = np.conj(refs["raw"][:, ry * 2 + a, rx * 2 + b])
    bad = np.abs(B - H) > 0
    bad_ch = np.flatnonzero(bad.any(axis=(1, 2, 3)))
    ch8 = np.flatnonzero(((re == -8) | (im == -8)).any(axis=(0, 2)))
    print(f"\nraw-conj mismatches: {bad.sum()} entries in {bad_ch.size} channels: {bad_ch[:20].tolist()}")
    print(f"channels containing -8 samples: {ch8.size}: {ch8[:20].tolist()}")
    print(f"mismatching channels without any -8: {np.setdiff1d(bad_ch, ch8).tolist()[:20]}")
    d = (B - H)[bad]
    print(f"real part of mismatches: max {np.abs(d.real).max():.3g}; "
          f"imag max {np.abs(d.imag).max():.3g}; imag % 16 == 0: {np.all(d.imag % 16 == 0)}")

    up, lo = B[:, diag, 0, 1], B[:, diag, 1, 0]
    print("\nupper entry of diagonal blocks (d_up):")
    print(f"  max |d_up|              = {np.abs(up).max():.3g}")
    print(f"  max |d_up - d_lo|       = {np.abs(up - lo).max():.3g}")
    print(f"  max |d_up - conj(d_lo)| = {np.abs(up - np.conj(lo)).max():.3g}")
    k = np.unravel_index(np.argmax(np.abs(refs['raw'][:, 1, 0])), refs['raw'][:, 1, 0].shape)[0]
    print(f"\nsample (channel {k}, elements 0-1):")
    print(f"  R_np[1,0] = {refs['raw'][k, 1, 0]:.0f}   R_np[0,1] = {refs['raw'][k, 0, 1]:.0f}")
    print(f"  B[0][1,0] = {B[k, 0, 1, 0]:.0f}   B[0][0,1] = {B[k, 0, 0, 1]:.0f}")


if __name__ == "__main__":
    main()
