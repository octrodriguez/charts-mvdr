#!/usr/bin/env python
"""How often MVDR weights must be recomputed: excess output power of stale weights.

Reads a covariance cube saved by 16_rfi_census.py extract (best with --stride 1, so that
lags start at one correlator frame) and a gains file, calibrates the cube, trains MVDR
weights for a fixed look direction on the --n-avg frames before every frame, and measures

    E_k = w(t - k)^H R_t w(t - k) / w(t)^H R_t w(t)

for weights k frames old (charts_mvdr.cadence). Per band it prints the median over channels
and time of E_k in dB for each lag, the 90th percentile over time, the largest lag that stays
below --threshold-db, and the suppression of fresh MVDR weights relative to the DFT beam
(how much MVDR is doing there in the first place).

Look direction: 'zenith' (all-ones steering vector: independent of positions and sign) or
'l,m' (needs --positions, logical order ENU metres, and the steering --sign; -1 for real
cubes, see README Conventions). A moving look direction is not supported (see cadence.py).

Example:
    python 21_weight_cadence.py --vis vis_day.npz --gains gains_sun_<window>.npz \\
        --bands 360-380 414-418 423-427 325-355 --out cadence_day
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from charts_mvdr import cadence, nulling, real_io
from charts_mvdr.beamforming import steering_vectors
from charts_mvdr.tracking import LOADING_REFS

DEFAULT_LAGS = (1, 2, 3, 6, 12, 30, 60, 120, 180, 360)


def parse_bands(items):
    out = []
    for s in items:
        lo, hi = (float(x) for x in s.split("-"))
        if not hi > lo:
            raise ValueError(f"band {s!r}: high edge must exceed low edge")
        out.append((lo, hi))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vis", required=True, help="Cube .npz from 16_rfi_census.py extract.")
    real_io.add_gains_arguments(ap)
    ap.add_argument("--look", default="zenith", help="'zenith' or 'l,m' (fixed in time).")
    ap.add_argument("--positions", default=None, help="Positions .npy (n, 3) ENU m, logical order (for 'l,m').")
    ap.add_argument("--sign", type=int, choices=(1, -1), default=-1,
                    help="Steering sign for 'l,m'; -1 for real cubes (README Conventions).")
    ap.add_argument("--n-avg", type=int, default=6, help="Frames averaged to train each weight set.")
    ap.add_argument("--loading", type=float, nargs="+", default=[10.0, 0.1],
                    help="Diagonal loading(s) in units of the reference eigenvalue.")
    ap.add_argument("--loading-ref", default="median", choices=list(LOADING_REFS))
    ap.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS), help="Lags in frames.")
    ap.add_argument("--bands", nargs="+", default=["360-380", "414-418", "423-427", "325-355"],
                    help="Bands lo-hi in MHz.")
    ap.add_argument("--threshold-db", type=float, default=0.1)
    ap.add_argument("--out", required=True, help="Output prefix (.npz and .json are written).")
    args = ap.parse_args(argv)

    try:
        bands = parse_bands(args.bands)
        g, ants, _ = real_io.gains_from_args(args)
        V, freqs, times, _ = nulling.load_cube(args.vis, ants)
        if g.shape[0] != len(freqs):
            raise ValueError(f"gains have {g.shape[0]} channels, the cube {len(freqs)}")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    n_frames, n_chan, n = V.shape[:3]
    if n_frames < args.n_avg + 2:
        print(f"error: {n_frames} frames, need at least n_avg + 2 = {args.n_avg + 2}", file=sys.stderr)
        return 1
    dt = np.diff(times)
    cad = float(np.median(dt))
    if np.any(dt > 1.5 * cad):
        print(f"warning: {int(np.sum(dt > 1.5 * cad))} gap(s) longer than 1.5 frames: lags are counted "
              "in frames and are only approximate times across gaps")
    Vc, ok = nulling.calibrate(V, g, freqs)
    if args.look == "zenith":
        a = np.ones((n_chan, n), np.complex128)
        look = (0.0, 0.0)
    else:
        if not args.positions:
            print("error: --look l,m needs --positions", file=sys.stderr)
            return 1
        look = tuple(float(x) for x in args.look.split(","))
        pos = np.load(args.positions)
        x, y = pos[ants, 0], pos[ants, 1]
        a = np.stack([steering_vectors(look[0], look[1], f * 1e6, x, y)[0] for f in freqs])
        if args.sign < 0:
            a = a.conj()
    lags = sorted({int(k) for k in args.lags if 0 < k < n_frames - args.n_avg})
    lag_s = np.array(lags) * cad
    print(f"{args.vis}: {n_frames} frames of {cad:.2f} s, {int(ok.sum())}/{n_chan} channels calibrated, "
          f"antennas {ants}, look {args.look}, training {args.n_avg} frames ({args.n_avg * cad:.0f} s)")
    wd = a / n
    p_dft = cadence.output_power(np.broadcast_to(wd, Vc.shape[:3]), Vc)
    result = {"lags_frames": lags, "lags_s": lag_s.tolist(), "bands": args.bands, "loadings": {}}
    for lo_val in args.loading:
        W = cadence.training_weights(Vc, a, args.n_avg, lo_val, args.loading_ref)
        E = cadence.excess_power(Vc, W, lags)
        med, p90, nch = cadence.band_excess_db(E, freqs, bands)
        with np.errstate(divide="ignore", invalid="ignore"):
            supp = 10 * np.log10(cadence.output_power(W, Vc) / p_dft)        # (n_frames, n_chan)
        print(f"\nloading {lo_val:g} x {args.loading_ref} eigenvalue")
        print("  lag     " + "".join(f"{b:>18}" for b in args.bands))
        print("  " + " " * 8 + "".join(f"{'(' + str(c) + ' ch)':>18}" for c in nch))
        for i, s in enumerate(lag_s):
            print(f"  {s:6.0f} s" + "".join(f"{med[i, j]:+8.3f} ({p90[i, j]:+6.2f})" for j in range(len(bands))))
        row_lag, row_supp, entry = [], [], {}
        for j, (lo, hi) in enumerate(bands):
            sel = (freqs >= lo) & (freqs < hi) & ok
            best = cadence.max_lag_below(lag_s, med[:, j], args.threshold_db)
            sv = float(np.nanmedian(supp[:, sel])) if sel.any() else np.nan
            row_lag.append(best)
            row_supp.append(sv)
            entry[args.bands[j]] = {"median_db": med[:, j].tolist(), "p90_db": p90[:, j].tolist(),
                                    "max_lag_s_below_threshold": None if np.isnan(best) else float(best),
                                    "mvdr_minus_dft_db": sv, "n_chan": nch[j]}
        print(f"  longest lag with median excess < {args.threshold_db:g} dB:"
              + "".join(f"{('> ' + format(lag_s[-1], '.0f') + ' s') if b == lag_s[-1] else ('< ' + format(lag_s[0], '.0f') + ' s' if np.isnan(b) else format(b, '.0f') + ' s'):>18}"
                        for b in row_lag))
        print("  fresh MVDR minus DFT power [dB]:  " + "".join(f"{v:+18.1f}" for v in row_supp))
        result["loadings"][f"{lo_val:g}"] = entry
        np.savez(f"{args.out}_loading{lo_val:g}.npz", E_band_median_db=med, E_band_p90_db=p90, lags_s=lag_s,
                 bands=np.array(args.bands), times=times)
    result.update(vis=str(Path(args.vis).resolve()), gains=str(Path(args.gains).resolve()), look=list(look),
                  n_avg=args.n_avg, loading_ref=args.loading_ref, threshold_db=args.threshold_db,
                  frame_s=cad, antennas=ants)
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=1))
    print(f"\nsaved {args.out}.json and {args.out}_loading*.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
