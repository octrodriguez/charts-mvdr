#!/usr/bin/env python3
"""
17_real_nulling.py — Nulling test of the stage algorithm on real covariances.

Reads a covariance cube saved by 16_rfi_census.py extract (vis (M, M, n_freq,
n_frames), times, freqs, ants), complex gains from cal_pipeline_gpu.py
(--save-avg-gains, columns in the order of --gain-antennas) and antenna
positions (ENU metres, logical order). For each frame it forms DFT and MVDR
beams towards a look direction (the Sun tracked frame by frame, zenith, or a
fixed l,m) with the rules of the Kotekan stage (charts_mvdr.tracking.run_tracking:
blind local window, loading 10 x median eigenvalue, gains G^-1 R G^-H).

Per channel it reports the median over frames of the MVDR/DFT power ratio and the
WNG. In channels with interference outside the look direction the ratio is the
rejection of the adaptive beam; in clean channels dominated by the look source
it should stay near 0 dB (no self-nulling). Channels whose gain was interpolated
(--interp-gains; usually the RFI channels flagged in the calibration) are marked.

The steering phase convention is checked on the data (--sign auto): the DFT power
towards the Sun is compared with the power towards the mirrored direction in the
--clean channels, and the larger one is used.

Usage:
    python 17_real_nulling.py --vis <vis.npz> --gains <gains.npy> --gain-antennas 0,5,32,37 \\
        --positions <positions.npy> --look sun --interp-gains \\
        --rfi 362-380 415-428 400.5-402.5 --clean 381.5-399.5 403-414 --out <out_dir>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from charts_mvdr import tracking


def parse_ranges(items):
    out = []
    for item in items or []:
        lo, hi = (float(x) for x in item.split("-"))
        out.append((lo, hi))
    return out


def in_ranges(freqs_mhz, ranges):
    m = np.zeros(freqs_mhz.shape, dtype=bool)
    for lo, hi in ranges:
        m |= (freqs_mhz >= lo) & (freqs_mhz < hi)
    return m


def sun_lm(times, positions):
    from charts_constants import CHARTS_ANTENNA_REF_LOCATION
    from charts_utils.cal.coordinates import CHARTSArray

    arr = CHARTSArray(*CHARTS_ANTENNA_REF_LOCATION, positions)
    alt, az = arr.get_source_altaz(np.asarray(times, dtype=float), "sun")
    alt, az = np.radians(alt), np.radians(az)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az)], axis=1), np.degrees(alt)


def choose_sign(vis, freqs_hz, x, y, look, gains, chans):
    """Median DFT power towards the look direction for both phase conventions."""
    out = {}
    for sign in (1, -1):
        r = tracking.run_tracking(vis[:, :, chans], freqs_hz[chans], x, y, look,
                                  gains=None if gains is None else gains[chans],
                                  half_window=2, guard=0, min_frames=10**9, sign=sign)
        out[sign] = float(np.nanmedian(r["power"][0]))
    return out


def summary(label, mask, ratio_db, wng_db):
    if not mask.any():
        return f"  {label:<28} no channels"
    r, w = ratio_db[mask], wng_db[mask]
    return (f"  {label:<28} n={mask.sum():3d}  MVDR/DFT median {np.nanmedian(r):+6.1f} dB "
            f"(p10 {np.nanpercentile(r, 10):+6.1f}, p90 {np.nanpercentile(r, 90):+6.1f})  "
            f"WNG median {np.nanmedian(w):+5.1f} dB")


def plot(path, freqs, ratio_db, wng_db, interp, rfi, clean, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for axi in ax:
        for lo, hi in rfi:
            axi.axvspan(lo, hi, color="tab:red", alpha=0.12, lw=0)
        for lo, hi in clean:
            axi.axvspan(lo, hi, color="tab:green", alpha=0.10, lw=0)
    ax[0].plot(freqs, ratio_db, ".", ms=3, color="k", label="calibrated gain")
    ax[0].plot(freqs[interp], ratio_db[interp], ".", ms=3, color="tab:orange", label="interpolated gain")
    ax[0].axhline(0, color="0.5", lw=0.8)
    ax[0].set_ylabel("MVDR / DFT power [dB]\n(median over frames)")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title(title, fontsize=9, loc="left")
    ax[1].plot(freqs, wng_db, ".", ms=3, color="tab:blue")
    ax[1].set_ylabel("WNG [dB]")
    ax[1].set_xlabel("Frequency [MHz]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(description="Nulling test of the stage algorithm on real covariances.")
    p.add_argument("--vis", required=True, help=".npz from 16_rfi_census.py extract.")
    p.add_argument("--gains", default=None, help="Complex gains .npy (n_freq, n_ant).")
    p.add_argument("--gain-antennas", default=None, help="Logical antennas of the gains columns, e.g. 0,5,32,37.")
    p.add_argument("--antennas", default=None, help="Antennas to use (default: --gain-antennas or all).")
    p.add_argument("--interp-gains", action="store_true", help="Interpolate gains over unsolved channels.")
    p.add_argument("--positions", required=True, help="Positions .npy (n, 3), ENU metres, logical order.")
    p.add_argument("--look", default="sun", help="'sun', 'zenith' or 'l,m'.")
    p.add_argument("--sign", default="auto", choices=["auto", "1", "-1"], help="Steering phase convention.")
    p.add_argument("--half-window", type=int, default=10)
    p.add_argument("--guard", type=int, default=1)
    p.add_argument("--loading", type=float, default=10.0, help="Loading in units of the median eigenvalue.")
    p.add_argument("--rfi", nargs="*", default=[], help="MHz ranges with interference (summary).")
    p.add_argument("--clean", nargs="*", default=[], help="MHz ranges dominated by the look source (summary, sign).")
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    z = np.load(args.vis)
    vis, freqs, times = z["vis"], np.asarray(z["freqs"], float), np.asarray(z["times"], float)
    cube_ants = [int(a) for a in z["ants"]]
    gain_ants = [int(a) for a in args.gain_antennas.split(",")] if args.gain_antennas else None
    ants = ([int(a) for a in args.antennas.split(",")] if args.antennas else gain_ants or cube_ants)
    missing = [a for a in ants if a not in cube_ants or (gain_ants is not None and a not in gain_ants)]
    if missing:
        print(f"error: antennas {missing} not in the cube {cube_ants} or the gains {gain_ants}", file=sys.stderr)
        return 1
    idx = [cube_ants.index(a) for a in ants]
    vis = vis[np.ix_(idx, idx)]

    gains, interp = None, np.zeros((len(freqs), len(ants)), dtype=bool)
    if args.gains:
        if gain_ants is None:
            print("error: --gains needs --gain-antennas", file=sys.stderr)
            return 1
        g = np.load(args.gains)
        if g.ndim != 2 or g.shape[0] != len(freqs) or g.shape[1] != len(gain_ants):
            print(f"error: gains {g.shape} do not match ({len(freqs)}, {len(gain_ants)})", file=sys.stderr)
            return 1
        g = g[:, [gain_ants.index(a) for a in ants]]
        if args.interp_gains:
            g, interp = tracking.interpolate_gains(g, freqs)
        gains = g
    interp_any = interp.any(axis=1)

    pos = np.load(args.positions)
    x, y = pos[ants, 0], pos[ants, 1]
    alt = None
    if args.look == "sun":
        look, alt = sun_lm(times, pos)
    elif args.look == "zenith":
        look = np.zeros((len(times), 2))
    else:
        l0, m0 = (float(v) for v in args.look.split(","))
        look = np.tile([l0, m0], (len(times), 1))
    freqs_hz = freqs * 1e6
    rfi, clean = parse_ranges(args.rfi), parse_ranges(args.clean)

    if args.sign == "auto":
        cm = in_ranges(freqs, clean) & ~interp_any
        if not cm.any():
            print("error: --sign auto needs --clean channels with calibrated gains", file=sys.stderr)
            return 1
        pw = choose_sign(vis, freqs_hz, x, y, look, gains, np.flatnonzero(cm))
        sign = 1 if pw[1] >= pw[-1] else -1
        print(f"steering sign: DFT power towards look, +1: {pw[1]:.4g}, -1: {pw[-1]:.4g} "
              f"(ratio {10 * np.log10(max(pw.values()) / min(pw.values())):.1f} dB) -> {sign:+d}")
    else:
        sign = int(args.sign)

    res = tracking.run_tracking(vis, freqs_hz, x, y, look, gains=gains, half_window=args.half_window,
                                guard=args.guard, loading=args.loading, sign=sign)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_db = np.nanmedian(10 * np.log10(res["power"][1] / res["power"][0]), axis=0)
        wng_db = np.nanmedian(10 * np.log10(res["wng"]), axis=0)
    n_used = int(np.sum(res["n_window"] >= args.half_window - args.guard))
    print(f"antennas {ants}, {len(times)} frames ({n_used} with a full window), look {args.look}"
          + (f", Sun elevation {alt.min():.0f}-{alt.max():.0f} deg" if alt is not None else ""))
    rm, cm = in_ranges(freqs, rfi), in_ranges(freqs, clean)
    print(summary("rfi, calibrated gain", rm & ~interp_any, ratio_db, wng_db))
    print(summary("rfi, interpolated gain", rm & interp_any, ratio_db, wng_db))
    print(summary("clean, calibrated gain", cm & ~interp_any, ratio_db, wng_db))
    print(summary("clean, interpolated gain", cm & interp_any, ratio_db, wng_db))
    for lo, hi in rfi:
        m = (freqs >= lo) & (freqs < hi)
        print(summary(f"rfi {lo:g}-{hi:g} MHz", m, ratio_db, wng_db))

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "real_nulling.npz"), freqs=freqs, times=times,
                        ants=np.array(ants), look=look, sign=sign, power=res["power"], wng=res["wng"],
                        ratio_db=ratio_db, wng_db=wng_db, interp=interp_any,
                        params=np.array([args.half_window, args.guard, args.loading]))
    plot(os.path.join(args.out, "real_nulling.png"), freqs, ratio_db, wng_db, interp_any, rfi, clean,
         f"antennas {ants}, look {args.look}, H = {args.half_window}, loading {args.loading:g} x median eig")
    print(f"saved {args.out}/real_nulling.npz and real_nulling.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
