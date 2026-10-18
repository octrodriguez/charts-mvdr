#!/usr/bin/env python
"""Voltage-domain beamforming of a dispersed burst: waterfalls and S/N.

For each channel f the weights w(f) towards the burst are computed from the
correlation dumps, and the baseband voltages are beamformed sample by sample,

    y(t, f) = w(f)^H v(t, f),     P(t, f) = |y(t, f)|^2,

then averaged in bins of --tbin samples. Weights:
  dft        w = d / N (conventional beam)
  kl_local   MVDR / KL with F from the off-pulse frames within --local-s of the
             burst arrival in that channel (excluding on frames and a guard)
  kl_global  MVDR / KL with F from the frames outside the whole burst window
KL weights use diagonal loading --loading in units of the noise floor of F.

Per channel the power is normalized with the off-local bins of that channel,
z = (P - mean) / std, so the waterfall is in units of the per-bin noise.
Outputs:
  per-channel S/N   sum of z over the on-pulse bins / sqrt(number of bins)
  dedispersed S/N   channels shifted by their dispersion delay, z summed over the
                    unflagged channels, boxcar of ~FWHM, peak / sqrt(width * Nch)
  waterfall.png     z(t, f) for each beam plus the dedispersed series
  channel_snr.png   per-channel S/N for each beam and the KL / DFT ratio
  waterfall.npz     per-channel S/N, dedispersed series and S/N, flags, and
                    (with --save-waterfall) the binned power
Channels whose leading tail is cut by the simulator (event start inside the
pulse) or without off-local frames are flagged and left out of the sums.

The per-bin noise is assumed independent between bins and channels (true for
the receiver noise; residual RFI breaks it). For a paired run on a window
without RFI, compare the per-channel S/N of the npz files (same frames and noise).

Example:
  python scripts/09_beamform_waterfall.py --window-dir WINDOW_DIR --corr-dir CORR_DIR \\
      --out OUT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import dft_weights, mvdr_weights, steering_vectors
from charts_mvdr.data_io import (list_correlation_dumps, load_events_file, load_window_dir_meta,
                                 read_astron_correlation, read_baseband_frame)
from charts_mvdr.transients import arrival_times_s, burst_from_events, dispersion_delay_s

SAMPLE_S = 10e-6 / 3


def dilate(mask, n):
    """Extend True runs along axis 0 by n frames on each side."""
    out = mask.copy()
    for k in range(1, n + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-dir", type=Path, required=True,
                    help="baseband frames, <window>_meta.h5 and the events file")
    ap.add_argument("--corr-dir", type=Path, required=True,
                    help="correlation dumps of the same frames (for the weights)")
    ap.add_argument("--channels", type=int, nargs="+", default=None, help="default: all")
    ap.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"),
                    help="burst direction (default: from the frb event)")
    ap.add_argument("--look-offset", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("DL", "DM"))
    ap.add_argument("--event-id", default=None)
    ap.add_argument("--weights", nargs="+", default=["dft", "kl_local", "kl_global"],
                    choices=["dft", "kl_local", "kl_global"])
    ap.add_argument("--loading", type=float, default=10.0,
                    help="KL diagonal loading in units of the noise floor of F")
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--local-s", type=float, default=0.05)
    ap.add_argument("--guard-frames", type=int, default=1)
    ap.add_argument("--tbin", type=int, default=16, help="samples per time bin")
    ap.add_argument("--flag-channels", type=int, nargs="+", default=[])
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--save-waterfall", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("beamform_waterfall"))
    args = ap.parse_args()
    if args.samples % args.tbin:
        raise SystemExit("--tbin must divide --samples")

    meta = load_window_dir_meta(args.window_dir)
    if "frame_time_s" not in meta:
        raise SystemExit("frame timestamps are needed in the window metadata")
    x, y = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"]
    fmhz = np.asarray(meta["frequencies_mhz"], dtype=float)
    N, nf = x.size, fmhz.size
    ch = np.arange(nf) if args.channels is None else np.asarray(args.channels, dtype=int)
    try:
        burst = burst_from_events(load_events_file(args.window_dir), args.event_id)
    except ValueError as err:
        raise SystemExit(str(err))
    look = tuple(args.look) if args.look else (burst["l"], burst["m"])
    steer = (look[0] + args.look_offset[0], look[1] + args.look_offset[1])
    dm, w_s, t_top = burst["dm"], burst["width_s"], burst["t_top_s"]
    frame_s = args.samples * SAMPLE_S
    bin_s = args.tbin * SAMPLE_S
    f_top = float(fmhz.max())
    sweep = float(dispersion_delay_s(fmhz.min(), dm, f_top))

    dumps = list_correlation_dumps(args.corr_dir)
    frames = list_correlation_dumps(args.window_dir)
    t_of = dict(zip(np.asarray(meta["frame_out_idx"]).astype(int), meta["frame_time_s"]))
    idx = sorted(k for k in t_of if k in dumps and k in frames)
    times = np.array([t_of[k] for k in idx])
    if not idx:
        raise SystemExit("no frames with both baseband and correlation dumps")
    print(f"{len(idx)} frames, {N} antennas, {ch.size} channels; burst {burst['id']}: DM {dm:g}, "
          f"sigma {w_s * 1e3:g} ms, top arrival {t_top:g} s, sweep {sweep * 1e3:.1f} ms; "
          f"weights steered to ({steer[0]:+.4f}, {steer[1]:+.4f}), loading {args.loading:g}")

    # Frame masks (frames, channels)
    t_arr = arrival_times_s(fmhz[ch], dm, t_top, f_top)
    half = args.n_sigma * w_s
    tt = times[:, None]
    on = (tt <= t_arr + half) & (tt + frame_s >= t_arr - half)
    local = (np.abs(tt + frame_s / 2 - t_arr) <= args.local_s) & ~dilate(on, args.guard_frames)
    glob = (times + frame_s < t_top - half) | (times > t_top + sweep + half)

    # Weights covariances, streamed over the dumps
    F_loc = np.zeros((ch.size, N, N), dtype=np.complex128)
    F_glob = np.zeros((ch.size, N, N), dtype=np.complex128)
    for i, k in enumerate(idx):
        if not (local[i].any() or glob[i]):
            continue
        R = read_astron_correlation(dumps[k], nf, N)[ch] / args.samples
        if glob[i]:
            F_glob += R
        F_loc[local[i]] += R[local[i]]
    n_loc = local.sum(0)
    F_loc /= np.maximum(n_loc, 1)[:, None, None]
    F_glob /= max(int(glob.sum()), 1)

    frame0 = np.floor(t_top / frame_s) * frame_s
    flag = np.zeros(ch.size, bool)
    flag |= t_arr - half < frame0                       # leading tail cut by the simulator
    flag |= n_loc == 0
    flag |= np.isin(ch, args.flag_channels)
    W = {}
    for name in args.weights:
        Wn = np.empty((ch.size, N), dtype=np.complex128)
        for c in range(ch.size):
            d = steering_vectors(*steer, fmhz[ch[c]] * 1e6, x, y)[0]
            if name == "dft":
                Wn[c] = dft_weights(d)[0]
            else:
                F = F_loc[c] if name == "kl_local" else F_glob[c]
                Wn[c] = (mvdr_weights(F, d, loading=args.loading, loading_ref="noise")[0]
                         if (name != "kl_local" or n_loc[c]) else dft_weights(d)[0])
        W[name] = Wn.conj().astype(np.complex64)

    # Time grid covering the off-local and on frames of every channel
    t0 = max(times.min(), t_top - args.local_s - frame_s)
    t1 = min(times.max() + frame_s, t_top + sweep + args.local_s + frame_s)
    k0 = int(np.floor(t0 / bin_s))
    nb = int(np.ceil(t1 / bin_s)) - k0
    bins_per_frame = args.samples // args.tbin
    P = {n: np.full((nb, ch.size), np.nan, dtype=np.float32) for n in W}
    bin_frame = np.full(nb, -1, dtype=int)
    for i, k in enumerate(idx):
        b0 = int(round(times[i] / bin_s)) - k0
        if b0 + bins_per_frame <= 0 or b0 >= nb:
            continue
        V = read_baseband_frame(frames[k], nf, N, args.samples)[:, ch, :]
        lo, hi = max(0, -b0), min(bins_per_frame, nb - b0)
        bin_frame[b0 + lo:b0 + hi] = i
        for n, Wc in W.items():
            yv = np.einsum("tcn,cn->tc", V, Wc)
            pw = (np.abs(yv) ** 2).reshape(bins_per_frame, args.tbin, ch.size).mean(1)
            P[n][b0 + lo:b0 + hi] = pw[lo:hi]
    t_bin = (k0 + np.arange(nb)) * bin_s
    covered = bin_frame >= 0
    print(f"time grid: {nb} bins of {bin_s * 1e6:.1f} us over {t0:.4f}-{t1:.4f} s, "
          f"{covered.mean() * 100:.0f} % covered by frames")

    # Normalization, per-channel S/N, dedispersion
    off_bin = np.zeros((nb, ch.size), bool)
    off_bin[covered] = local[bin_frame[covered]]
    on_bin = np.abs(t_bin[:, None] + bin_s / 2 - t_arr) <= half
    shift = np.rint((t_arr - t_top) / bin_s).astype(int)
    width_b = max(1, int(round(2.355 * w_s / bin_s)))
    good = ~flag
    res = {}
    for n in W:
        Z = np.full_like(P[n], np.nan)
        for c in range(ch.size):
            off = P[n][off_bin[:, c], c]
            if off.size > 2:
                Z[:, c] = (P[n][:, c] - off.mean()) / off.std()
        snr_ch = np.array([np.nansum(Z[on_bin[:, c], c]) / np.sqrt(max(1, np.sum(
            np.isfinite(Z[on_bin[:, c], c])))) for c in range(ch.size)])
        Zd = np.full_like(Z, np.nan)
        for c in np.flatnonzero(good):
            s = shift[c]
            if 0 <= s < nb:
                Zd[:nb - s, c] = Z[s:, c]
        n_ok = np.sum(np.isfinite(Zd), axis=1)
        series = np.where(n_ok > 0, np.nansum(Zd, axis=1) / np.sqrt(np.maximum(n_ok, 1)), np.nan)
        box = np.convolve(np.nan_to_num(series), np.ones(width_b), mode="same") / np.sqrt(width_b)
        res[n] = {"Z": Z, "snr_ch": snr_ch, "series": series, "box": box,
                  "dd_snr": float(np.nanmax(box)), "t_peak": float(t_bin[np.nanargmax(box)])}

    print(f"\nflagged channels: {np.flatnonzero(flag).size} "
          f"({', '.join(str(c) for c in ch[flag][:12])}{' ...' if flag.sum() > 12 else ''})")
    print(f"{'beam':<10} {'dedispersed S/N':>16} {'peak at [s]':>12} {'median ch S/N':>14}")
    for n, r in res.items():
        print(f"{n:<10} {r['dd_snr']:16.1f} {r['t_peak']:12.4f} "
              f"{np.nanmedian(r['snr_ch'][good]):14.2f}")
    if "dft" in res:
        for n in res:
            if n != "dft":
                g = 10 * np.log10(np.maximum(res[n]["snr_ch"], 1e-3)
                                  / np.maximum(res["dft"]["snr_ch"], 1e-3))
                worst = ch[good][np.argsort(g[good])[:5]]
                print(f"{n} / dft per-channel S/N gain: median {np.nanmedian(g[good]):+.2f} dB, "
                      f"max {np.nanmax(g[good]):+.2f} dB (ch {ch[good][np.nanargmax(g[good])]}), "
                      f"lowest in ch {list(worst)}")

    args.out.mkdir(parents=True, exist_ok=True)
    save = {"channels": ch, "freq_mhz": fmhz[ch], "flag": flag, "look": np.array(look),
            "steer": np.array(steer), "dm": dm, "width_s": w_s, "t_top": t_top,
            "loading": args.loading, "t_bin": t_bin, "beams": np.array(list(W))}
    for n, r in res.items():
        save[f"{n}_snr_ch"], save[f"{n}_series"] = r["snr_ch"], r["series"]
        save[f"{n}_dd_snr"] = r["dd_snr"]
        if args.save_waterfall:
            save[f"{n}_power"] = P[n]
    np.savez_compressed(args.out / "waterfall.npz", **save)

    nb_ = len(res)
    fig, axes = plt.subplots(2, nb_, figsize=(5 * nb_, 7.5), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    tms = (t_bin - t_top) * 1e3
    for j, (n, r) in enumerate(res.items()):
        ax = axes[0, j]
        ax.imshow(r["Z"].T, aspect="auto", origin="lower", cmap="viridis", vmin=-2, vmax=8,
                  extent=[tms[0], tms[-1] + bin_s * 1e3, fmhz[ch[0]], fmhz[ch[-1]]],
                  interpolation="nearest")
        ax.set_title(f"{n}: dedispersed S/N {r['dd_snr']:.1f}", fontsize=10)
        ax.set_ylabel("frequency [MHz]")
        axes[1, j].plot(tms, r["box"], lw=0.8)
        axes[1, j].set_xlabel("time from top-channel arrival [ms]")
        axes[1, j].set_ylabel("dedispersed S/N")
        axes[1, j].grid(alpha=0.3)
    fig.suptitle(f"burst {burst['id']}, DM {dm:g}: beam power z-score (bins of "
                 f"{bin_s * 1e6:.0f} us)", fontsize=10)
    fig.tight_layout(); fig.savefig(args.out / "waterfall.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for n, r in res.items():
        axes[0].plot(fmhz[ch][good], r["snr_ch"][good], ".", ms=3, label=n)
        if n != "dft" and "dft" in res:
            axes[1].plot(fmhz[ch][good], 10 * np.log10(np.maximum(r["snr_ch"][good], 1e-3)
                                                       / np.maximum(res["dft"]["snr_ch"][good],
                                                                    1e-3)),
                         ".", ms=3, label=f"{n} / dft")
    axes[0].set_ylabel("per-channel S/N"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_ylabel("S/N gain [dB]"); axes[1].set_xlabel("frequency [MHz]")
    axes[1].axhline(0, color="0.4", lw=0.8); axes[1].grid(alpha=0.3)
    if len(res) > 1:
        axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / "channel_snr.png", dpi=130); plt.close(fig)
    print(f"\noutputs -> {args.out}")


if __name__ == "__main__":
    main()
