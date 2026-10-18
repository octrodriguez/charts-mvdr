#!/usr/bin/env python
"""Signal and S/N of a dispersed transient through DFT and MVDR beamformers.

Works on correlation dumps (one covariance per frame) of a window with a
dispersed burst of known DM, width and top-channel arrival time (from the
events file, or given). For each channel the frames are split with the
dispersion delay (transients.on_pulse_mask):

  on        frames that overlap the burst in that channel (+- n_sigma widths)
  off-local frames within --local-s of the burst arrival in that channel,
            excluding the on frames and --guard-frames on each side
  off-global frames outside the whole burst window
            [t_top - n_sigma w, t_top + sweep + n_sigma w] (frame overlap)

Signal through a beamformer w (any weights, fixed):
    S(w) = w^H (R_on - R_off-local) w,    noise + interference  Q(w) = w^H R_off-local w,
so S/N(w) = S / Q. Interference that is present both on and off pulse cancels in
S; the off-local frames are close in time, so transient RFI active around the
burst is in both. The power budget of 05 is not used: interferers missing from
the events file (e.g. a bright sky source seen through the sidelobes) would be
unmodelled.

Weights (towards --look, optionally offset by --look-offset to emulate a
pointing error; S/N is always measured on the true burst):
  dft         w = d / N
  on          MVDR with R_on (signal-in: the burst is in the training data)
  all         MVDR with the average of all frames
  off_global  MVDR with F from the off-global frames (KL, off-pulse)
  off_local   MVDR with F from the off-local frames (KL, off-pulse near the burst)
Each MVDR type for every --loading (units of the noise floor of its covariance).

Reported per channel and weights: S / S_DFT (signal retained relative to the
DFT), S/N gain over the DFT [dB], WNG = N ||w||^2 [dB] and the response towards
the interferers of the events file in that channel and towards --extra-dirs.

Expected behaviour: off_global cannot null interference that only exists while
the burst is on (it is not in its training frames); on loses signal by finite K
(rho ~ 1 / (1 + SINR_opt (N - 1) / K)) and by self-nulling under pointing errors.

Example:
  python scripts/08_transient_snr.py --corr-dir CORR_DIR --window-dir WINDOW_DIR \\
      --channels 100 147 192 --look -0.10 0.25 --extra-dirs 0.4677 0.4801 --out OUT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import (beam_response, dft_weights, mvdr_weights,
                                     steering_vectors)
from charts_mvdr.data_io import (interferers_by_channel, list_correlation_dumps,
                                 load_events_file, load_window_dir_meta,
                                 read_astron_correlation)
from charts_mvdr.transients import arrival_times_s, dispersion_delay_s, on_pulse_mask

KINDS = ("on", "all", "off_global", "off_local")


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-30))


def burst_params(args, events):
    """(dm, width_s, t_top_s, event_id) from the arguments or the events file."""
    if args.dm is not None:
        if args.width_ms is None or args.t_top is None:
            raise SystemExit("--dm needs --width-ms and --t-top")
        return args.dm, args.width_ms * 1e-3, args.t_top, "given"
    frbs = [e for e in events if e.get("event_type") == "frb"
            and (args.event_id is None or e.get("event_id") == args.event_id)]
    if not frbs:
        raise SystemExit("no frb event in the events file; give --dm --width-ms --t-top")
    e = frbs[0]
    return float(e["dm"]), float(e["pulse_width_ms"]) * 1e-3, float(e["t_start_s"]), e["event_id"]


def dilate(mask, n):
    """Extend True runs of a 1-D frame mask by n frames on each side."""
    out = mask.copy()
    for k in range(1, n + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corr-dir", type=Path, required=True)
    ap.add_argument("--window-dir", type=Path, required=True)
    ap.add_argument("--channels", type=int, nargs="+", required=True)
    ap.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"),
                    help="burst direction (default: from the frb event)")
    ap.add_argument("--look-offset", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("DL", "DM"), help="steer the weights to look + offset")
    ap.add_argument("--event-id", default=None, help="frb event to use (default: first)")
    ap.add_argument("--dm", type=float, default=None)
    ap.add_argument("--width-ms", type=float, default=None, help="envelope sigma [ms]")
    ap.add_argument("--t-top", type=float, default=None, help="top-channel arrival [s]")
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--local-s", type=float, default=0.05,
                    help="half-width of the off-local window around the arrival [s]")
    ap.add_argument("--guard-frames", type=int, default=1)
    ap.add_argument("--loading", type=float, nargs="+", default=[0.0, 10.0])
    ap.add_argument("--extra-dirs", type=float, nargs="+", default=[],
                    help="extra directions (l m pairs) for the response table")
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--out", type=Path, default=Path("transient_snr"))
    args = ap.parse_args()
    if len(args.extra_dirs) % 2:
        raise SystemExit("--extra-dirs takes l m pairs")
    extra = list(zip(args.extra_dirs[::2], args.extra_dirs[1::2]))

    meta = load_window_dir_meta(args.window_dir)
    if "frame_time_s" not in meta:
        raise SystemExit("frame timestamps are needed in the window metadata")
    x, y = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"]
    fmhz = np.asarray(meta["frequencies_mhz"], dtype=float)
    N = x.size
    events = load_events_file(args.window_dir)
    rfi = interferers_by_channel(events)
    dm, width_s, t_top, eid = burst_params(args, events)
    if args.look is None:
        e = [e for e in events if e.get("event_id") == eid]
        if not e:
            raise SystemExit("give --look")
        look = (float(e[0]["l0"]), float(e[0]["m0"]))
    else:
        look = tuple(args.look)
    steer = (look[0] + args.look_offset[0], look[1] + args.look_offset[1])
    frame_s = args.samples * 10e-6 / 3

    dumps = list_correlation_dumps(args.corr_dir)
    t_of = dict(zip(np.asarray(meta["frame_out_idx"]).astype(int), meta["frame_time_s"]))
    idx = [k for k in sorted(dumps) if k in t_of]
    times = np.array([t_of[k] for k in idx])
    order = np.argsort(times)
    idx, times = [idx[i] for i in order], times[order]
    ch = np.asarray(args.channels, dtype=int)
    print(f"{len(idx)} dumps, {N} antennas; burst {eid}: DM {dm:g}, sigma {width_s * 1e3:g} ms, "
          f"top arrival {t_top:g} s; look ({look[0]:+.4f}, {look[1]:+.4f}), weights steered to "
          f"({steer[0]:+.4f}, {steer[1]:+.4f})")

    # Per-frame covariances of the selected channels, per sample: (frames, ch, N, N)
    C = np.empty((len(idx), ch.size, N, N), dtype=np.complex128)
    for i, k in enumerate(idx):
        C[i] = read_astron_correlation(dumps[k], fmhz.size, N)[ch] / args.samples

    f_top = float(fmhz.max())
    sweep = float(dispersion_delay_s(fmhz.min(), dm, f_top))
    w0, w1 = t_top - args.n_sigma * width_s, t_top + sweep + args.n_sigma * width_s
    off_global = (times + frame_s < w0) | (times > w1)
    on_all = on_pulse_mask(times, fmhz[ch], dm, t_top, width_s, args.n_sigma, frame_s, f_top)
    t_arr = arrival_times_s(fmhz[ch], dm, t_top, f_top)
    frame0 = np.floor(t_top / frame_s) * frame_s
    print(f"off-global: {off_global.sum()} frames outside [{w0:.4f}, {w1:.4f}] s")

    res = {}
    print("\nS/S_DFT: signal retained relative to the DFT; gain: S/N over the DFT; "
          "resp: |w^H a|^2 towards interferers / extra directions")
    for c, chn in enumerate(ch):
        f = fmhz[chn] * 1e6
        on = on_all[:, c]
        guard = dilate(on, args.guard_frames)
        local = (np.abs(times + frame_s / 2 - t_arr[c]) <= args.local_s) & ~guard
        n_on, n_loc = int(on.sum()), int(local.sum())
        if n_on == 0 or n_loc == 0:
            print(f"\nch {chn}: {n_on} on / {n_loc} off-local frames, skipped")
            continue
        R_on, R_loc = C[on, c].mean(0), C[local, c].mean(0)
        covs = {"on": R_on, "all": C[:, c].mean(0), "off_global": C[off_global, c].mean(0),
                "off_local": R_loc}
        d = steering_vectors(*steer, f, x, y)[0]
        dirs = [(a, b) for a, b, _ in rfi.get(chn, [])] + extra
        W = {"dft": dft_weights(d)[0]}
        for kind in KINDS:
            for ld in args.loading:
                W[f"{kind}_d{ld:g}"] = mvdr_weights(covs[kind], d, loading=ld,
                                                    loading_ref="noise")[0]
        dR = R_on - R_loc
        S = {k: float(np.real(np.vdot(w, dR @ w))) for k, w in W.items()}
        Q = {k: float(np.real(np.vdot(w, R_loc @ w))) for k, w in W.items()}
        trunc = " (leading tail truncated)" if t_arr[c] - args.n_sigma * width_s < frame0 else ""
        print(f"\nch {chn} ({f / 1e6:.1f} MHz): arrival {t_arr[c]:.4f} s, {n_on} on / {n_loc} "
              f"off-local frames{trunc}; DFT S = {S['dft']:.4f}, S/N = {S['dft'] / Q['dft']:.3f}; "
              f"directions: {[(round(a, 3), round(b, 3)) for a, b in dirs] or '-'}")
        print(f"  {'weights':<14} {'S/S_DFT':>8} {'gain dB':>8} {'WNG dB':>7}  resp [dB]")
        for k, w in W.items():
            ret = S[k] / S["dft"] if S["dft"] > 0 else np.nan
            gain = db(S[k] / Q[k]) - db(S["dft"] / Q["dft"]) if S[k] > 0 else np.nan
            wng = float(db(N * np.sum(np.abs(w) ** 2)))
            resp = [float(db(beam_response(w, a, b, f, x, y))[0]) for a, b in dirs]
            res[(int(chn), k)] = {"ret": ret, "gain_db": gain, "wng_db": wng,
                                  "resp_db": resp, "S": S[k], "Q": Q[k]}
            print(f"  {k:<14} {ret:8.3f} {gain:8.2f} {wng:7.2f}  "
                  + (" ".join(f"{v:6.1f}" for v in resp) or "-"))

    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for _, k in res}, key=lambda s: (s != "dft", s))
    chans = sorted({c for c, _ in res})
    np.savez(args.out / "transient_snr.npz", channels=np.array(chans), weights=np.array(keys),
             look=np.array(look), steer=np.array(steer), dm=dm, width_s=width_s, t_top=t_top,
             **{f"{m}": np.array([[res.get((c, k), {}).get(m, np.nan) for k in keys]
                                  for c in chans]) for m in ("ret", "gain_db", "wng_db", "S", "Q")})
    if chans:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        xx = np.arange(len(chans))
        mv = [k for k in keys if k != "dft"]
        wbar = 0.8 / max(1, len(mv))
        for j, k in enumerate(mv):
            for ax, m in zip(axes, ("gain_db", "ret")):
                ax.bar(xx + (j - (len(mv) - 1) / 2) * wbar,
                       [res.get((c, k), {}).get(m, np.nan) for c in chans], wbar, label=k)
        for ax, lab in zip(axes, ("S/N gain over DFT [dB]", "signal retained S / S_DFT")):
            ax.set_xticks(xx, [str(c) for c in chans]); ax.set_xlabel("channel")
            ax.set_ylabel(lab); ax.grid(alpha=0.3, axis="y")
        axes[1].axhline(1.0, color="0.4", lw=0.8)
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"burst {eid}: weights steered to ({steer[0]:+.4f}, {steer[1]:+.4f}), "
                     f"true ({look[0]:+.4f}, {look[1]:+.4f})", fontsize=10)
        fig.tight_layout(); fig.savefig(args.out / "transient_snr.png", dpi=130); plt.close(fig)
    print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
