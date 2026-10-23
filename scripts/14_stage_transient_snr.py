#!/usr/bin/env python
"""S/N of a dispersed burst in the beam power of the MVDR stage, with a paired window.

Reads the stage output (npz from 12_stage_reference.py or 13_compare_kotekan_mvdr.py:
power (2, n_frames, n_chan, n_beam), planes DFT and MVDR) of a window with RFI (--rfi) and
of its paired window without RFI (--clean, same noise and burst), plus the window metadata
and events (frame times, burst DM, width and arrival). Frame k of the stage output is
window frame k (the order of the replay and of the correlation dumps).

Per channel the frames are split with the dispersion delay (transients.on_pulse_mask), as
in 08: on-pulse frames, and off-local frames within --local-s of the arrival in that
channel, excluding the on frames and --guard-frames on each side. For each beamformer
(DFT, MVDR) and window:

    S = mean_on P - mean_off P,    sigma = std_off P,    S/N_c = S / (sigma / sqrt(n_on))

sigma is the fluctuation of the beam power, the quantity a detection pipeline sees (09
metric), not its mean. The dedispersed S/N adds the channels with equal weight:

    S/N_tot = sum_c S_c / sqrt(sum_c sigma_c^2 / n_on_c).

Everything is reported in dB relative to the DFT of the clean window (10 log10 of S/N
ratios). Channels whose leading tail is cut by the simulator are excluded (as in 08/09),
and so are channels with fewer than --min-off off-local frames.

Optional --groups name:ch,ch-ch ... prints the median per group (e.g. the RFI groups of a
window), plus the channels in no group as "other".

Example:
  python scripts/14_stage_transient_snr.py --rfi V3.npz --clean V3C.npz \\
      --window-dir WINDOW_DIR --groups B:210-217 D:170-174 ch192:192
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import load_events_file, load_window_dir_meta
from charts_mvdr.transients import arrival_times_s, burst_from_events, on_pulse_mask

FRAME_S = 1536 * 10e-6 / 3


def db(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float))


def parse_groups(items):
    groups = {}
    for item in items or []:
        name, _, spec = item.partition(":")
        chans = []
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                chans.extend(range(int(a), int(b) + 1))
            elif part:
                chans.append(int(part))
        groups[name] = chans
    return groups


def channel_snr(P, on, off):
    """S, sigma, n_on per channel for one beam: P (n_frames, n_chan), masks (n_frames, n_chan)."""
    n_chan = P.shape[1]
    S, sig, n_on = np.full(n_chan, np.nan), np.full(n_chan, np.nan), on.sum(0)
    for c in range(n_chan):
        if n_on[c] == 0 or off[:, c].sum() < 2:
            continue
        S[c] = P[on[:, c], c].mean() - P[off[:, c], c].mean()
        sig[c] = P[off[:, c], c].std(ddof=1)
    return S, sig, n_on


def masks(meta, events, n_frames, freqs_mhz, n_sigma, local_s, guard, frame_s=FRAME_S):
    burst = burst_from_events(events)
    t_of = dict(zip(np.asarray(meta["frame_out_idx"]).astype(int), meta["frame_time_s"]))
    missing = [k for k in range(n_frames) if k not in t_of]
    if missing:
        raise SystemExit(f"{len(missing)} stage frames have no time in the metadata")
    times = np.array([t_of[k] for k in range(n_frames)])
    f_top = float(np.max(meta["frequencies_mhz"]))
    on = on_pulse_mask(times, freqs_mhz, burst["dm"], burst["t_top_s"], burst["width_s"],
                       n_sigma, frame_s, f_top)
    t_arr = arrival_times_s(freqs_mhz, burst["dm"], burst["t_top_s"], f_top)
    guard_m = on.copy()
    for k in range(1, guard + 1):
        guard_m[k:] |= on[:-k]
        guard_m[:-k] |= on[k:]
    off = (np.abs(times[:, None] + frame_s / 2 - t_arr[None, :]) <= local_s) & ~guard_m
    frame0 = np.floor(burst["t_top_s"] / frame_s) * frame_s
    truncated = t_arr - n_sigma * burst["width_s"] < frame0
    return on, off, truncated, burst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rfi", required=True, type=Path, help="stage npz of the window with RFI")
    ap.add_argument("--clean", required=True, type=Path, help="stage npz of the paired window")
    ap.add_argument("--window-dir", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=0, help="beam index in the npz")
    ap.add_argument("--n-sigma", type=float, default=3.0)
    ap.add_argument("--local-s", type=float, default=0.05)
    ap.add_argument("--guard-frames", type=int, default=1)
    ap.add_argument("--min-off", type=int, default=4)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None, help="per-channel table (npz)")
    args = ap.parse_args(argv)

    zr, zc = np.load(args.rfi), np.load(args.clean)
    if zr["power"].shape != zc["power"].shape:
        raise SystemExit("the two windows must have the same frames, channels and beams")
    chans = np.asarray(zr["channels"], dtype=int)
    freqs = np.asarray(zr["freqs_mhz"], dtype=float)
    meta = load_window_dir_meta(args.window_dir)
    events = load_events_file(args.window_dir)
    n_frames = zr["power"].shape[1]
    on, off, trunc, burst = masks(meta, events, n_frames, freqs, args.n_sigma, args.local_s,
                                  args.guard_frames)
    usable = ~trunc & (off.sum(0) >= args.min_off) & (on.sum(0) > 0)
    print(f"burst DM {burst['dm']:g}, sigma {burst['width_s'] * 1e3:g} ms; {usable.sum()} of "
          f"{chans.size} channels usable ({trunc.sum()} with truncated leading tail)")

    res = {}
    for name, z in (("rfi", zr), ("clean", zc)):
        for plane, kind in ((0, "dft"), (1, "mvdr")):
            res[(name, kind)] = channel_snr(z["power"][plane, :, :, args.beam], on, off)
    S0, s0, n0 = res[("clean", "dft")]
    snr0 = S0 / (s0 / np.sqrt(n0))
    table = {}
    print("\nS/N relative to the DFT of the clean window [dB]")
    print(f"{'':12s} {'clean MVDR':>10s} {'RFI DFT':>9s} {'RFI MVDR':>9s}")
    for key in (("clean", "mvdr"), ("rfi", "dft"), ("rfi", "mvdr")):
        S, s, n = res[key]
        table[key] = db((S / (s / np.sqrt(n))) / snr0)
    tot = {}
    for key, (S, s, n) in res.items():
        u = usable & np.isfinite(S) & np.isfinite(s)
        tot[key] = S[u].sum() / np.sqrt(np.sum(s[u] ** 2 / n[u]))
    ref = tot[("clean", "dft")]
    print(f"{'dedispersed':12s} {db(tot[('clean', 'mvdr')] / ref):10.2f} "
          f"{db(tot[('rfi', 'dft')] / ref):9.2f} {db(tot[('rfi', 'mvdr')] / ref):9.2f}"
          f"   (clean DFT S/N {ref:.1f})")

    groups = parse_groups(args.groups)
    in_group = set()
    for name, chs in groups.items():
        sel = np.isin(chans, chs) & usable
        in_group.update(chs)
        if not sel.any():
            print(f"{name:12s} (no usable channels)")
            continue
        med = [np.nanmedian(table[k][sel]) for k in (("clean", "mvdr"), ("rfi", "dft"),
                                                      ("rfi", "mvdr"))]
        print(f"{name:12s} {med[0]:10.2f} {med[1]:9.2f} {med[2]:9.2f}   n={sel.sum()}")
    if groups:
        sel = ~np.isin(chans, list(in_group)) & usable
        med = [np.nanmedian(table[k][sel]) for k in (("clean", "mvdr"), ("rfi", "dft"),
                                                      ("rfi", "mvdr"))]
        print(f"{'other':12s} {med[0]:10.2f} {med[1]:9.2f} {med[2]:9.2f}   n={sel.sum()}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, channels=chans, usable=usable, clean_dft_snr=snr0,
                 clean_mvdr_db=table[("clean", "mvdr")], rfi_dft_db=table[("rfi", "dft")],
                 rfi_mvdr_db=table[("rfi", "mvdr")],
                 dedispersed_db=np.array([db(tot[k] / ref) for k in
                                          (("clean", "mvdr"), ("rfi", "dft"), ("rfi", "mvdr"))]))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
