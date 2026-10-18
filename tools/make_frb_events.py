#!/usr/bin/env python
"""Events file with a dispersed FRB and optional narrowband RFI.

Writes a JSON list in the format of the simulator events file
(<window>_events.json), to be passed to the generator with --events-json, and
prints what the capture schedule needs to sample the burst:

  sweep              dispersion delay between the top and bottom channels
  dense capture      --event-dense-s covering the sweep plus n_sigma widths
                     after the top-channel arrival (the generator writes frames
                     contiguously for event_dense_s from each event start)
  off-pulse window   [t_top - n_sigma w, t_top + sweep + n_sigma w], to exclude
                     from the weights covariance (05/07 --cov-exclude)
  truncated channels channels whose pulse starts before the frame holding the
                     event start: the generator renders an event only in frames
                     overlapping [t_start, t_start + duration], and t_start is
                     the top-channel arrival, so the leading tail is cut there
  RFI overlap        RFI events whose channels see the burst while active
  frames             estimate of the frames written (same rules as the
                     generator's frame selection), with the dense capture of
                     the burst in the events file (or --event-dense-s, the
                     value passed to the generator); --compare-window lists
                     the differences with the frames of a generated window

The FRB envelope in the simulator is a Gaussian in amplitude with standard
deviation pulse_width_ms (FWHM = 2.355 pulse_width_ms), peaking at t_start_s in
the top channel and later in lower channels by the dispersion delay.

RFI events are given with --rfi as key=value tokens:
    --rfi "l=-0.83 m=-0.482 amp=7 t0=0 dur=2 ch=94,133,147"
    --rfi "l=0.858 m=0.43 amp=7 t0=burst dur=0.1 ch=192 id=rfi_burst"
t0=burst centres the event on the burst arrival in its first channel. Optional
keys: id, type (rfi_narrow, default), persistent (0/1, default 0: persistent
events trigger no captures). Several calls can be combined with --append; a
call without --frb takes the burst (t0=burst, dense capture, report) from the
frb event already in --out.

Example (DM 10 burst at (-0.10, 0.25), 2 s window):
  python tools/make_frb_events.py --frb -0.10 0.25 --dm 10 --width-ms 2 --amp 4 \\
      --t-top 1.0 --window 2.0 --out frb_events.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from charts_mvdr.transients import arrival_times_s, burst_from_events, dispersion_delay_s

FRAME_S = 1536 * 10e-6 / 3          # samples per frame x sample time
EVENT_KEYS = ("event_id", "event_type", "t_start_s", "duration_s", "dm", "nominal_amp",
              "l0", "m0", "drift_dl", "drift_dm", "pulse_period_s", "pulse_width_ms",
              "channels", "is_persistent", "description", "ra_deg", "dec_deg",
              "source_name")


def base_event(**kw) -> dict:
    ev = {"event_id": "", "event_type": "rfi_narrow", "t_start_s": 0.0, "duration_s": 1.0,
          "dm": 0.0, "nominal_amp": 3.0, "l0": 0.0, "m0": 0.0, "drift_dl": 0.0,
          "drift_dm": 0.0, "pulse_period_s": 0.0, "pulse_width_ms": 1.0, "channels": None,
          "is_persistent": False, "description": "", "ra_deg": None, "dec_deg": None,
          "source_name": ""}
    ev.update(kw)
    return ev


def parse_rfi(spec: str, t_burst_by_ch, index: int) -> dict:
    """RFI event from a 'key=value ...' string (see the module docstring)."""
    kv = {}
    for tok in spec.split():
        if "=" not in tok:
            raise ValueError(f"--rfi token {tok!r} is not key=value")
        k, v = tok.split("=", 1)
        kv[k.strip()] = v.strip()
    allowed = {"l", "m", "amp", "t0", "dur", "ch", "id", "type", "persistent"}
    unknown = set(kv) - allowed
    missing = {"l", "m", "amp", "t0", "dur", "ch"} - set(kv)
    if unknown or missing:
        raise ValueError(f"--rfi {spec!r}: unknown keys {sorted(unknown)}, "
                         f"missing keys {sorted(missing)}")
    chans = [int(c) for c in kv["ch"].split(",") if c]
    dur = float(kv["dur"])
    if kv["t0"] == "burst":
        t0 = float(t_burst_by_ch(chans[0])) - dur / 2.0
    else:
        t0 = float(kv["t0"])
    etype = kv.get("type", "rfi_narrow")
    if etype not in ("rfi_narrow", "rfi_leo"):
        raise ValueError(f"--rfi type must be rfi_narrow or rfi_leo, got {etype!r}")
    amp = float(kv["amp"])
    return base_event(
        event_id=kv.get("id", f"rfi_{index:02d}"), event_type=etype, t_start_s=max(t0, 0.0),
        duration_s=dur, nominal_amp=amp, l0=float(kv["l"]), m0=float(kv["m"]),
        channels=chans, is_persistent=bool(int(kv.get("persistent", "0"))),
        description=f"Narrowband RFI on channels {chans} (amp={amp:g} LSB)")


def frame_schedule(total_frames, events, background_s, dense_s, sparse_ms):
    """Frame indices written by the generator (same rules as its frame selection)."""
    sel = set(range(0, total_frames, max(1, int(round(background_s / FRAME_S)))))
    n_dense = max(1, int(round(dense_s / FRAME_S)))
    step = max(1, int(round(sparse_ms * 1e-3 / FRAME_S)))
    for ev in events:
        if ev["is_persistent"]:
            continue
        k0 = max(0, int(math.floor(ev["t_start_s"] / FRAME_S)))
        k1 = min(total_frames, int(math.ceil((ev["t_start_s"] + ev["duration_s"]) / FRAME_S)))
        kd = min(k1, k0 + n_dense)
        sel.update(range(k0, kd))
        sel.update(range(kd, k1, step))
    return sorted(sel)


def dense_capture_s(sweep_s: float, width_s: float, n_sigma: float) -> float:
    """--event-dense-s covering the sweep plus n_sigma widths, in whole frames."""
    return math.ceil((sweep_s + n_sigma * width_s) / FRAME_S + 1) * FRAME_S


def written_frames(window_dir) -> list:
    """Frame indices of a generated window, from its metadata timestamps."""
    from charts_mvdr.data_io import load_window_dir_meta

    meta = load_window_dir_meta(window_dir)
    if "frame_time_s" not in meta:
        raise SystemExit(f"no frame timestamps in the metadata of {window_dir}")
    return sorted({int(round(t / FRAME_S)) for t in meta["frame_time_s"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frb", type=float, nargs=2, metavar=("L", "M"),
                    help="burst direction; omit to write only --rfi events")
    ap.add_argument("--dm", type=float, default=10.0, help="dispersion measure [pc cm^-3]")
    ap.add_argument("--width-ms", type=float, default=2.0,
                    help="envelope standard deviation [ms] (pulse_width_ms)")
    ap.add_argument("--amp", type=float, default=4.0, help="burst peak amplitude [LSB]")
    ap.add_argument("--t-top", type=float, default=1.0,
                    help="arrival time in the top channel [s since window start]")
    ap.add_argument("--n-sigma", type=float, default=3.0,
                    help="pulse extent in widths for the capture and off-pulse window")
    ap.add_argument("--rfi", action="append", default=[], metavar="SPEC",
                    help="RFI event as 'l=.. m=.. amp=.. t0=..|burst dur=.. ch=a,b [id=..]'")
    ap.add_argument("--window", type=float, default=2.0, help="window duration [s]")
    ap.add_argument("--f0-mhz", type=float, default=300.0, help="frequency of channel 0")
    ap.add_argument("--df-mhz", type=float, default=0.3, help="channel width")
    ap.add_argument("--num-freq", type=int, default=336)
    ap.add_argument("--background-cadence-s", type=float, default=1.0,
                    help="for the frame estimate (quick preset: 1.0)")
    ap.add_argument("--event-sparse-cadence-ms", type=float, default=50.0,
                    help="for the frame estimate (quick preset: 50)")
    ap.add_argument("--event-dense-s", type=float, default=None,
                    help="for the frame estimate: value passed to the generator "
                         "(default: the dense capture of the burst, 0.02 without one)")
    ap.add_argument("--compare-window", type=Path, default=None, metavar="WINDOW_DIR",
                    help="compare the estimate with the frames of a generated window")
    ap.add_argument("--frame-mb", type=float, default=1536 * 336 * 64 / 1e6,
                    help="size of one frame [MB] for the volume estimate")
    ap.add_argument("--id", default="frb_01", help="event id of the burst")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--append", action="store_true",
                    help="add to the events already in --out instead of overwriting it")
    args = ap.parse_args()

    freqs = args.f0_mhz + args.df_mhz * np.arange(args.num_freq)
    f_top = float(freqs[-1])
    events = json.loads(args.out.read_text()) if args.append and args.out.exists() else []
    burst = None
    if not args.frb:
        try:
            burst = burst_from_events(events)
        except ValueError:
            burst = None
        if burst is not None:              # burst parameters from the events file
            args.dm, args.width_ms, args.t_top = burst["dm"], burst["width_s"] * 1e3, burst["t_top_s"]
    w_s = args.width_ms * 1e-3
    sweep = float(dispersion_delay_s(freqs[0], args.dm, f_top))
    t_arr = arrival_times_s(freqs, args.dm, args.t_top, f_top)
    if burst is None and not args.frb and any(
            "t0=burst" in s.replace(" ", "") for s in args.rfi):
        raise SystemExit("t0=burst needs --frb or an frb event in --out (with --append)")
    new = []
    if args.frb:
        l, m = args.frb
        if l * l + m * m >= 1.0:
            raise SystemExit(f"burst direction ({l}, {m}) is below the horizon")
        new.append(base_event(
            event_id=args.id, event_type="frb", t_start_s=args.t_top,
            duration_s=sweep + 2 * args.n_sigma * w_s, dm=args.dm, nominal_amp=args.amp,
            l0=l, m0=m, pulse_width_ms=args.width_ms, source_name="Synthetic FRB",
            description=(f"Synthetic FRB (DM={args.dm:g} pc/cm^3, width={args.width_ms:g} ms "
                         f"sigma, sweep={sweep * 1e3:.1f} ms)")))
    for i, spec in enumerate(args.rfi):
        try:
            new.append(parse_rfi(spec, lambda ch: t_arr[ch], len(events) + i + 1))
        except ValueError as err:
            raise SystemExit(str(err))
    ids = [ev["event_id"] for ev in events + new]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicated event ids in {ids}")
    for ev in new:
        for ch in ev["channels"] or []:
            if not 0 <= ch < args.num_freq:
                raise SystemExit(f"{ev['event_id']}: channel {ch} outside 0-{args.num_freq - 1}")
    events += new

    if args.frb or burst is not None:
        if burst is not None:
            print(f"burst from {args.out.name}: {burst['id']}")
        dense = dense_capture_s(sweep, w_s, args.n_sigma)
        off0, off1 = args.t_top - args.n_sigma * w_s, args.t_top + sweep + args.n_sigma * w_s
        frame0 = math.floor(args.t_top / FRAME_S) * FRAME_S
        trunc = np.flatnonzero(t_arr - args.n_sigma * w_s < frame0)
        print(f"burst: DM {args.dm:g}, sigma {args.width_ms:g} ms (FWHM "
              f"{2.355 * args.width_ms:.2f} ms), sweep {sweep * 1e3:.1f} ms over "
              f"{freqs[0]:.1f}-{f_top:.1f} MHz, top-channel arrival {args.t_top:g} s")
        print(f"  dense capture : --event-dense-s {dense:.3f}")
        print(f"  off-pulse     : --cov-exclude {off0:.4f} {off1:.4f}")
        if trunc.size:
            print(f"  truncated     : leading tail cut in channels {trunc.min()}-{trunc.max()} "
                  f"({freqs[trunc.min()]:.1f}-{freqs[trunc.max()]:.1f} MHz), event starts "
                  f"in the frame at {frame0:.5f} s")
        for ev in events:
            if ev["event_type"].startswith("rfi") and ev["channels"]:
                t0, t1 = ev["t_start_s"], ev["t_start_s"] + ev["duration_s"]
                hit = [ch for ch in ev["channels"]
                       if t0 <= t_arr[ch] <= t1 or ev["is_persistent"]]
                if hit:
                    print(f"  RFI overlap   : {ev['event_id']} on channels {hit} "
                          f"(burst at {', '.join(f'{t_arr[c]:.4f}' for c in hit)} s)")
    else:
        dense = 0.02
    if args.event_dense_s is not None:
        dense = args.event_dense_s
    n_tot = int(math.floor(args.window / FRAME_S))
    sel = frame_schedule(n_tot, events, args.background_cadence_s, dense,
                         args.event_sparse_cadence_ms)
    print(f"frames written (estimate, dense {dense:.3f} s): {len(sel)} of {n_tot}, "
          f"~{len(sel) * args.frame_mb / 1e3:.1f} GB")
    if args.compare_window is not None:
        got = written_frames(args.compare_window)
        extra, missing = sorted(set(got) - set(sel)), sorted(set(sel) - set(got))
        print(f"window frames: {len(got)}; written but not estimated: {extra or 'none'}; "
              f"estimated but not written: {missing or 'none'}")
        for k in extra[:10]:
            hits = [ev["event_id"] for ev in events if not ev["is_persistent"]
                    and ev["t_start_s"] <= (k + 1) * FRAME_S
                    and k * FRAME_S < ev["t_start_s"] + ev["duration_s"]]
            print(f"  frame {k} ({k * FRAME_S:.5f} s): overlaps {hits or 'no event'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(events, indent=2) + "\n")
    print(f"{len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
