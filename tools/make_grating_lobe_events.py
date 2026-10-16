#!/usr/bin/env python
"""Events file with narrowband interferers near a grating lobe of a look direction.

On a rectangular grid the beam steered to (l0, m0) repeats at the grating lobes
(l0 + p c / (f dx), m0 + q c / (f dy)), where the steering vector is identical to
the look direction (beamforming.grating_lobes). This tool places one narrowband
interferer per channel, present over the whole window, at a given offset from
the lobe of order (p, q) in that channel, moving along --direction, so that a
single window samples MVDR performance against the spatial correlation between
the look direction and the interferer:

    rho^2 = |a_look^H a_int|^2 / N^2   (1 at the lobe, the DFT beam pattern elsewhere)

The output is a JSON list in the format of the simulator events file
(<window>_events.json), to be passed to the generator with --events-json.
Several lobes or amplitudes go into the same file with --append.

Frame selection: the generator writes frames at a background cadence plus a
dense and sparse capture around each non-persistent event; persistent events
trigger no captures. A window with only persistent interferers therefore holds
a handful of frames, so by default the events are written as non-persistent
(t_start 0, the full duration), which gives the event capture cadence.

For each event the tool also prints the MVDR prediction for an ideal
covariance (infinite K, no quantization):
    R = noise I + p_look a_look a_look^H + amp^2 a_int a_int^H,
with diagonal loading in units of the noise floor (as in 05 with
--loading-ref noise): white-noise gain, null depth towards the interferer and
leak (amp^2 |w^H a_int|^2) relative to p_look. At the lobe MVDR and DFT coincide.

Example (8 x 8 grid with 0.6 m spacing, lobe (-1, 0) near 399 MHz):
  python tools/make_grating_lobe_events.py --look 0.4677 0.4801 --spacing 0.6 0.6 \\
      --lobe -1 0 --direction 1 0 --channels 321 323 325 --offsets 0 0.01 0.04 \\
      --amp 4 --duration 5 --noise 6 --p-look 0.17 --out grating_lobe_events.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from charts_mvdr.beamforming import grating_lobes, mvdr_weights, steering_vectors


def grid_positions(nx: int, ny: int, dx: float, dy: float):
    """Antenna positions of an nx x ny grid, x along columns (index % nx)."""
    k = np.arange(nx * ny)
    return (k % nx) * dx, (k // nx) * dy


def make_event(event_id, l, m, ch, freq_mhz, amp, duration, lobe, offset, persistent=False):
    return {
        "event_id": event_id,
        "event_type": "rfi_narrow",
        "t_start_s": 0.0,
        "duration_s": float(duration),
        "dm": 0.0,
        "nominal_amp": float(amp),
        "l0": float(l),
        "m0": float(m),
        "drift_dl": 0.0,
        "drift_dm": 0.0,
        "pulse_period_s": 0.0,
        "pulse_width_ms": 1.0,
        "channels": [int(ch)],
        "is_persistent": bool(persistent),
        "description": (f"Narrowband RFI on channel {ch} ({freq_mhz:.1f} MHz), offset "
                        f"{offset:g} from grating lobe {tuple(lobe)} (amp={amp:g} LSB)"),
        "ra_deg": None,
        "dec_deg": None,
        "source_name": "",
    }


def predict(d, a, amp, noise, p_look, loading):
    """MVDR on the ideal covariance: (rho^2, WNG dB, null dB, leak / p_look)."""
    N = d.size
    R = (noise * np.eye(N) + p_look * np.outer(d, d.conj())
         + amp ** 2 * np.outer(a, a.conj()))
    w = mvdr_weights(R, d, loading=loading, loading_ref="noise")[0]
    rho2 = abs(np.vdot(d, a)) ** 2 / N ** 2
    gain = abs(np.vdot(w, a)) ** 2
    wng = N * np.sum(np.abs(w) ** 2)
    leak = amp ** 2 * gain / p_look if p_look > 0 else np.nan
    return rho2, 10 * np.log10(wng), 10 * np.log10(gain + 1e-30), leak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--look", type=float, nargs=2, required=True, metavar=("L", "M"))
    ap.add_argument("--spacing", type=float, nargs=2, required=True, metavar=("DX", "DY"),
                    help="grid spacings in m")
    ap.add_argument("--grid", type=int, nargs=2, default=(8, 8), metavar=("NX", "NY"),
                    help="antennas along x and y (for the prediction; default 8 8)")
    ap.add_argument("--f0-mhz", type=float, default=300.0, help="frequency of channel 0")
    ap.add_argument("--df-mhz", type=float, default=0.3, help="channel width")
    ap.add_argument("--lobe", type=int, nargs=2, metavar=("P", "Q"),
                    help="grating-lobe order; omit to list the visible lobes and exit")
    ap.add_argument("--direction", type=float, nargs=2, default=(1.0, 0.0),
                    metavar=("DL", "DM"), help="offset direction from the lobe (normalized)")
    ap.add_argument("--channels", type=int, nargs="+")
    ap.add_argument("--offsets", type=float, nargs="+",
                    help="offset from the lobe in (l, m) units, one per channel")
    ap.add_argument("--amp", type=float, default=4.0, help="interferer amplitude [LSB]")
    ap.add_argument("--duration", type=float, default=5.0, help="event duration [s]")
    ap.add_argument("--persistent", action="store_true",
                    help="mark the events persistent (see the note on frame selection)")
    ap.add_argument("--id-prefix", default="gl", help="prefix of the event ids")
    ap.add_argument("--noise", type=float, default=1.0,
                    help="per-antenna noise power for the prediction [LSB^2]")
    ap.add_argument("--p-look", type=float, default=0.0,
                    help="look-direction source power for the prediction [LSB^2]")
    ap.add_argument("--loading", type=float, default=10.0,
                    help="MVDR loading for the prediction, in units of the noise floor")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--append", action="store_true",
                    help="add to the events already in --out instead of overwriting it")
    args = ap.parse_args()

    l0, m0 = args.look
    dx, dy = args.spacing
    if args.lobe is None:
        for ch in args.channels or []:
            f_mhz = args.f0_mhz + ch * args.df_mhz
            lobes = grating_lobes(l0, m0, f_mhz * 1e6, dx, dy)
            txt = ", ".join(f"({p:+d},{q:+d}): ({l:+.4f}, {m:+.4f})" for p, q, l, m in lobes)
            print(f"ch {ch:4d} {f_mhz:7.1f} MHz  {txt or 'no visible lobe'}")
        if not args.channels:
            ap.error("give --channels to list lobes, or --lobe to write events")
        return
    if not (args.channels and args.offsets and args.out):
        ap.error("--lobe needs --channels, --offsets and --out")
    if len(args.channels) != len(args.offsets):
        ap.error("--channels and --offsets must have the same length")

    u = np.asarray(args.direction, dtype=float)
    if not np.any(u):
        ap.error("--direction must be non-zero")
    u = u / np.hypot(*u)
    x, y = grid_positions(*args.grid, dx, dy)

    events = json.loads(args.out.read_text()) if args.append and args.out.exists() else []
    used = {ev["event_id"] for ev in events}
    print(f"{'ch':>4} {'MHz':>7} {'lobe (l, m)':>18} {'offset':>7} {'int (l, m)':>18} "
          f"{'rho^2':>7} {'WNG dB':>7} {'null dB':>8} {'leak/p':>8}")
    for ch, off in zip(args.channels, args.offsets):
        f_mhz = args.f0_mhz + ch * args.df_mhz
        f_hz = f_mhz * 1e6
        lobe = {(p, q): (l, m) for p, q, l, m in
                grating_lobes(l0, m0, f_hz, dx, dy, max_order=max(map(abs, args.lobe)))}
        if tuple(args.lobe) not in lobe:
            raise SystemExit(f"lobe {tuple(args.lobe)} is not visible at channel {ch}")
        ll, lm = lobe[tuple(args.lobe)]
        l, m = ll + off * u[0], lm + off * u[1]
        if l * l + m * m >= 1.0:
            raise SystemExit(f"channel {ch}: interferer ({l:+.4f}, {m:+.4f}) is not visible")
        eid = f"{args.id_prefix}_p{args.lobe[0]:+d}q{args.lobe[1]:+d}_ch{ch:03d}"
        if eid in used:
            raise SystemExit(f"duplicated event id {eid}")
        used.add(eid)
        events.append(make_event(eid, l, m, ch, f_mhz, args.amp, args.duration,
                                 args.lobe, off, args.persistent))
        d = steering_vectors(l0, m0, f_hz, x, y)[0]
        a = steering_vectors(l, m, f_hz, x, y)[0]
        rho2, wng, null, leak = predict(d, a, args.amp, args.noise, args.p_look, args.loading)
        print(f"{ch:4d} {f_mhz:7.1f} ({ll:+.4f}, {lm:+.4f}) {off:7.3f} ({l:+.4f}, {m:+.4f}) "
              f"{rho2:7.4f} {wng:7.2f} {null:8.1f} {leak:8.3g}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(events, indent=2) + "\n")
    print(f"{len(events)} events -> {args.out}")


if __name__ == "__main__":
    main()
