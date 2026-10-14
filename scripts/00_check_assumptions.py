#!/usr/bin/env python3
"""Pre-flight check of the assumptions required for MVDR beamforming.

Runs the checks in charts_mvdr.checks on one baseband frame. Global checks run
once; per-channel checks are summarized over channels (worst status, counts and
failing channels). Use --verbose for the full per-channel table.

Multi-frame checks use every frame of the same window (same directory and
prefix as the given frame), or the matching Kotekan dumps with --corr-dir:
  --multi-frame              #9 stationarity over the window (or --t-range)
  --on T0 T1 --off T0 T1     #10 on/off-pulse consistency (needs --look)

Examples:
    python 00_check_assumptions.py <window>_0000000.bin \
        --expected 1 --expected-ch 94=2 133=2 147=2
    python 00_check_assumptions.py <window>_0000000.bin --channels 94 95 166 \
        --multi-frame --corr-dir <kotekan_dumps>
    python 00_check_assumptions.py <window>_0000000.bin --channels 95 --look 0.46 0.48 \
        --on 1.0 1.2 --off 0.0 0.9
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from charts_mvdr import checks as ck
from charts_mvdr.beamforming import steering_vectors
from charts_mvdr.data_io import read_astron_correlation, read_baseband_frame, read_window_meta
from charts_mvdr.np_correlator import correlate

ORDER = [ck.Status.FAIL, ck.Status.WARN, ck.Status.PASS, ck.Status.NA]
FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{7})\.bin$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("frame", type=Path, help="baseband frame file (<window>_NNNNNNN.bin)")
    p.add_argument("--channels", type=int, nargs="+", default=None,
                   help="channels to check (default: all)")
    p.add_argument("--expected", type=int, default=None,
                   help="expected number of sources in every channel")
    p.add_argument("--expected-ch", nargs="*", default=[], metavar="CH=N",
                   help="per-channel overrides of the expected source count")
    p.add_argument("--num-freq", type=int, default=None)
    p.add_argument("--num-ant", type=int, default=None)
    p.add_argument("--verbose", action="store_true", help="print every per-channel result")
    p.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"),
                   help="look direction for the spatial-separation check")
    p.add_argument("--interferer", type=float, nargs=2, action="append", default=[],
                   metavar=("L", "M"), help="known interferer direction (repeatable)")
    g = p.add_argument_group("multi-frame checks")
    g.add_argument("--multi-frame", action="store_true",
                   help="#9 stationarity over the frames of the window")
    g.add_argument("--corr-dir", type=Path, default=None,
                   help="Kotekan correlation dumps (corr_NNNNNNN.bin) instead of correlating frames")
    g.add_argument("--t-range", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="frames used by --multi-frame [s]")
    g.add_argument("--on", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="on-pulse time range for #10 [s]")
    g.add_argument("--off", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="off-pulse time range for #10 [s] (at least two frames)")
    return p.parse_args()


def parse_expected(items):
    out = {}
    for item in items:
        ch, n = item.split("=")
        out[int(ch)] = int(n)
    return out


def summarize(results):
    """One line per check: worst status, counts per status, flagged channels."""
    groups = defaultdict(list)
    for r in results:
        groups[r.name.split("[")[0]].append(r)
    lines = []
    for base, rs in groups.items():
        by_status = {s: [r for r in rs if r.status == s] for s in ORDER}
        worst = next(s for s in ORDER if by_status[s])
        counts = ", ".join(f"{s.value} {len(by_status[s])}" for s in ORDER if by_status[s])
        lines.append(f"{base:<16} {worst.value:<6} {counts}")
        for s in (ck.Status.FAIL, ck.Status.WARN):
            chans = [r.name.split("[")[1].rstrip("]") for r in by_status[s]]
            if chans:
                more = " ..." if len(chans) > 20 else ""
                lines.append(f"{'':<24}{s.value} channels: {', '.join(chans[:20])}{more}")
    return "\n".join(lines)


def window_frames(frame: Path, corr_dir: Path | None, meta: dict) -> list[tuple]:
    """(index, time [s], path, kind) for every frame of the window, sorted by index."""
    m = FRAME_RE.match(frame.name)
    if m is None:
        raise SystemExit(f"cannot parse the frame index of {frame.name}")
    times = meta.get("frame_time_s", {})
    if corr_dir is not None:
        paths = {int(q["index"]): p for p in corr_dir.glob("corr_*.bin")
                 if (q := FRAME_RE.match(p.name))}
        kind = "corr"
    else:
        paths = {int(q["index"]): p for p in frame.parent.glob(f"{m['prefix']}_*.bin")
                 if (q := FRAME_RE.match(p.name)) and q["prefix"] == m["prefix"]}
        kind = "baseband"
    if not times:
        print("WARNING: no frame timestamps in the metadata; using frame indices as times")
    return [(k, times.get(k, float(k)), paths[k], kind) for k in sorted(paths)]


def frame_covariance(entry, chans, num_freq, num_ant):
    """Per-sample covariance of the selected channels for one frame, and K."""
    _, _, path, kind = entry
    if kind == "corr":
        R = read_astron_correlation(path, num_freq, num_ant)[chans]
        return R, None
    V = read_baseband_frame(path, num_freq, num_ant)
    return correlate(V[:, chans, :].astype(np.complex128)), V.shape[0]


def stationarity_results(frames, chans, num_freq, num_ant, K):
    """#9 per channel: two passes (mean, then differences to the mean)."""
    M = len(frames)
    if M < 3:
        raise SystemExit("--multi-frame needs at least three frames")
    R_sum, prev, z_consec = None, None, []
    for e in frames:
        R, k_raw = frame_covariance(e, chans, num_freq, num_ant)
        R = R / (k_raw or K)
        R_sum = R if R_sum is None else R_sum + R
        if prev is not None:
            z_consec.append([ck.whitened_change(prev[i], R[i], 0.5 * (prev[i] + R[i]), K / 2)
                             for i in range(len(chans))])
        prev = R
    R_mean = R_sum / M
    z_mean = []
    for e in frames:
        R, k_raw = frame_covariance(e, chans, num_freq, num_ant)
        R = R / (k_raw or K)
        z_mean.append([ck.whitened_change(R[i], R_mean[i], R_mean[i], K) / (1 - 1 / M)
                       for i in range(len(chans))])
    z_mean, z_consec = np.array(z_mean), np.array(z_consec)
    times = [e[1] for e in frames]
    return [ck.check_stationarity(z_mean[:, i], z_consec[:, i], times_s=times, channel=c)
            for i, c in enumerate(chans)]


def on_off_results(frames, chans, num_freq, num_ant, K, on, off, look, meta):
    """#10 per channel: on window vs two halves of the off window."""
    sel_on = [e for e in frames if on[0] <= e[1] <= on[1]]
    sel_off = [e for e in frames if off[0] <= e[1] <= off[1]]
    if not sel_on or len(sel_off) < 2:
        raise SystemExit(f"#10 needs >= 1 on and >= 2 off frames "
                         f"(found {len(sel_on)} and {len(sel_off)})")
    half = len(sel_off) // 2

    def mean_cov(sel):
        acc = None
        for e in sel:
            R, k_raw = frame_covariance(e, chans, num_freq, num_ant)
            R = R / (k_raw or K)
            acc = R if acc is None else acc + R
        return acc / len(sel)

    R_on = mean_cov(sel_on)
    R_a, R_b = mean_cov(sel_off[:half]), mean_cov(sel_off[half: 2 * half])
    x, y, fr = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"], meta["frequencies_mhz"]
    out = []
    for i, c in enumerate(chans):
        d = steering_vectors(*look, fr[c] * 1e6, x, y)[0]
        out.append(ck.check_on_off(R_on[i], len(sel_on) * K, R_a[i], R_b[i], half * K, d,
                                   channel=c))
    print(f"#10: {len(sel_on)} on frames, 2 x {half} off frames")
    return out


def main():
    args = parse_args()
    meta = read_window_meta(args.frame)
    num_freq = args.num_freq or (len(meta["frequencies_mhz"]) if meta else None)
    num_ant = args.num_ant or (len(meta["antenna_pos_x_m"]) if meta else None)
    if meta is None or num_freq is None or num_ant is None:
        raise SystemExit("Window metadata not found next to the frame.")

    V = read_baseband_frame(args.frame, num_freq, num_ant)
    K = V.shape[0]
    chans = args.channels if args.channels is not None else list(range(num_freq))
    expected = parse_expected(args.expected_ch)

    B = float(np.median(np.diff(meta["frequencies_mhz"]))) * 1e6
    global_results = [
        ck.check_narrowband(meta["antenna_pos_x_m"], meta["antenna_pos_y_m"], B),
        ck.check_sample_support(K, num_ant),
    ]

    R = correlate(V[:, chans, :], normalize=True)
    per_channel = []
    x, y, fr = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"], meta["frequencies_mhz"]
    for c, Rc in zip(chans, R):
        sc = ck.check_source_count(Rc, expected=expected.get(c, args.expected), channel=c)
        per_channel += [
            sc,
            ck.check_degrees_of_freedom(sc.value, num_ant, channel=c),
            ck.check_noise_correlation(Rc, K, channel=c),
            ck.check_antenna_health(Rc, channel=c),
            ck.check_saturation(V[:, c, :], channel=c),
            ck.check_conditioning(Rc, channel=c),
        ]
        if args.look is not None:
            per_channel.append(ck.check_spatial_separation(
                args.look, args.interferer, fr[c] * 1e6, x, y, channel=c))

    if args.multi_frame or args.on is not None:
        frames = window_frames(args.frame, args.corr_dir, meta)
        if args.multi_frame:
            sel = frames if args.t_range is None else [
                e for e in frames if args.t_range[0] <= e[1] <= args.t_range[1]]
            print(f"#9: {len(sel)} frames ({sel[0][3]})")
            per_channel += stationarity_results(sel, chans, num_freq, num_ant, K)
        if args.on is not None:
            if args.off is None or args.look is None:
                raise SystemExit("--on requires --off and --look")
            per_channel += on_off_results(frames, chans, num_freq, num_ant, K,
                                          args.on, args.off, args.look, meta)

    print(f"Frame: {args.frame}")
    print(f"K = {K} samples, N = {num_ant} antennas, {len(chans)} channel(s)\n")
    print("Global checks")
    print(ck.format_results(global_results))
    print("\nPer-channel checks (summary)")
    print(summarize(per_channel))
    if args.verbose:
        print("\nPer-channel checks (detail)")
        print(ck.format_results(per_channel))


if __name__ == "__main__":
    main()
