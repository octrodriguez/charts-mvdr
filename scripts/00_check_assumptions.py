#!/usr/bin/env python3
"""Pre-flight check of the assumptions required for MVDR beamforming.

Runs the checks in charts_mvdr.checks on one baseband frame. Global checks run
once; per-channel checks are summarized over channels (worst status, counts and
failing channels). Use --verbose for the full per-channel table.

Example:
    python 00_check_assumptions.py <window>_0000000.bin \
        --expected 1 --expected-ch 94=2 133=2 147=2
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from charts_mvdr import checks as ck
from charts_mvdr.data_io import read_baseband_frame, read_window_meta
from charts_mvdr.np_correlator import correlate

ORDER = [ck.Status.FAIL, ck.Status.WARN, ck.Status.PASS, ck.Status.NA]


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
    for c, Rc in zip(chans, R):
        per_channel += [
            ck.check_source_count(Rc, expected=expected.get(c, args.expected), channel=c),
            ck.check_noise_correlation(Rc, K, channel=c),
            ck.check_antenna_health(Rc, channel=c),
            ck.check_saturation(V[:, c, :], channel=c),
            ck.check_conditioning(Rc, channel=c),
        ]

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