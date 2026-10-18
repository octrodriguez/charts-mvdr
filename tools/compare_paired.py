#!/usr/bin/env python
"""Paired-window comparison: S/N with RFI against the same window without it.

Two windows generated with the same seed, scene and capture schedule, one of
them with the RFI amplitudes set to zero, share every frame and random draw
except the RFI. Running the same analysis on both gives a ground truth per
channel: the S/N in the RFI window relative to the conventional (DFT) S/N of
the clean window,

    rel_db(beam, ch) = 10 log10( S/N_rfi(beam, ch) / S/N_clean(dft, ch) ),

so 0 dB means full recovery. Channels are grouped by the RFI events that
cover them (event ids from the events file of the RFI window; --group renames
or merges them), channels without RFI form the group "clean".

Inputs are the output directories of the same script run on both windows:
  09_beamform_waterfall.py  waterfall.npz (per-channel S/N, off-pulse std/mean,
                            dedispersed S/N); the detection metric
  08_transient_snr.py       transient_snr.npz (S/N = S / Q, mean-power metric)

Pairing check: in the clean channels the RFI window must reproduce the clean
window (same frames and noise), |S/N_rfi(dft) / S/N_clean(dft) - 1| < --tol.
If it does not, the windows are not paired and the script exits with status 2.

Outputs: a per-group table on stdout (median rel_db per beam, channels with
non-positive S/N counted as lost, median off-pulse std/mean for 09), the
dedispersed S/N per beam for 09, and with --csv a per-channel table.

Example:
  python tools/compare_paired.py --rfi RFI_OUT --clean CLEAN_OUT \\
      --window-dir RFI_WINDOW_DIR --group A=rfi_01,rfi_05 --csv paired.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from charts_mvdr.data_io import load_events_file

KINDS = {"waterfall.npz": "09", "transient_snr.npz": "08"}


def load_result(path: Path) -> dict:
    """Per-channel S/N for each beam from a 09 or 08 output directory (or npz)."""
    path = Path(path)
    if path.is_dir():
        found = [path / f for f in KINDS if (path / f).exists()]
        if not found:
            raise SystemExit(f"no {' or '.join(KINDS)} in {path}")
        path = found[0]
    kind = KINDS.get(path.name)
    if kind is None:
        raise SystemExit(f"unknown result file {path.name}")
    z = np.load(path, allow_pickle=False)
    out = {"kind": kind, "channels": z["channels"].astype(int), "snr": {}, "cv": {}, "dd": {}}
    if kind == "09":
        out["flag"] = z["flag"].astype(bool)
        for b in [str(x) for x in z["beams"]]:
            out["snr"][b] = z[f"{b}_snr_ch"].astype(float)
            out["cv"][b] = z[f"{b}_sd_off"] / z[f"{b}_mu_off"]
            out["dd"][b] = float(z[f"{b}_dd_snr"])
    else:
        out["flag"] = np.zeros(out["channels"].size, dtype=bool)
        S, Q = z["S"].astype(float), z["Q"].astype(float)
        for i, b in enumerate(str(x) for x in z["weights"]):
            out["snr"][b] = S[:, i] / Q[:, i]
    return out


def channel_groups(events, rename=None) -> dict:
    """{channel: group label} from the RFI events; several events join with '+'."""
    rename = rename or {}
    by_ch = {}
    for ev in events:
        if str(ev.get("event_type", "")).startswith("rfi") and ev.get("channels"):
            lab = rename.get(ev["event_id"], ev["event_id"])
            for ch in ev["channels"]:
                by_ch.setdefault(int(ch), set()).add(lab)
    return {ch: "+".join(sorted(labs)) for ch, labs in by_ch.items()}


def parse_groups(specs) -> dict:
    """--group NAME=id1,id2 ... -> {event_id: NAME}."""
    out = {}
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--group {s!r} is not NAME=id1,id2")
        name, ids = s.split("=", 1)
        for i in ids.split(","):
            if i:
                out[i.strip()] = name.strip()
    return out


def rel_db(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(num, float) / np.asarray(den, float)
    return np.where(r > 0, 10 * np.log10(np.where(r > 0, r, 1.0)), np.nan)


def compare(rfi: dict, clean: dict, groups: dict, tol: float):
    if rfi["kind"] != clean["kind"]:
        raise SystemExit(f"result kinds differ: {rfi['kind']} vs {clean['kind']}")
    common = np.intersect1d(rfi["channels"], clean["channels"])
    ir = np.searchsorted(rfi["channels"], common)
    ic = np.searchsorted(clean["channels"], common)
    if "dft" not in clean["snr"] or "dft" not in rfi["snr"]:
        raise SystemExit("both results need the dft beam")
    ref = clean["snr"]["dft"][ic]
    flag = rfi["flag"][ir] | clean["flag"][ic]
    label = np.array([groups.get(int(c), "clean") for c in common])
    beams = [b for b in rfi["snr"]]
    rel = {b: rel_db(rfi["snr"][b][ir], ref) for b in beams}
    cv = {b: rfi["cv"][b][ir] for b in rfi["cv"]}
    ok = (label == "clean") & ~flag & np.isfinite(ref) & (ref > 0)
    dev = np.abs(rfi["snr"]["dft"][ir][ok] / ref[ok] - 1.0) if ok.any() else np.array([np.nan])
    return dict(channels=common, label=label, flag=flag, ref=ref, rel=rel, cv=cv, beams=beams,
                pair_dev=float(np.nanmax(dev)), n_clean=int(ok.sum()),
                paired=bool(ok.any() and np.nanmax(dev) < tol))


def group_order(labels):
    labs = sorted(set(labels) - {"clean"})
    return labs + (["clean"] if "clean" in set(labels) else [])


def summarize(res, rfi, clean):
    beams = res["beams"]
    wd = max(12, max(len(b) for b in beams) + 2)
    head = f"{'group':<16}{'n':>4}" + "".join(f"{b:>{wd}}" for b in beams) + f"{'lost':>6}"
    if res["cv"]:
        head += "".join(f"{'sd/mu ' + b:>16}" for b in res["cv"])
    print("median S/N relative to the clean window, DFT [dB] (flagged channels excluded; "
          "lost: DFT S/N <= 0)")
    print(head)
    for g in group_order(res["label"]):
        sel = (res["label"] == g) & ~res["flag"]
        if not sel.any():
            continue
        lost = int(np.sum(~np.isfinite(res["rel"]["dft" if "dft" in beams else beams[0]][sel])))
        line = f"{g:<16}{int(sel.sum()):>4}"
        for b in beams:
            v = res["rel"][b][sel]
            line += f"{np.nanmedian(v):>{wd}.2f}" if np.isfinite(v).any() else f"{'lost':>{wd}}"
        line += f"{lost:>6}"
        for b in res["cv"]:
            line += f"{np.nanmedian(res['cv'][b][sel]):>16.3f}"
        print(line)
    if rfi["dd"] and "dft" in clean["dd"]:
        ref = clean["dd"]["dft"]
        print(f"dedispersed S/N: clean dft {ref:.1f}; " + ", ".join(
            f"{b} {v:.1f} ({10 * np.log10(v / ref):+.2f} dB)" for b, v in rfi["dd"].items()
            if v > 0))


def write_csv(res, path):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["channel", "group", "flag", "snr_clean_dft"]
                    + [f"rel_db_{b}" for b in res["beams"]]
                    + [f"sd_mu_{b}" for b in res["cv"]])
        for i, ch in enumerate(res["channels"]):
            wr.writerow([int(ch), res["label"][i], int(res["flag"][i]), f"{res['ref'][i]:.4g}"]
                        + [f"{res['rel'][b][i]:.3f}" for b in res["beams"]]
                        + [f"{res['cv'][b][i]:.4f}" for b in res["cv"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rfi", type=Path, required=True, help="output dir (or npz) of the RFI window")
    ap.add_argument("--clean", type=Path, required=True,
                    help="output dir (or npz) of the paired window without RFI")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--window-dir", type=Path, help="RFI window dir (events file inside)")
    src.add_argument("--events", type=Path, help="events JSON of the RFI window")
    ap.add_argument("--group", action="append", default=[], metavar="NAME=ID1,ID2",
                    help="rename or merge RFI events into a group (repeatable)")
    ap.add_argument("--tol", type=float, default=1e-3, help="pairing tolerance on clean channels")
    ap.add_argument("--csv", type=Path, default=None, help="per-channel table")
    args = ap.parse_args(argv)

    if args.events:
        import json
        events = json.loads(args.events.read_text())
    else:
        events = load_events_file(args.window_dir)
    if not events:
        raise SystemExit("no events found")
    rfi, clean = load_result(args.rfi), load_result(args.clean)
    res = compare(rfi, clean, channel_groups(events, parse_groups(args.group)), args.tol)
    print(f"{rfi['kind']} results, {res['channels'].size} common channels; pairing check on "
          f"{res['n_clean']} clean channels: max |S/N ratio - 1| = {res['pair_dev']:.2e} "
          f"({'paired' if res['paired'] else 'NOT PAIRED'})")
    summarize(res, rfi, clean)
    if args.csv:
        write_csv(res, args.csv)
        print(f"wrote {args.csv}")
    return 0 if res["paired"] else 2


if __name__ == "__main__":
    sys.exit(main())
