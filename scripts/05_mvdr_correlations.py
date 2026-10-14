#!/usr/bin/env python
"""Two-stage DFT localization + MVDR beamforming on correlation dumps.

Stage 1 (localization): DFT image summed over clean channels -> look direction
(brightest source), unless --look is given. DFT and Capon images are also made
for each analysis channel to locate interferers.
Stage 2 (MVDR): for each analysis channel, DFT and MVDR weights towards the look
direction; output power and null depth towards each interferer.

Inputs are Kotekan cudaCorrelatorAstron dumps (corr_%07d.bin, read with
read_astron_correlation) and the window metadata (<window>_meta.h5) with antenna
positions, frequencies and frame timestamps. Interferer directions are taken from
the events file when available (ground truth for simulations) and otherwise from
the DFT image of each channel.

Example:
  python scripts/05_mvdr_correlations.py \\
      --corr-dir $SCRATCH/data/kotekan_corr/charts_sim_64ant_<jobid> \\
      --window-dir $SCRATCH/data/sim_test/clean_v1/charts_sim_64ant \\
      --channels 94 95 133 134 147 148 --out $SCRATCH/data/figs/mvdr_kotekan
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import (beam_power, dft_weights, lm_grid,
                                     mvdr_weights, steering_vectors)
from charts_mvdr.data_io import read_astron_correlation

CORR_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{7})\.bin$")


def load_meta(window_dir: Path) -> dict:
    meta_path = next(window_dir.glob("*_meta.h5"))
    with h5py.File(meta_path, "r") as f:
        meta = {k: f[k][:] for k in ("antenna_pos_x_m", "antenna_pos_y_m", "frequencies_mhz")}
        if "frames/timestamp_s" in f:
            meta["frame_time_s"] = f["frames/timestamp_s"][:]
            meta["frame_out_idx"] = f["frames/output_file_index"][:]
    return meta


def load_events(window_dir: Path) -> list[dict]:
    files = sorted(window_dir.glob("*events*.json"))
    if not files:
        return []
    events = json.loads(files[0].read_text())
    return events if isinstance(events, list) else []


def mean_covariance(corr_dir, indices, num_freq, num_ant, samples):
    """Average of the selected dumps, normalized to power per sample."""
    R = np.zeros((num_freq, num_ant, num_ant), dtype=np.complex128)
    for p in indices:
        R += read_astron_correlation(p, num_freq, num_ant)
    return R / (len(indices) * samples)


def image(R, freq_hz, x, y, l, m, kind="dft"):
    A = steering_vectors(l, m, freq_hz, x, y)
    if kind == "dft":
        return beam_power(R, dft_weights(A))
    Rinv_A = np.linalg.solve(R, A.T)                       # Capon: 1 / (a^H R^-1 a)
    return 1.0 / np.real(np.einsum("in,in->n", A.T.conj(), Rinv_A))


def peak(img, l, m):
    k = int(np.argmax(img))
    return float(l[k]), float(m[k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corr-dir", type=Path, required=True)
    ap.add_argument("--window-dir", type=Path, required=True,
                    help="directory with <window>_meta.h5 and the events file")
    ap.add_argument("--channels", type=int, nargs="+", required=True,
                    help="local channel indices to analyse")
    ap.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"),
                    help="look direction; default: DFT peak over clean channels")
    ap.add_argument("--t-range", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="use only frames with timestamp in [T0, T1] s")
    ap.add_argument("--loading", type=float, default=0.0,
                    help="diagonal loading, in units of --loading-ref (e.g. 10 with 'noise')")
    ap.add_argument("--loading-ref", choices=["trace", "noise", "absolute"], default="noise",
                    help="reference for --loading (default: noise floor, median eigenvalue)")
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--grid", type=int, default=161)
    ap.add_argument("--out", type=Path, default=Path("mvdr_correlations"))
    args = ap.parse_args()

    meta = load_meta(args.window_dir)
    x, y = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"]
    freq_hz = meta["frequencies_mhz"] * 1e6
    num_ant, num_freq = x.size, freq_hz.size
    events = load_events(args.window_dir)

    dumps = {int(m["index"]): p for p in sorted(args.corr_dir.glob("*.bin"))
             if (m := CORR_RE.match(p.name))}
    idx = sorted(dumps)
    if args.t_range and "frame_time_s" in meta:
        t = dict(zip(meta["frame_out_idx"].astype(int), meta["frame_time_s"]))
        idx = [k for k in idx if args.t_range[0] <= t.get(k, -1) <= args.t_range[1]]
    if not idx:
        raise SystemExit("no correlation dumps selected")
    print(f"{len(idx)} dumps, {num_ant} antennas, {num_freq} channels "
          f"({freq_hz[0] / 1e6:.1f}-{freq_hz[-1] / 1e6:.1f} MHz)")
    R = mean_covariance(args.corr_dir, [dumps[k] for k in idx], num_freq, num_ant, args.samples)

    # Interferers from the events file: {channel: [(l, m, type)]}
    rfi_by_ch: dict[int, list] = {}
    for ev in events:
        if str(ev.get("event_type", "")).startswith("rfi") and ev.get("channels"):
            for ch in ev["channels"]:
                rfi_by_ch.setdefault(int(ch), []).append(
                    (float(ev["l0"]), float(ev["m0"]), ev["event_type"]))
    if rfi_by_ch:
        print("interferers from events:",
              {ch: [(round(a, 3), round(b, 3), t) for a, b, t in v] for ch, v in sorted(rfi_by_ch.items())})

    # Stage 1: localization
    gl, gm, mask = lm_grid(args.grid)
    if args.look is None:
        clean = [c for c in range(num_freq) if c not in rfi_by_ch][::8]
        img = sum(image(R[c], freq_hz[c], x, y, gl, gm) for c in clean)
        look = peak(img, gl, gm)
        print(f"look direction (DFT peak over {len(clean)} clean channels): "
              f"l = {look[0]:+.3f}, m = {look[1]:+.3f}")
    else:
        look = tuple(args.look)
        print(f"look direction (given): l = {look[0]:+.3f}, m = {look[1]:+.3f}")

    # Stage 2: MVDR per channel
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'ch':>4} {'MHz':>7} {'P_DFT':>9} {'P_MVDR':>9} {'MVDR/DFT':>9} "
          f"{'null_DFT dB':>11} {'null_MVDR dB':>12}  interferer (l, m)")
    rows = []
    for ch in args.channels:
        f = freq_hz[ch]
        d = steering_vectors(*look, f, x, y)
        w_dft = dft_weights(d)[0]
        w_mv = mvdr_weights(R[ch], d, loading=args.loading, loading_ref=args.loading_ref)[0]
        p_dft = beam_power(R[ch], w_dft[None])[0]
        p_mv = beam_power(R[ch], w_mv[None])[0]
        if events:   # simulation: ground truth; clean channels have no interferer
            ints = rfi_by_ch.get(ch, [])
        else:        # real data: strongest DFT peak outside the main beam of the look direction
            img = image(R[ch], f, x, y, gl, gm)
            extent = np.ptp(np.hypot(x - x.mean(), y - y.mean())) * 2 or 1.0
            far = np.hypot(gl - look[0], gm - look[1]) > 2.0 * (3e8 / f) / extent
            ints = [(*peak(np.where(far, img, -np.inf), gl, gm), "dft-peak")]
        if not ints:
            print(f"{ch:4d} {f / 1e6:7.1f} {p_dft:9.4f} {p_mv:9.4f} {p_mv / p_dft:9.3f} "
                  f"{'-':>11} {'-':>12}  (no interferer)")
            rows.append((ch, f / 1e6, p_dft, p_mv, np.nan, np.nan, np.nan, np.nan))
        for li, mi, kind in ints:
            a = steering_vectors(li, mi, f, x, y)[0]
            n_dft = 10 * np.log10(np.abs(np.vdot(w_dft, a)) ** 2 + 1e-30)
            n_mv = 10 * np.log10(np.abs(np.vdot(w_mv, a)) ** 2 + 1e-30)
            print(f"{ch:4d} {f / 1e6:7.1f} {p_dft:9.4f} {p_mv:9.4f} {p_mv / p_dft:9.3f} "
                  f"{n_dft:11.1f} {n_mv:12.1f}  ({li:+.3f}, {mi:+.3f}) {kind}")
            rows.append((ch, f / 1e6, p_dft, p_mv, n_dft, n_mv, li, mi))

        # Figure: DFT and Capon images with look and interferer directions
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        n = args.grid
        for ax, kind in zip(axes, ("dft", "capon")):
            full = np.full(mask.shape, np.nan)
            full[mask] = 10 * np.log10(np.maximum(image(R[ch], f, x, y, gl, gm, kind), 1e-12))
            im = ax.imshow(full, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis")
            ax.plot(*look, "w*", ms=12, label="look")
            for li, mi, _ in ints:
                ax.plot(li, mi, "rx", ms=10, mew=2, label="interferer")
            ax.set_title(f"ch {ch} ({f / 1e6:.1f} MHz): {kind.upper()} [dB]")
            ax.set_xlabel("l"); ax.set_ylabel("m")
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[0].legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"images_ch{ch:03d}.png", dpi=130)
        plt.close(fig)

    np.savez(args.out / "mvdr_summary.npz", rows=np.array(rows), look=np.array(look),
             frames=np.array(idx), loading=args.loading, loading_ref=args.loading_ref)
    print(f"\nfigures and summary -> {args.out}")


if __name__ == "__main__":
    main()
