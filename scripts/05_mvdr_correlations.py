#!/usr/bin/env python
"""Two-stage DFT localization + MVDR beamforming on correlation dumps.

Stage 1 (localization): DFT image summed over clean channels -> look direction
(brightest source), unless --look is given. DFT and Capon images are also made
for each analysis channel to locate interferers.
The coarse image peak is refined with successively finer local grids
(beamforming.refine_peak): a look direction on the coarse grid node is off by up
to half a grid step, enough for MVDR to partially null the source itself.
Stage 2 (MVDR): for each analysis channel, DFT and MVDR weights towards the look
direction; output power and null depth towards each interferer.

Power budget: the powers of the look-direction source and the interferers are
fitted to the cross-correlations (beamforming.fit_source_powers; the diagonal is
excluded, so noise does not bias them). The output power of each beamformer is
then split as P = noise + signal + leak + residual (beamforming.power_budget):
  noise     sum_n |w_n|^2 sigma_n^2, with the measured per-antenna noise
  leak      sum_i p_i |w^H a_i|^2 over the interferers
  signal    P - noise - leak, the noise-free output attributed to the look source
  retained  signal / p_look: 1 for an ideal beamformer; < 1 for MVDR self-nulling
            (steering mismatch, finite K); > 1 when unmodelled power leaks in.
  SINR      p_look / (noise + leak), output SINR of the modelled sources (both
            beamformers have unit gain towards the look direction). The header of
            each channel gives the interference-free DFT value p_look / (noise / N).
With --p-look-ref clean, p_look is not fitted in the analysis channel but taken
as the median fitted power over the nearest clean channels. Needed when an
interferer is close to a grating lobe of the look direction: the steering
vectors are then nearly collinear and the joint fit cannot split the power
between the look source and the interferer.
P_MVDR / P_DFT alone is ambiguous: it drops both when MVDR loses signal and when
the DFT beam collects interference through its sidelobes. For moving sources
(LEO) the fixed event position is only approximate, and the fit and leak are
biased; use short time ranges. Under heavy 4-bit saturation (rail fraction
~0.5) the quantization distortion is correlated between antennas: MVDR partly
suppresses it, so the white-noise term overestimates the MVDR noise and
retained underestimates the signal (see 03_validate_mvdr.py --sweep leo, which
compares it with the true retention).

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

from charts_mvdr.beamforming import (beam_power, dft_weights, fit_source_powers, lm_grid,
                                     mvdr_weights, power_budget, refine_peak,
                                     steering_vectors)
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
    ap.add_argument("--no-refine", action="store_true",
                    help="keep the coarse grid peak as look direction (no local refinement)")
    ap.add_argument("--p-look-ref", choices=["fit", "clean"], default="fit",
                    help="look-source power: fitted in each channel (default) or the "
                         "median fit over the nearest clean channels")
    ap.add_argument("--clean-channels", type=int, nargs="+", default=None,
                    help="clean channels for --p-look-ref clean (default: channels "
                         "without interferers in the events file)")
    ap.add_argument("--clean-neighbours", type=int, default=8,
                    help="number of nearest clean channels for --p-look-ref clean")
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
        print(f"look direction (DFT peak over {len(clean)} clean channels, grid): "
              f"l = {look[0]:+.4f}, m = {look[1]:+.4f}")
        if not args.no_refine:
            step = 2.0 / (args.grid - 1)
            look = refine_peak(
                lambda L, M: sum(image(R[c], freq_hz[c], x, y, L, M) for c in clean),
                *look, step=step)
            print(f"look direction (refined):                         "
                  f"l = {look[0]:+.4f}, m = {look[1]:+.4f}")
    else:
        look = tuple(args.look)
        print(f"look direction (given): l = {look[0]:+.4f}, m = {look[1]:+.4f}")

    # Look-source power from the nearest clean channels (--p-look-ref clean)
    p_look_clean = {}
    if args.p_look_ref == "clean":
        if args.clean_channels is not None:
            clean_ch = sorted(set(args.clean_channels))
        elif events:
            clean_ch = [c for c in range(num_freq) if c not in rfi_by_ch]
        else:
            raise SystemExit("--p-look-ref clean needs --clean-channels or an events file")
        for ch in args.channels:
            near = sorted(clean_ch, key=lambda c: (abs(c - ch), c))[:args.clean_neighbours]
            fits = [fit_source_powers(R[c], steering_vectors(*look, freq_hz[c], x, y))[0][0]
                    for c in near]
            p_look_clean[ch] = float(np.median(fits))
        print(f"p_look from the {args.clean_neighbours} nearest clean channels "
              f"(of {len(clean_ch)})")

    # Stage 2: MVDR per channel, with the power budget of each beamformer
    args.out.mkdir(parents=True, exist_ok=True)
    print("\nPowers in units of R per sample; signal = P - noise - leak; "
          "retained = signal / p_look")
    print(f"\n{'ch':>4} {'MHz':>7} {'beam':<5} {'P':>8} {'noise':>8} {'leak':>8} "
          f"{'signal':>8} {'retained':>8} {'WNG dB':>7} {'SINR dB':>7}  nulls [dB]")
    summary = {k: [] for k in ("ch", "freq_mhz", "p_look", "p_int", "int_lm",
                               "noise_mean", "sinr_ref_db", "dft", "mvdr")}
    for ch in args.channels:
        f = freq_hz[ch]
        d = steering_vectors(*look, f, x, y)
        w_dft = dft_weights(d)[0]
        w_mv = mvdr_weights(R[ch], d, loading=args.loading, loading_ref=args.loading_ref)[0]
        if events:   # simulation: ground truth; clean channels have no interferer
            ints = rfi_by_ch.get(ch, [])
        else:        # real data: strongest DFT peak outside the main beam of the look direction
            img = image(R[ch], f, x, y, gl, gm)
            extent = np.ptp(np.hypot(x - x.mean(), y - y.mean())) * 2 or 1.0
            far = np.hypot(gl - look[0], gm - look[1]) > 2.0 * (3e8 / f) / extent
            ints = [(*peak(np.where(far, img, -np.inf), gl, gm), "dft-peak")]

        A_int = (steering_vectors([i[0] for i in ints], [i[1] for i in ints], f, x, y)
                 if ints else np.zeros((0, num_ant), dtype=complex))
        A = np.vstack([d, A_int])
        fixed = {0: p_look_clean[ch]} if ch in p_look_clean else None
        p, noise = fit_source_powers(R[ch], A, fixed=fixed)
        sinr_ref = 10 * np.log10(max(p[0], 1e-30) * num_ant / noise.mean())
        p_int_str = ", ".join(f"{pi:.3g} at ({li:+.3f}, {mi:+.3f}) {kind}"
                              for pi, (li, mi, kind) in zip(p[1:], ints)) or "none"
        print(f"{ch:4d} {f / 1e6:7.1f} fit: p_look = {p[0]:.4f}"
              f"{' (clean channels)' if fixed else ''}, "
              f"noise = {noise.mean():.3f} (per antenna {noise.min():.2f}-{noise.max():.2f}); "
              f"interferers: {p_int_str}; interference-free DFT SINR {sinr_ref:.2f} dB")
        beams = {}
        for name, w in (("DFT", w_dft), ("MVDR", w_mv)):
            b = power_budget(R[ch], w, A, p, noise)
            leak = float(b["source"][1:].sum())
            signal = b["total"] - b["noise"] - leak
            nulls = 10 * np.log10(np.abs(A_int.conj() @ w) ** 2 + 1e-30)
            sinr = 10 * np.log10(max(p[0], 1e-30) / (b["noise"] + leak))
            beams[name] = {"P": b["total"], "noise": b["noise"], "leak": leak,
                           "signal": signal, "retained": signal / p[0],
                           "wng_db": 10 * np.log10(b["wng"]), "sinr_db": sinr,
                           "nulls_db": nulls}
            null_str = " ".join(f"{v:6.1f}" for v in nulls) or "-"
            print(f"{'':4} {'':7} {name:<5} {b['total']:8.4f} {b['noise']:8.4f} {leak:8.4f} "
                  f"{signal:8.4f} {signal / p[0]:8.3f} {beams[name]['wng_db']:7.2f} "
                  f"{sinr:7.2f}  {null_str}")
        summary["ch"].append(ch); summary["freq_mhz"].append(f / 1e6)
        summary["p_look"].append(p[0]); summary["p_int"].append(p[1:])
        summary["int_lm"].append([(li, mi) for li, mi, _ in ints])
        summary["noise_mean"].append(noise.mean())
        summary["sinr_ref_db"].append(sinr_ref)
        summary["dft"].append(beams["DFT"]); summary["mvdr"].append(beams["MVDR"])

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

    flat = {"ch": np.array(summary["ch"]), "freq_mhz": np.array(summary["freq_mhz"]),
            "p_look": np.array(summary["p_look"]), "noise_mean": np.array(summary["noise_mean"]),
            "sinr_ref_db": np.array(summary["sinr_ref_db"])}
    for beam in ("dft", "mvdr"):
        for key in ("P", "noise", "leak", "signal", "retained", "wng_db", "sinr_db"):
            flat[f"{beam}_{key}"] = np.array([b[key] for b in summary[beam]])
    np.savez(args.out / "mvdr_summary.npz", **flat, look=np.array(look), frames=np.array(idx),
             loading=args.loading, loading_ref=args.loading_ref, p_look_ref=args.p_look_ref,
             p_int=np.array(summary["p_int"], dtype=object),
             int_lm=np.array(summary["int_lm"], dtype=object),
             nulls_dft=np.array([b["nulls_db"] for b in summary["dft"]], dtype=object),
             nulls_mvdr=np.array([b["nulls_db"] for b in summary["mvdr"]], dtype=object))
    print(f"\nfigures and summary -> {args.out}")


if __name__ == "__main__":
    main()
