#!/usr/bin/env python
"""Beam response of the DFT and MVDR beamformers over the sky (l, m).

For each channel the weights towards a fixed look direction are computed from
the correlation dumps, and the power response

    B(l, m) = |w^H a(l, m)|^2,     w^H d = 1,

is mapped over the visible region l^2 + m^2 < 1 (0 dB at the look direction).
The DFT beam (w = d / N) shows the main lobe, the sidelobes and, on a regular
grid with spacing above lambda / 2, the grating lobes. The MVDR beam
(w = F^-1 d / d^H F^-1 d) keeps unit gain towards the look direction and places
nulls towards the interferers. A grating lobe has the same steering vector as
the look direction, so MVDR keeps 0 dB there too and cannot null an interferer
that sits on it.

Weights covariance F: by default the average of the same dumps used for the
analysis (--t-range). With --cov-t-range and/or --cov-exclude the covariance is
taken from other frames, e.g. the signal-free (off-pulse) frames of a transient
for the KL filter of Andrew et al. (b proportional to F^-1 d). With exact steering
and infinite integration F^-1 d and R^-1 d give the same normalized weights
(matrix inversion lemma); they differ under steering errors (self-nulling of
the source) and finite integration.

Interferer directions come from the events file in --window-dir (simulations)
or from --interferer L M (repeatable). Grating lobes of the look direction are
marked when the layout is a rectangular grid (spacing inferred, or --spacing).

Outputs in --out: beam_response_chNNN.png per channel (maps for DFT and each
MVDR loading, plus a cut through the look direction and the first interferer
or grating lobe) and beam_response.npz with the maps and the table values.

Example:
  python scripts/07_beam_response.py --corr-dir CORR_DIR --window-dir WINDOW_DIR \\
      --channels 147 --look 0.4677 0.4801 --loading 0 10 --out OUT_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import (beam_response, dft_weights, grating_lobes, lm_grid,
                                     mvdr_weights, noise_floor, regular_grid_spacing,
                                     steering_vectors)
from charts_mvdr.data_io import (interferers_by_channel, list_correlation_dumps,
                                 load_events_file, load_window_dir_meta, mean_correlation,
                                 select_frames)


def db(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(x, floor))


def cut_axis(look, target, n=2001):
    """Directions along the line through look and target, visible part only.

    Returns s (signed distance from look, positive towards target) and l, m.
    """
    u = np.asarray(target, float) - np.asarray(look, float)
    u /= np.hypot(*u)
    s = np.linspace(-2.0, 2.0, n)
    l, m = look[0] + s * u[0], look[1] + s * u[1]
    vis = l ** 2 + m ** 2 < 1.0
    return s[vis], l[vis], m[vis], u


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corr-dir", type=Path, required=True)
    ap.add_argument("--window-dir", type=Path, required=True,
                    help="directory with <window>_meta.h5 and, optionally, the events file")
    ap.add_argument("--channels", type=int, nargs="+", required=True,
                    help="local channel indices")
    ap.add_argument("--look", type=float, nargs=2, required=True, metavar=("L", "M"),
                    help="look direction (use a refined direction, e.g. from 05)")
    ap.add_argument("--t-range", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="frames for the analysis covariance R [s since window start]")
    ap.add_argument("--cov-t-range", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="frames for the weights covariance F (default: same as R)")
    ap.add_argument("--cov-exclude", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                    help="exclude these frames from F (e.g. the on-pulse window)")
    ap.add_argument("--loading", type=float, nargs="+", default=[0.0, 10.0],
                    help="MVDR diagonal loadings, in units of --loading-ref (one map each)")
    ap.add_argument("--loading-ref", choices=["trace", "noise", "absolute"], default="noise")
    ap.add_argument("--interferer", type=float, nargs=2, action="append", default=None,
                    metavar=("L", "M"), help="interferer direction (repeatable); "
                    "overrides the events file")
    ap.add_argument("--spacing", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                    help="grid spacing for the grating lobes (default: inferred)")
    ap.add_argument("--samples", type=int, default=1536, help="samples per dump")
    ap.add_argument("--grid", type=int, default=501, help="map points per axis")
    ap.add_argument("--vmin", type=float, default=-60.0, help="lower colour limit [dB]")
    ap.add_argument("--out", type=Path, default=Path("beam_response"))
    args = ap.parse_args()

    meta = load_window_dir_meta(args.window_dir)
    x, y = meta["antenna_pos_x_m"], meta["antenna_pos_y_m"]
    freq_hz = meta["frequencies_mhz"] * 1e6
    num_ant, num_freq = x.size, freq_hz.size
    for ch in args.channels:
        if not 0 <= ch < num_freq:
            raise SystemExit(f"channel {ch} outside 0-{num_freq - 1}")
    look = tuple(args.look)

    dumps = list_correlation_dumps(args.corr_dir)
    idx_r = select_frames(dumps, meta, t_range=args.t_range)
    use_f = args.cov_t_range is not None or args.cov_exclude is not None
    idx_f = (select_frames(dumps, meta, t_range=args.cov_t_range, exclude=args.cov_exclude)
             if use_f else idx_r)
    if not idx_r or not idx_f:
        raise SystemExit("no correlation dumps selected")
    print(f"{num_ant} antennas, {num_freq} channels; R from {len(idx_r)} dumps, "
          f"F from {len(idx_f)} dumps{' (separate frames)' if use_f else ' (same as R)'}")
    R = mean_correlation([dumps[k] for k in idx_r], num_freq, num_ant, args.samples,
                         channels=args.channels)
    F = (mean_correlation([dumps[k] for k in idx_f], num_freq, num_ant, args.samples,
                          channels=args.channels) if use_f else R)

    if args.interferer:
        ints_fixed = [(l, m, "given") for l, m in args.interferer]
        rfi = {}
    else:
        ints_fixed = None
        rfi = interferers_by_channel(load_events_file(args.window_dir))
    spacing = tuple(args.spacing) if args.spacing else regular_grid_spacing(x, y)
    print(f"look direction: l = {look[0]:+.4f}, m = {look[1]:+.4f}; grid spacing: "
          f"{'%.3f x %.3f m' % spacing if spacing else 'not a regular grid'}")

    gl, gm, mask = lm_grid(args.grid)
    args.out.mkdir(parents=True, exist_ok=True)
    beams = ["DFT"] + [f"MVDR d={ld:g}" for ld in args.loading]
    save = {"look": np.array(look), "channels": np.array(args.channels),
            "freq_mhz": freq_hz[args.channels] / 1e6, "loading": np.array(args.loading),
            "loading_ref": args.loading_ref, "frames_r": np.array(idx_r),
            "frames_f": np.array(idx_f), "grid_mask": mask, "beams": np.array(beams)}

    print("\nResponse |w^H a|^2 in dB relative to the look direction; "
          "WNG = N ||w||^2 (0 dB for DFT)")
    for c, ch in enumerate(args.channels):
        f = freq_hz[ch]
        d = steering_vectors(*look, f, x, y)
        ints = ints_fixed if ints_fixed is not None else rfi.get(ch, [])
        lobes = (grating_lobes(*look, f, *spacing) if spacing else [])
        W = [dft_weights(d)[0]] + [
            mvdr_weights(F[c], d, loading=ld, loading_ref=args.loading_ref)[0]
            for ld in args.loading]
        sig_n = noise_floor(F[c])
        print(f"\nch {ch} ({f / 1e6:.1f} MHz), noise floor of F {sig_n:.3f}; "
              f"interferers: {[(round(a, 3), round(b, 3)) for a, b, _ in ints] or 'none'}; "
              f"grating lobes: {[(p, q, round(a, 3), round(b, 3)) for p, q, a, b in lobes] or 'none'}")
        print(f"  {'beam':<12} {'WNG dB':>7} {'P out':>9}  "
              f"{'interferers [dB]':<24} grating lobes [dB]")
        maps = []
        for name, w in zip(beams, W):
            maps.append(db(beam_response(w, gl, gm, f, x, y)).astype(np.float32))
            wng = 10 * np.log10(num_ant * np.sum(np.abs(w) ** 2))
            p_out = float(np.real(np.vdot(w, R[c] @ w)))
            r_int = [db(beam_response(w, a, b, f, x, y))[0] for a, b, _ in ints]
            r_lob = [db(beam_response(w, a, b, f, x, y))[0] for _, _, a, b in lobes]
            print(f"  {name:<12} {wng:7.2f} {p_out:9.4f}  "
                  f"{' '.join(f'{v:6.1f}' for v in r_int) or '-':<24} "
                  f"{' '.join(f'{v:6.1f}' for v in r_lob) or '-'}")
            save[f"ch{ch}_{name.replace(' ', '_').replace('=', '')}_wng_db"] = wng
            save[f"ch{ch}_{name.replace(' ', '_').replace('=', '')}_int_db"] = np.array(r_int)
        save[f"ch{ch}_maps_db"] = np.stack(maps)
        save[f"ch{ch}_int_lm"] = np.array([(a, b) for a, b, _ in ints]).reshape(-1, 2)
        save[f"ch{ch}_lobes_lm"] = np.array([(a, b) for _, _, a, b in lobes]).reshape(-1, 2)

        # Figure: one map per beam plus a cut through look and interferer / lobe
        nb = len(beams)
        vmax = max(0.0, float(np.ceil(max(float(mp.max()) for mp in maps))))
        fig, axes = plt.subplots(1, nb + 1, figsize=(4.4 * (nb + 1), 4.3), layout="constrained")
        for ax, name, mp in zip(axes, beams, maps):
            full = np.full(mask.shape, np.nan)
            full[mask] = mp
            im = ax.imshow(full, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis",
                           vmin=args.vmin, vmax=vmax)
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.5", lw=0.8))
            ax.plot(*look, "w*", ms=12, mec="k", label="look")
            for k, (_, _, a, b) in enumerate(lobes):
                ax.plot(a, b, "o", ms=11, mfc="none", mec="w", mew=1.5,
                        label="grating lobe" if k == 0 else None)
            for k, (a, b, _) in enumerate(ints):
                ax.plot(a, b, "rx", ms=9, mew=2, label="interferer" if k == 0 else None)
            ax.set_title(f"{name}", fontsize=10)
            ax.set_xlabel("l"); ax.set_ylabel("m"); ax.set_aspect("equal")
        fig.colorbar(im, ax=list(axes[:nb]), fraction=0.02, pad=0.01, label="dB")
        axes[0].legend(loc="lower left", fontsize=7)
        ax = axes[nb]
        target = ((ints[0][0], ints[0][1]) if ints else
                  (lobes[0][2], lobes[0][3]) if lobes else (look[0] + 1.0, look[1]))
        s, lc, mc, u = cut_axis(look, target)
        for name, w in zip(beams, W):
            ax.plot(s, db(beam_response(w, lc, mc, f, x, y)), lw=1.0, label=name)
        for a, b, _ in ints:
            ax.axvline((a - look[0]) * u[0] + (b - look[1]) * u[1], color="r", ls=":", lw=1)
        for _, _, a, b in lobes:
            on_line = abs(-(a - look[0]) * u[1] + (b - look[1]) * u[0]) < 0.02
            if on_line:
                ax.axvline((a - look[0]) * u[0] + (b - look[1]) * u[1], color="0.4",
                           ls="--", lw=1)
        ax.set_ylim(args.vmin - 10, vmax + 3); ax.set_xlabel("distance from look along cut")
        ax.set_ylabel("dB"); ax.set_title("cut: look to interferer / lobe", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        fig.suptitle(f"ch {ch} ({f / 1e6:.1f} MHz): beam response |w$^H$a(l, m)|$^2$", fontsize=11)
        fig.savefig(args.out / f"beam_response_ch{ch:03d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    np.savez_compressed(args.out / "beam_response.npz", **save)
    print(f"\nfigures and maps -> {args.out}")


if __name__ == "__main__":
    main()
