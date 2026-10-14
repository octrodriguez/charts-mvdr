#!/usr/bin/env python
"""Quantization and saturation analysis of a baseband window.

Per channel (averaged over the selected frames and antennas):
  std_out   standard deviation per real component of the 4-bit samples [LSB]
  f_rail    fraction of components at the rails (|x| >= 7)
  sigma_in  input std per component: from f_rail assuming Gaussian input,
            f_rail = 2 Q(6.5 / sigma_in) (rounding to +/-7 starts at 6.5), where
            f_rail >= 1e-4; otherwise sqrt(std_out^2 - 1/12)
  g2_pred   Bussgang prediction of the gain on a weak signal's cross-power:
            g = erf(L / (sqrt(2) sigma_in)) with saturation level L = 7.5
  g2_meas   measured gain: power of the look-direction source fitted to the
            cross-correlations (rank-1 templates for the source and, if present,
            the dominant interferer), relative to the nearest channels with
            f_rail below the clean threshold (removes bandpass and noise bias)

Optionally uses Kotekan correlation dumps (--corr-dir) for R; otherwise R is
computed from the baseband frames with np_correlator.

Example:
  python scripts/06_saturation_analysis.py \\
      --window-dir $SCRATCH/data/sim_test/clean_v1/charts_sim_64ant \\
      --corr-dir $SCRATCH/data/kotekan_corr/charts_sim_64ant_<jobid> \\
      --out $SCRATCH/data/figs/saturation
"""

from __future__ import annotations

import argparse
import re
from math import erf, sqrt
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfcinv

from charts_mvdr.beamforming import beam_power, dft_weights, lm_grid, steering_vectors
from charts_mvdr.data_io import read_astron_correlation, read_baseband_frame
from charts_mvdr.np_correlator import correlate

FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{7})\.bin$")
L_SAT = 7.5          # saturation level of the [-7, 7] quantizer with rounding
RAIL_START = 6.5     # inputs above this round to the rail value 7
WARN_RAIL = 1e-3     # checks.check_saturation WARN threshold


def frames(directory: Path, prefix: str | None = None) -> dict[int, Path]:
    out = {}
    for p in sorted(directory.glob("*.bin")):
        m = FRAME_RE.match(p.name)
        if m and (prefix is None or m["prefix"] == prefix):
            out[int(m["index"])] = p
    return out


def sigma_from_rail(f_rail: np.ndarray) -> np.ndarray:
    """Invert f_rail = 2 Q(6.5 / sigma) = erfc(6.5 / (sqrt(2) sigma))."""
    f = np.clip(f_rail, 1e-12, 1 - 1e-12)
    return RAIL_START / (np.sqrt(2.0) * erfcinv(f))


def bussgang_g2(sigma_in: np.ndarray) -> np.ndarray:
    return np.array([erf(L_SAT / (sqrt(2.0) * s)) ** 2 for s in np.atleast_1d(sigma_in)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-dir", type=Path, required=True,
                    help="baseband frames <window>_%%07d.bin and <window>_meta.h5")
    ap.add_argument("--window", default=None, help="frame prefix (default: directory name)")
    ap.add_argument("--corr-dir", type=Path, default=None,
                    help="Kotekan correlation dumps (default: correlate the baseband frames)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"))
    ap.add_argument("--samples", type=int, default=1536)
    ap.add_argument("--clean-rail", type=float, default=WARN_RAIL,
                    help="channels with f_rail below this are used as reference")
    ap.add_argument("--out", type=Path, default=Path("saturation"))
    args = ap.parse_args()

    meta_path = next(args.window_dir.glob("*_meta.h5"))
    with h5py.File(meta_path, "r") as f:
        x, y = f["antenna_pos_x_m"][:], f["antenna_pos_y_m"][:]
        freq_hz = f["frequencies_mhz"][:] * 1e6
    N, F = x.size, freq_hz.size
    window = args.window or args.window_dir.name
    bb = frames(args.window_dir, window)
    idx = sorted(bb)[: args.max_frames]
    if not idx:
        raise SystemExit("no baseband frames found")

    # Pass over baseband: rail statistics, variance and (optionally) covariance
    n_rail = np.zeros(F)
    s2 = np.zeros(F)
    R = np.zeros((F, N, N), dtype=np.complex128)
    for k in idx:
        V = read_baseband_frame(bb[k], F, N, args.samples)
        comps = np.concatenate([V.real, V.imag], axis=0)          # (2T, F, N)
        n_rail += np.sum(np.abs(comps) >= 7, axis=(0, 2))
        s2 += np.sum(comps.astype(np.float64) ** 2, axis=(0, 2))
        if args.corr_dir is None:
            R += correlate(V.astype(np.complex128))
    n_comp = len(idx) * 2 * args.samples * N
    f_rail = n_rail / n_comp
    std_out = np.sqrt(s2 / n_comp)

    if args.corr_dir is not None:
        cr = frames(args.corr_dir, "corr")
        use = [k for k in idx if k in cr]
        for k in use:
            R += read_astron_correlation(cr[k], F, N)
        print(f"covariance from {len(use)} Kotekan dumps")
        R /= len(use) * args.samples
    else:
        print(f"covariance from {len(idx)} baseband frames (np_correlator)")
        R /= len(idx) * args.samples

    clean = f_rail < args.clean_rail
    if args.look is None:
        gl, gm, _ = lm_grid(161)
        img = sum(beam_power(R[c], dft_weights(steering_vectors(gl, gm, freq_hz[c], x, y)))
                  for c in np.flatnonzero(clean)[::8])
        look = (float(gl[np.argmax(img)]), float(gm[np.argmax(img)]))
    else:
        look = tuple(args.look)
    print(f"{len(idx)} frames, {N} antennas, {F} channels; look direction "
          f"({look[0]:+.3f}, {look[1]:+.3f}); reference channels: {clean.sum()}")

    # Weak-signal power in each channel from the cross-correlations only (noise
    # bias removed): least-squares fit of the off-diagonal entries of R with
    # rank-1 templates d d^H (look direction) and, when another source dominates,
    # u u^H (principal eigenvector, i.e. the interferer). The second template keeps
    # interferer leakage out of the estimate.
    off = ~np.eye(N, dtype=bool)
    cross = np.empty(F)
    for c in range(F):
        d = steering_vectors(*look, freq_hz[c], x, y)[0]
        _, U = np.linalg.eigh(R[c])
        u = U[:, -1] * np.sqrt(N)
        templates = [np.outer(d, d.conj())[off]]
        if np.abs(np.vdot(d, u)) ** 2 / N**2 < 0.5:      # dominant source is not the look
            templates.append(np.outer(u, u.conj())[off])
        A = np.stack(templates, axis=1)
        A_ri = np.concatenate([A.real, A.imag])
        b_ri = np.concatenate([R[c][off].real, R[c][off].imag])
        cross[c] = np.linalg.lstsq(A_ri, b_ri, rcond=None)[0][0]
    ref = np.full(F, np.nan)
    clean_idx = np.flatnonzero(clean)
    for c in range(F):
        near = clean_idx[np.argsort(np.abs(clean_idx - c))]
        near = near[near != c][:4]
        if near.size:
            ref[c] = np.mean(cross[near])
    g2_meas = cross / ref

    # Rail-based estimate where saturation is measurable; elsewhere the output std
    # (minus uniform quantization noise, 1/12 LSB^2) is an unbiased estimate.
    sigma_in = np.where(f_rail >= 1e-4, sigma_from_rail(f_rail),
                        np.sqrt(np.maximum(std_out**2 - 1.0 / 12.0, 1e-6)))
    g2_pred = bussgang_g2(sigma_in)

    flagged = np.flatnonzero(f_rail >= 1e-2)
    print(f"\nrail fraction: median {np.median(f_rail):.2e}; channels >= {WARN_RAIL:g}: "
          f"{np.sum(f_rail >= WARN_RAIL)}; >= 1e-2: {flagged.size}")
    print(f"std per component: median {np.median(std_out):.2f} LSB "
          f"(range {std_out.min():.2f}-{std_out.max():.2f}); "
          f"f_rail = {WARN_RAIL:g} corresponds to sigma_in = "
          f"{float(sigma_from_rail(np.array([WARN_RAIL]))[0]):.2f} LSB")
    print(f"\n{'ch':>4} {'MHz':>7} {'std_out':>8} {'f_rail':>9} {'sigma_in':>9} "
          f"{'g2_pred':>8} {'g2_meas':>8}")
    show = np.union1d(flagged, np.argsort(std_out)[-5:])
    for c in show:
        print(f"{c:4d} {freq_hz[c] / 1e6:7.1f} {std_out[c]:8.2f} {f_rail[c]:9.2e} "
              f"{sigma_in[c]:9.2f} {g2_pred[c]:8.3f} {g2_meas[c]:8.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "saturation_summary.npz", freq_hz=freq_hz, std_out=std_out,
             f_rail=f_rail, sigma_in=sigma_in, g2_pred=g2_pred, g2_meas=g2_meas,
             look=np.array(look), frames=np.array(idx))

    fmhz = freq_hz / 1e6
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(fmhz, std_out, lw=1)
    ax[0].set_ylabel("std per component [LSB]")
    ax[1].semilogy(fmhz, np.maximum(f_rail, 1e-7), lw=1)
    ax[1].axhline(WARN_RAIL, color="C1", ls="--", lw=1, label="WARN")
    ax[1].axhline(1e-2, color="C3", ls="--", lw=1, label="FAIL")
    ax[1].set_ylabel("fraction at rails"); ax[1].legend(fontsize=8)
    ax[2].plot(fmhz, g2_meas, ".", ms=3, label="measured (cross-correlation fit)")
    ax[2].plot(fmhz, g2_pred, lw=1, label=r"Bussgang $g^2$")
    ax[2].set_ylim(0, 1.3); ax[2].set_ylabel("gain on weak-signal power")
    ax[2].set_xlabel("frequency [MHz]"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "saturation_vs_channel.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    s = np.geomspace(0.5, 20, 200)
    ax.plot(s, bussgang_g2(s), "k-", label=r"$\mathrm{erf}^2(L/\sqrt{2}\sigma)$, $L=7.5$")
    ok = np.isfinite(g2_meas)
    ax.scatter(sigma_in[ok], g2_meas[ok], s=8, c=np.log10(np.maximum(f_rail[ok], 1e-7)),
               cmap="viridis", label="channels")
    ax.set_xscale("log"); ax.set_ylim(0, 1.3)
    ax.set_xlabel(r"$\sigma_\mathrm{in}$ per component [LSB]")
    ax.set_ylabel("gain on weak-signal power"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / "bussgang_gain.png", dpi=130)
    plt.close(fig)
    print(f"\nfigures and summary -> {args.out}")


if __name__ == "__main__":
    main()
