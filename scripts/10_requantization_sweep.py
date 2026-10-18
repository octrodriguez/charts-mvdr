#!/usr/bin/env python
"""CPU-tier requantization sweep: per-channel digital gain before the 4-bit cut.

One channel, one strong emitter at --emitter (l, m) plus white receiver noise
(std --noise-sigma per real component) and a weak Gaussian sky source. Two
beams are evaluated, each with its own sky source in a separate run: the
"ghost" beam (--look, default the visible alias of the k = -3 ghost closest to
zenith, i.e. the worst case of a search beam that receives the ghost) and a
"clear" beam (--clear-look) far from the emitter and its ghosts. Before quantization the channel is multiplied by a gain G
chosen so that the total input std per component is G sigma_in = target, for
each --targets value ("asis" = G = 1); the quantized voltages are divided by G
to return to input units. For each emitter amplitude (--rfi-amp, complex
amplitude in LSB, E|s|^2 = amp^2 as in the generator), source model
(gaussian, tone) and target the script measures:

  g2_clear, g2_ghost
              flux-scale gain: sky power in the DFT beam, relative to the
              unquantized channel without RFI (compression; a calibration
              error, not an S/N loss)
  ghost_db    power of the k = -3 intermodulation ghost relative to the
              fundamental, from a least-squares fit of the harmonics
              k = 1, -3, 5, -7 to the off-diagonal off-source correlations
  snr_<beam>_<weights>_db
              S/N relative to the unquantized channel without RFI (DFT
              weights), for DFT and KL weights; S/N is the sky power over the
              std of the beam power in bins of --tbin samples, the detection
              metric of 09_beamform_waterfall

KL weights are MVDR with F from the off-source samples (signal-free) and
diagonal loading --loading in units of the noise floor of F.

On and off segments share the noise and emitter waveforms (the sky source has
power zero off source), and the run without RFI shares the noise of the run
with RFI, so differences are free of most estimation noise. The remaining 1-
sigma uncertainty of an S/N ratio is about 4.34 / sqrt(n_bins) dB, with n_bins
= --samples / --tbin.

Predictions (Gaussian model, charts_mvdr.ghosts, exact for the rounding
quantizer, at G sigma_in): g2_bussgang = (c_1 / sigma)^2 (independent
clipping), g2_sky_pred = sky_gain2 (clipping correlated by the emitter; the
prediction for g2_clear) and ghost_pred_db. For the tone model they are
printed for reference only.

Outputs in --out: requantization_sweep.csv, requantization_sweep.png.

Example:
  python scripts/10_requantization_sweep.py --out requant_sweep
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import dft_weights, mvdr_weights, steering_vectors
from charts_mvdr.ghosts import (ghost_beam_response, ghost_directions, ghost_ratio,
                                hermite_weights, sky_gain2)
from charts_mvdr.sim_cpu import Scenario, Source, grid_positions, quantize_4bit, simulate_voltages

HARMONICS = (1, -3, 5, -7)
BEAMS = ("ghost", "clear")
FIELDS = ["model", "rfi_amp", "sigma_in", "target", "gain", "g2_clear", "g2_ghost",
          "g2_bussgang", "g2_sky_pred", "ghost_db", "ghost_pred_db",
          "snr_ghost_dft_db", "snr_ghost_kl_db", "snr_clear_dft_db", "snr_clear_kl_db", "n_bins"]


def default_look(emitter, freq_hz, dx, dy):
    """Visible alias of the k = -3 ghost closest to zenith, or None."""
    d = ghost_directions(*emitter, freq_hz, dx, dy, ks=(-3,))
    if not d:
        return None
    k, p, q, lk, mk = min(d, key=lambda x: x[3] ** 2 + x[4] ** 2)
    return (lk, mk)


def simulate_pair(freq_hz, pos, noise_sigma, rfi_power, emitter, model, look, p_sky,
                  n_samples, seed):
    """On (with sky) and off (without) segments, (K, N), sharing all draws."""
    out = []
    for p in (p_sky, 0.0):
        scn = Scenario(freqs_hz=np.array([freq_hz]), positions=pos, noise_sigma=noise_sigma,
                       sources=[Source(*emitter, power=rfi_power, model=model, name="rfi"),
                                Source(*look, power=p, name="sky")])
        out.append(simulate_voltages(scn, n_samples, rng=seed, dtype=np.complex64)[:, 0, :])
    return out


def requantize(x, gain):
    """4-bit quantization of gain * x, returned in input units."""
    return quantize_4bit(gain * x) / gain


def binned_power(y, tbin):
    nb = y.size // tbin
    return np.mean(np.abs(y[: nb * tbin].reshape(nb, tbin)) ** 2, axis=1)


def beam_snr(x_on, x_off, w, tbin):
    """Sky power S, std of the binned off power, S / std, number of bins."""
    y_on, y_off = x_on @ w.conj(), x_off @ w.conj()
    s = np.mean(np.abs(y_on) ** 2) - np.mean(np.abs(y_off) ** 2)
    p_off = binned_power(y_off, tbin)
    sd = float(np.std(p_off, ddof=1))
    return float(s), sd, float(s / sd), p_off.size


def harmonic_fit(x_off, a_emitter, ks=HARMONICS):
    """Least-squares amplitudes of a^k (a^k)^H in the off-diagonal correlations."""
    K, N = x_off.shape
    R = (x_off.T @ x_off.conj()) / K
    off = ~np.eye(N, dtype=bool)
    basis = np.stack([np.outer(a_emitter**k, (a_emitter**k).conj())[off] for k in ks], axis=1)
    return np.linalg.lstsq(basis, R[off], rcond=None)[0].real, R


def g2_round(sigma):
    return float(hermite_weights(sigma, 1, "round")[1] / sigma**2)


def run(args):
    pos = grid_positions(args.nx, args.ny, args.spacing)
    f = args.freq_mhz * 1e6
    emitter = tuple(args.emitter)
    look = tuple(args.look) if args.look else default_look(emitter, f, args.spacing, args.spacing)
    if look is None:
        raise SystemExit("no visible k = -3 ghost alias; give --look")
    looks = {"ghost": look, "clear": tuple(args.clear_look)}
    a_emit = steering_vectors(*emitter, f, pos[:, 0], pos[:, 1])[0]
    a_look = {b: steering_vectors(*lk, f, pos[:, 0], pos[:, 1])[0] for b, lk in looks.items()}
    w_dft = {b: dft_weights(a[None, :])[0] for b, a in a_look.items()}
    p_sky = args.sky_frac * 2.0 * args.noise_sigma**2
    for b, lk in looks.items():
        resp = ghost_beam_response(*emitter, f, [lk], pos[:, 0], pos[:, 1], ks=(1, -3, 5))[:, 0]
        print(f"{b} beam ({lk[0]:.4f}, {lk[1]:.4f}); DFT response to emitter k=1/-3/+5: "
              + " / ".join(f"{10 * np.log10(r):.1f}" for r in resp) + " dB")

    rows = []
    for model in args.models:
        # reference per beam: no RFI, unquantized, DFT weights (same noise and sky draws)
        s_ref, snr_ref = {}, {}
        for b, lk in looks.items():
            c_on, c_off = simulate_pair(f, pos, args.noise_sigma, 0.0, emitter, model, lk,
                                        p_sky, args.samples, args.seed)
            s_ref[b], _, snr_ref[b], _ = beam_snr(c_on, c_off, w_dft[b], args.tbin)
        del c_on, c_off
        for amp in args.rfi_amp:
            sigma_in = float(np.sqrt(args.noise_sigma**2 + 0.5 * amp**2))
            r = amp**2 / (amp**2 + 2.0 * args.noise_sigma**2)
            res = {tg: {} for tg in args.targets}
            for b, lk in looks.items():
                x_on, x_off = simulate_pair(f, pos, args.noise_sigma, amp**2, emitter, model, lk,
                                            p_sky, args.samples, args.seed)
                for target in args.targets:
                    gain = 1.0 if target == "asis" else float(target) / sigma_in
                    q_on, q_off = requantize(x_on, gain), requantize(x_off, gain)
                    hb, F = harmonic_fit(q_off, a_emit)
                    w_kl = mvdr_weights(F, a_look[b][None, :], args.loading, "noise")[0]
                    s_dft, _, snr_dft, nb = beam_snr(q_on, q_off, w_dft[b], args.tbin)
                    _, _, snr_kl, _ = beam_snr(q_on, q_off, w_kl, args.tbin)
                    d = res[target]
                    d["gain"], d["n_bins"] = gain, nb
                    d[f"g2_{b}"] = s_dft / s_ref[b]
                    d[f"snr_{b}_dft_db"] = 10 * np.log10(snr_dft / snr_ref[b])
                    d[f"snr_{b}_kl_db"] = 10 * np.log10(snr_kl / snr_ref[b])
                    if b == "clear":
                        d["ghost_db"] = 10 * np.log10(abs(hb[1] / hb[0]))
                del x_on, x_off
            for target in args.targets:
                d = res[target]
                sg = d["gain"] * sigma_in
                row = dict(model=model, rfi_amp=amp, sigma_in=sigma_in,
                           target=target if target == "asis" else float(target), gain=d["gain"],
                           g2_clear=d["g2_clear"], g2_ghost=d["g2_ghost"],
                           g2_bussgang=float(hermite_weights(sg, 1, "round")[1] / sg**2),
                           g2_sky_pred=sky_gain2(sg, r), ghost_db=d["ghost_db"],
                           ghost_pred_db=10 * np.log10(ghost_ratio(sg, r, -3, "round")),
                           snr_ghost_dft_db=d["snr_ghost_dft_db"],
                           snr_ghost_kl_db=d["snr_ghost_kl_db"],
                           snr_clear_dft_db=d["snr_clear_dft_db"],
                           snr_clear_kl_db=d["snr_clear_kl_db"], n_bins=d["n_bins"])
                rows.append(row)
                print(" ".join(f"{k}={_fmt(v)}" for k, v in row.items()), flush=True)
    return rows, looks


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: _fmt(v) for k, v in row.items()})


def plot(rows, path, looks):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    amps = sorted({r["rfi_amp"] for r in rows})
    colors = {a: c for a, c in zip(amps, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    for model, ls, mk in (("gaussian", "-", "o"), ("tone", "--", "s")):
        for amp in amps:
            sel = [r for r in rows if r["model"] == model and r["rfi_amp"] == amp]
            if not sel:
                continue
            sel.sort(key=lambda r: r["gain"] * r["sigma_in"])
            x = np.array([r["gain"] * r["sigma_in"] for r in sel])
            lab = f"{model}, {amp:g} LSB"
            kw = dict(ls=ls, marker=mk, color=colors[amp])
            axes[0].plot(x, [r["g2_clear"] for r in sel], label=lab, **kw)
            axes[1].plot(x, [r["ghost_db"] for r in sel], label=lab, **kw)
            for ax, b in ((axes[2], "ghost"), (axes[3], "clear")):
                ax.plot(x, [r[f"snr_{b}_kl_db"] for r in sel], label=f"KL, {lab}", **kw)
                ax.plot(x, [r[f"snr_{b}_dft_db"] for r in sel], alpha=0.35, mfc="none",
                        label=f"DFT, {lab}", **kw)
            if model == "gaussian":
                axes[0].plot(x, [r["g2_sky_pred"] for r in sel], ":", color="k", lw=1)
                axes[0].plot(x, [r["g2_bussgang"] for r in sel], "-.", color="0.5", lw=0.8)
                axes[1].plot(x, [r["ghost_pred_db"] for r in sel], ":", color="k", lw=1)
    axes[0].set_ylabel(r"flux-scale gain $g^2$ (clear beam)")
    axes[1].set_ylabel(r"ghost $k=-3$ / fundamental [dB]")
    for ax, b in ((axes[2], "ghost"), (axes[3], "clear")):
        ax.set_ylabel("S/N relative to no RFI, unquantized [dB]")
        ax.set_title(f"{b} beam ({looks[b][0]:.3f}, {looks[b][1]:.3f})", fontsize=9)
    for ax in axes:
        ax.set_xlabel(r"$G\,\sigma_\mathrm{in}$ [LSB]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[3].legend(fontsize=6, ncol=2)
    axes[0].text(0.02, 0.02, "dotted: Gaussian model (correlated clipping)\n"
                 "dash-dot: Bussgang, independent clipping",
                 transform=axes[0].transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_targets(values):
    out = []
    for v in values:
        out.append("asis" if v == "asis" else float(v))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("requant_sweep"))
    p.add_argument("--freq-mhz", type=float, default=390.6)
    p.add_argument("--emitter", type=float, nargs=2, default=[-0.83, -0.482], metavar=("L", "M"))
    p.add_argument("--look", type=float, nargs=2, default=None, metavar=("L", "M"),
                   help="ghost beam direction (default: k = -3 ghost alias closest to zenith)")
    p.add_argument("--clear-look", type=float, nargs=2, default=[0.3, 0.5], metavar=("L", "M"),
                   help="beam far from the emitter and its ghosts")
    p.add_argument("--rfi-amp", type=float, nargs="+", default=[7.0, 12.0],
                   help="emitter complex amplitude [LSB], E|s|^2 = amp^2")
    p.add_argument("--models", nargs="+", choices=["gaussian", "tone"], default=["gaussian", "tone"])
    p.add_argument("--targets", nargs="+", default=["asis", "4.0", "3.5", "3.0", "2.5", "2.0"],
                   help="G sigma_in values [LSB]; 'asis' for G = 1")
    p.add_argument("--noise-sigma", type=float, default=2.0, help="noise std per component [LSB]")
    p.add_argument("--sky-frac", type=float, default=0.1,
                   help="sky power per element relative to the noise power 2 sigma_n^2")
    p.add_argument("--samples", type=int, default=2**18)
    p.add_argument("--tbin", type=int, default=192)
    p.add_argument("--loading", type=float, default=10.0, help="KL loading [noise floor of F]")
    p.add_argument("--nx", type=int, default=8)
    p.add_argument("--ny", type=int, default=8)
    p.add_argument("--spacing", type=float, default=0.6, help="grid spacing [m]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.targets = parse_targets(args.targets)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, looks = run(args)
    write_csv(rows, args.out / "requantization_sweep.csv")
    plot(rows, args.out / "requantization_sweep.png", looks)
    print(f"wrote {args.out / 'requantization_sweep.csv'} and requantization_sweep.png")


if __name__ == "__main__":
    main()
