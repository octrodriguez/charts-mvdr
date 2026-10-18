#!/usr/bin/env python
"""Composite beam-response figure (Fig. 7 analogue) from 07_beam_response outputs.

Panels:
  (a) normal case, DFT map |w^H a(l, m)|^2
  (b) normal case, MVDR map with the largest loading in the npz
  (c) interferer near a grating lobe, MVDR map with the smallest loading
      (superdirective weights when the loading is 0)
  (d) response towards the interferer vs. its distance to the nearest grating
      lobe of the look direction, one point per channel of --sweep with a
      visible lobe and an interferer, for DFT and
      each MVDR loading, with the ideal-covariance model as lines
  (e) white-noise gain WNG = N ||w||^2 (0 dB for DFT, higher is worse) vs. the
      same distance, measured and ideal

Ideal model (as in tools/make_grating_lobe_events.py): MVDR on
R = sigma^2 I + amp^2 a_int a_int^H with loading delta = loading * sigma^2, where
sigma^2 is the noise floor of F saved by 07 (or --noise) and amp the nominal
amplitude of the interferer in the events file (or --amp). With exact steering
the look-direction source does not change the normalized weights (matrix
inversion lemma), so it is left out. Quantization and finite integration are
not modelled; the difference between points and lines shows their effect.

Inputs are beam_response.npz files written by 07 (with positions and noise
floor, i.e. 07 from this version on). A channel is selected as NPZ:CH.

Example:
  python tools/fig_beam_response.py --normal NORMAL/beam_response.npz:147 \\
      --grating GRATING/beam_response.npz:319 --events GRATING_EVENTS.json \\
      --out fig7_composite.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charts_mvdr.beamforming import mvdr_weights, steering_vectors


def parse_sel(s: str):
    path, _, ch = s.rpartition(":")
    if not path or not ch.isdigit():
        raise SystemExit(f"{s!r} is not NPZ:CH")
    return Path(path), int(ch)


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def beam_key(name: str) -> str:
    return name.replace(" ", "_").replace("=", "")


def full_map(z, ch, beam_idx):
    mask = z["grid_mask"]
    out = np.full(mask.shape, np.nan)
    out[mask] = z[f"ch{ch}_maps_db"][beam_idx]
    return out


def mvdr_indices(z):
    """(index, loading) of the MVDR beams in the order of z['beams']."""
    beams = [str(b) for b in z["beams"]]
    return [(i, float(ld)) for i, ld in zip(range(1, len(beams)), z["loading"])]


def ideal(z, ch, amp, noise, loading):
    """Ideal MVDR: (response to the interferer dB, WNG dB)."""
    k = int(np.flatnonzero(z["channels"] == ch)[0])
    f = float(z["freq_mhz"][k]) * 1e6
    x, y = z["pos_x"], z["pos_y"]
    d = steering_vectors(*z["look"], f, x, y)
    a = steering_vectors(*z[f"ch{ch}_int_lm"][0], f, x, y)[0]
    n = x.size
    R = noise * np.eye(n) + amp**2 * np.outer(a, a.conj())
    w = mvdr_weights(R, d, loading=loading * noise, loading_ref="absolute")[0]
    resp = abs(np.vdot(w, a)) ** 2
    return 10 * np.log10(resp + 1e-30), 10 * np.log10(n * np.sum(np.abs(w) ** 2))


def lobe_offset(z, ch):
    lobes = z[f"ch{ch}_lobes_lm"]
    ints = z[f"ch{ch}_int_lm"]
    if lobes.size == 0 or ints.size == 0:
        return np.nan
    return float(np.min(np.hypot(lobes[:, 0] - ints[0, 0], lobes[:, 1] - ints[0, 1])))


def amps_from_events(path):
    out = {}
    if path is None:
        return out
    for ev in json.loads(Path(path).read_text()):
        if str(ev.get("event_type", "")).startswith("rfi") and ev.get("channels"):
            for ch in ev["channels"]:
                out[int(ch)] = float(ev["nominal_amp"])
    return out


def sweep_table(z, amps, amp_default, noise_override):
    """Rows per channel with an interferer: offset, measured and ideal response / WNG."""
    rows = []
    mv = mvdr_indices(z)
    for ch in [int(c) for c in z["channels"]]:
        if f"ch{ch}_int_lm" not in z or z[f"ch{ch}_int_lm"].size == 0:
            continue
        if not np.isfinite(lobe_offset(z, ch)):      # no visible grating lobe: not in the sweep
            continue
        amp = amps.get(ch, amp_default)
        noise = noise_override if noise_override is not None else z.get(f"ch{ch}_noise_floor")
        row = {"ch": ch, "offset": lobe_offset(z, ch),
               "DFT": (float(z[f"ch{ch}_DFT_int_db"][0]), 0.0)}
        for i, ld in mv:
            name = str(z["beams"][i])
            meas = (float(z[f"ch{ch}_{beam_key(name)}_int_db"][0]),
                    float(z[f"ch{ch}_{beam_key(name)}_wng_db"]))
            idl = (ideal(z, ch, amp, float(noise), ld)
                   if amp is not None and noise is not None and "pos_x" in z else (np.nan, np.nan))
            row[name] = meas
            row[name + " ideal"] = idl
        rows.append(row)
    rows.sort(key=lambda r: r["offset"])
    return rows


def draw_map(ax, z, ch, idx, title, vmin, vmax):
    im = ax.imshow(full_map(z, ch, idx), origin="lower", extent=[-1, 1, -1, 1], cmap="viridis",
                   vmin=vmin, vmax=vmax)
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.5", lw=0.8))
    ax.plot(*z["look"], "w*", ms=12, mec="k", label="look")
    for k, (a, b) in enumerate(z[f"ch{ch}_lobes_lm"].reshape(-1, 2)):
        ax.plot(a, b, "o", ms=11, mfc="none", mec="w", mew=1.5, label="grating lobe" if k == 0 else None)
    for k, (a, b) in enumerate(z[f"ch{ch}_int_lm"].reshape(-1, 2)):
        ax.plot(a, b, "rx", ms=9, mew=2, label="interferer" if k == 0 else None)
    k = int(np.flatnonzero(z["channels"] == ch)[0])
    ax.set_title(f"{title}\nch {ch} ({float(z['freq_mhz'][k]):.1f} MHz)", fontsize=9)
    ax.set_xlabel("$l$")
    ax.set_ylabel("$m$")
    ax.set_aspect("equal")
    return im


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normal", required=True, help="NPZ:CH of the normal case")
    ap.add_argument("--grating", required=True, help="NPZ:CH of the grating-lobe case")
    ap.add_argument("--sweep", default=None, help="NPZ for panels d, e (default: the --grating npz)")
    ap.add_argument("--events", type=Path, default=None, help="events JSON (interferer amplitudes)")
    ap.add_argument("--amp", type=float, default=None, help="interferer amplitude if not in --events")
    ap.add_argument("--noise", type=float, default=None, help="noise floor (default: from the npz)")
    ap.add_argument("--vmin", type=float, default=-60.0)
    ap.add_argument("--out", type=Path, default=Path("fig_beam_response.png"))
    args = ap.parse_args(argv)

    zn_path, ch_n = parse_sel(args.normal)
    zg_path, ch_g = parse_sel(args.grating)
    zn, zg = load(zn_path), load(zg_path)
    zs = load(Path(args.sweep)) if args.sweep else zg
    for z, ch, p in ((zn, ch_n, zn_path), (zg, ch_g, zg_path)):
        if f"ch{ch}_maps_db" not in z:
            raise SystemExit(f"channel {ch} not in {p}")

    mv_n, mv_g = mvdr_indices(zn), mvdr_indices(zg)
    i_n, ld_n = max(mv_n, key=lambda t: t[1])
    i_g, ld_g = min(mv_g, key=lambda t: t[1])
    maps = [(zn, ch_n, 0, "(a) DFT"), (zn, ch_n, i_n, rf"(b) MVDR, $\delta={ld_n:g}\,\hat\sigma_n^2$"),
            (zg, ch_g, i_g, rf"(c) MVDR, $\delta={ld_g:g}\,\hat\sigma_n^2$")]
    vmax = max(0.0, float(np.ceil(max(np.nanmax(full_map(z, c, i)) for z, c, i, _ in maps))))

    fig = plt.figure(figsize=(14, 9), layout="constrained")
    gs = fig.add_gridspec(2, 6)
    axm = [fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(3)]
    for ax, (z, c, i, title) in zip(axm, maps):
        im = draw_map(ax, z, c, i, title, args.vmin, vmax)
    fig.colorbar(im, ax=axm, fraction=0.02, pad=0.01, label="dB relative to look")
    axm[0].legend(loc="lower left", fontsize=7)

    rows = sweep_table(zs, amps_from_events(args.events), args.amp, args.noise)
    ax_d, ax_e = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])
    names = [str(b) for b in zs["beams"]]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(f"{'ch':>4} {'offset':>7}  " + "  ".join(f"{n:>22}" for n in names)
          + "   (response dB / WNG dB; ideal in brackets)")
    for r in rows:
        cells = []
        for n in names:
            m = r[n]
            idl = r.get(n + " ideal")
            cells.append(f"{m[0]:6.1f}/{m[1]:5.2f}" + (f" [{idl[0]:5.1f}/{idl[1]:5.2f}]"
                                                      if idl is not None else " " * 15))
        print(f"{r['ch']:>4} {r['offset']:7.3f}  " + "  ".join(f"{c:>22}" for c in cells))
    if rows:
        off = np.array([r["offset"] for r in rows])
        for k, n in enumerate(names):
            c = colors[k % len(colors)]
            ax_d.plot(off, [r[n][0] for r in rows], "o", color=c, label=f"{n} (measured)")
            ax_e.plot(off, [r[n][1] for r in rows], "o", color=c, label=f"{n} (measured)")
            if n + " ideal" in rows[0]:
                ax_d.plot(off, [r[n + " ideal"][0] for r in rows], "--", color=c, lw=1,
                          label=f"{n} (ideal)")
                ax_e.plot(off, [r[n + " ideal"][1] for r in rows], "--", color=c, lw=1,
                          label=f"{n} (ideal)")
    ax_d.set_ylabel("response towards interferer [dB]")
    ax_d.set_title("(d) null depth vs. distance to the grating lobe", fontsize=9)
    ax_e.set_ylabel(r"WNG $= N\|w\|^2$ [dB] (0 dB: DFT)")
    ax_e.set_title("(e) white-noise gain", fontsize=9)
    for ax in (ax_d, ax_e):
        ax.set_xlabel("interferer distance to the nearest grating lobe in $(l, m)$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
