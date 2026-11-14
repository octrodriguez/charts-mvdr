#!/usr/bin/env python
"""Fit a beamformed burst with fitburst (CHIME/FRB) and compare with the simulated truth.

export   writes the binned power of one beam of a 09_beamform_waterfall.py output
         (run with --save-waterfall) in the fitburst 'generic' format, plus a
         sidecar <out>.json with the initial guesses and the values fitburst
         should return for the simulated burst
fit      fits an exported file with the fitburst API (unmodified: DataReader,
         SpectrumModeler, LSFitter), as fitburst_pipeline.py does without --window,
         but records the solver status instead of crashing when a fit does not
         converge, and writes results_fitburst.json in the same layout
compare  reads a results_fitburst*.json (of fit or of fitburst_pipeline.py) and the
         sidecar and prints fit, uncertainty, expected value and pull per parameter

Data written (fitburst generic .npz):
  data_full         (n_freq, n_time) = P - mean off-pulse power, per channel, on a
                    contiguous channel grid; flagged, missing or uncovered channels
                    are rows of zeros (fitburst_pipeline.py keeps a channel only if
                    its row does not sum to zero; the metadata bad_chans alone are
                    ignored by the pipeline)
  metadata          num_freq, num_time, res_time, res_freq, freqs_bin0 (lower edge
                    of the first channel: channel centres are f_k), times_bin0
                    (window time of the first bin, s), bad_chans, is_dedispersed
  burst_parameters  initial guesses (lists, one component)

Expected values (sidecar "expected"), for the simulator burst model:
  dm             DM_sim * K_sim / K_fitburst: the simulator delays use K_sim =
                 4148.741601 MHz^2 s and fitburst 4149.3776 MHz^2 s, so fitburst
                 returns a DM lower by 1.5e-4 (relative)
  burst_width    sigma of the intensity profile: the simulator envelope is a
                 Gaussian in amplitude with sigma pulse_width_ms, so the power is a
                 Gaussian with sigma pulse_width_ms / sqrt(2)
  arrival_time   top-channel arrival, relative to the first exported bin, at
                 ref_freq = the top frequency of the simulated band
  amplitude      log10(amp^2): peak power of the burst per sample after distortionless
                 weights (w^H d = 1, DFT and MVDR alike); the 4-bit Bussgang gain
                 (g^2 >= 0.99 at the simulator noise levels) and the bin averaging
                 (< 1e-3 for bins much shorter than the width) are neglected
  spectral_index, spectral_running   0 (flat amplitude across the band)
Scattering is not simulated: scattering_timescale is 0 (Gaussian profile) and stays
fixed, as fitburst_pipeline.py does by default.

Initial guesses: DM offset by --dm-perturb (default 1%, the accuracy of a search
pipeline), width and arrival by --perturb. With a 10% DM offset at DM 30 the model
starts ~60 ms (40 widths) away from the burst in the lowest channels and the solver
can stop without converging (seen on high-S/N beams).

fitburst weights each channel by the rms of its data over --weight_range (time-bin
indices; default: the whole span, burst included, which down-weights the channels
where the burst is bright): export prints the off-pulse range before the top-channel
arrival (minus 5 widths) to pass as --weight_range.

The uncertainties come from channel weights measured off-pulse, so they ignore the
self-noise of the burst (the simulator burst is a Gaussian random waveform: its power
per bin fluctuates by 1/sqrt(samples per bin)); for bright bursts the pulls are
inflated (reduced chi^2 > 1 is the sign) and the uncertainties are lower bounds.

Run fitburst_pipeline.py without --window: at fitburst 11a049f the windowed data are
dedispersed with the initial DM while the model still applies the full dispersion
delay, and the fit fails even from the true parameters. Limit the cost with
--margin-s here instead (bins kept: top-channel arrival - margin to bottom-channel
arrival + margin).

Example:
  python tools/waterfall_to_fitburst.py export --waterfall OUT/waterfall.npz --beam dft \\
      --margin-s 0.02 --out fit_dft/burst.npz
  python tools/waterfall_to_fitburst.py fit --data fit_dft/burst.npz
  python tools/waterfall_to_fitburst.py compare --truth fit_dft/burst.json \\
      --results fit_dft/results_fitburst.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from charts_mvdr.transients import K_DM

K_FITBURST_DEFAULT = 4149.377593360996   # fitburst/backend/general.yaml (commit 11a049f)
PARAMETERS = ("amplitude", "arrival_time", "burst_width", "dm", "spectral_index",
              "spectral_running")


def fitburst_dispersion_constant() -> float:
    """Dispersion constant of the installed fitburst (MHz^2 s), else the 11a049f value."""
    try:
        from fitburst.backend import general
        return float(general["constants"]["dispersion"])
    except Exception:  # fitburst not installed here
        return K_FITBURST_DEFAULT


def longest_run(mask) -> tuple:
    """(start, stop) of the longest run of True in a 1-D mask; (0, 0) if none."""
    best, start = (0, 0), None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def channel_grid(freqs_mhz, df_mhz=None) -> tuple:
    """(row of each channel on a contiguous grid, number of rows, df) for channel centres."""
    f = np.asarray(freqs_mhz, dtype=float)
    if df_mhz is None:
        df_mhz = float(np.min(np.diff(np.sort(f)))) if f.size > 1 else 0.3
    pos = (f - f.min()) / df_mhz
    rows = np.rint(pos).astype(int)
    if np.max(np.abs(pos - rows)) > 1e-3:
        raise ValueError(f"channel centres are not on a grid of {df_mhz:g} MHz")
    return rows, int(rows.max()) + 1, df_mhz


def build_export(wf: dict, beam: str, df_mhz=None, perturb: float = 0.1,
                 margin_s: float | None = None, k_fitburst: float | None = None,
                 dm_perturb: float = 0.01) -> tuple:
    """(npz dict in fitburst generic format, sidecar dict) for one beam of a waterfall file."""
    key = f"{beam}_power"
    if key not in wf:
        raise ValueError(f"no {key!r} in the waterfall file (run 09 with --save-waterfall; "
                         f"beams: {list(np.asarray(wf['beams']))})")
    P = np.asarray(wf[key], dtype=float)                # (n_bin, n_ch)
    mu = np.asarray(wf[f"{beam}_mu_off"], dtype=float)
    t_bin = np.asarray(wf["t_bin"], dtype=float)
    bin_s = float(wf["bin_s"]) if "bin_s" in wf else float(np.median(np.diff(t_bin)))
    fmhz = np.asarray(wf["freq_mhz"], dtype=float)
    flag = np.asarray(wf["flag"], dtype=bool)
    dm, w_s, t_top = float(wf["dm"]), float(wf["width_s"]), float(wf["t_top"])
    f_top = float(wf["f_top"]) if "f_top" in wf else float(fmhz.max())
    amp = float(wf["amp"]) if "amp" in wf else float("nan")
    k_fb = fitburst_dispersion_constant() if k_fitburst is None else float(k_fitburst)

    good = ~flag & np.isfinite(mu)
    if not good.any():
        raise ValueError("no usable channel")
    covered = np.all(np.isfinite(P[:, good]), axis=1)
    if margin_s is not None:
        sweep = K_DM * dm * (fmhz[good].min() ** -2.0 - f_top ** -2.0)
        covered &= (t_bin >= t_top - margin_s) & (t_bin + bin_s <= t_top + sweep + margin_s)
    i0, i1 = longest_run(covered)
    if i1 - i0 < 3:
        raise ValueError("no contiguous span of bins covered in every usable channel")
    D = (P[i0:i1] - mu).T                                # (n_ch, n_time)
    rows, n_rows, df = channel_grid(fmhz, df_mhz)
    data = np.zeros((n_rows, i1 - i0))
    data[rows[good]] = D[good]
    bad = np.ones(n_rows, dtype=bool)
    bad[rows[good]] = False
    t0 = float(t_bin[i0])
    span = (i1 - i0) * bin_s

    # Truth for the fitburst model, and initial guesses
    width_i = w_s / math.sqrt(2.0)
    arrival = t_top - t0
    expected = {"dm": dm * K_DM / k_fb, "burst_width": width_i, "arrival_time": arrival,
                "amplitude": math.log10(amp ** 2) if amp > 0 else float("nan"),
                "spectral_index": 0.0, "spectral_running": 0.0}
    t_arr = t_top + K_DM * dm * (fmhz ** -2.0 - f_top ** -2.0) - t0
    idx = np.clip(np.rint(t_arr / bin_s).astype(int), 0, i1 - i0 - 1)
    peak = np.median(D[good, idx[good]])
    initial = {"amplitude": [math.log10(peak) if peak > 0 else 0.0],
               "arrival_time": [arrival + perturb * width_i],
               "burst_width": [width_i * (1.0 + perturb)],
               "dm": [dm * (1.0 + dm_perturb)], "dm_index": [-2.0],
               "ref_freq": [f_top], "scattering_index": [-4.0], "scattering_timescale": [0.0],
               "spectral_index": [0.0], "spectral_running": [0.0]}
    metadata = {"num_freq": n_rows, "num_time": int(i1 - i0), "res_time": bin_s,
                "res_freq": df, "freqs_bin0": float(fmhz.min() - df / 2.0), "times_bin0": t0,
                "bad_chans": list(np.flatnonzero(bad).astype(int)), "is_dedispersed": False}

    n_pre = int(np.floor((arrival - 5.0 * width_i) / bin_s))
    weight_range = [0, n_pre] if n_pre >= 10 else None
    sidecar = {"beam": beam, "expected": expected, "weight_range": weight_range,
               "initial": {k: v[0] for k, v in initial.items()},
               "span_s": [t0, t0 + span], "n_good_channels": int(good.sum()),
               "n_rows": n_rows, "bin_s": bin_s, "k_sim": K_DM, "k_fitburst": k_fb}
    npz = {"data_full": data, "metadata": metadata, "burst_parameters": initial}
    return npz, sidecar


FIXED_ALWAYS = ("dm_index", "scattering_index", "scattering_timescale")


def run_fit(data_path, weight_range=None, iterations: int = 2, fix=()) -> dict:
    """Fit a fitburst generic .npz with the fitburst API; returns the results dict.

    Same steps as fitburst_pipeline.py (11a049f) without --window, --preprocess or
    resampling: channels kept where the data row does not sum to zero, weights from
    the per-channel rms over weight_range, dm_index and the scattering parameters
    fixed (plus fix), least squares with the exact Jacobian, repeated iterations
    starting from the previous best fit. A fit that does not converge is reported
    (success False, solver status and message) instead of raising.
    """
    from fitburst.analysis.fitter import LSFitter
    from fitburst.analysis.model import SpectrumModeler
    from fitburst.backend.generic import DataReader

    data = DataReader(str(data_path))
    data.load_data()
    good = np.sum(data.data_full, axis=1) != 0.0
    params = {k: list(v) for k, v in data.burst_parameters.items()}
    model = SpectrumModeler(data.freqs, data.times, num_components=len(params["dm"]),
                            is_dedispersed=bool(data.is_dedispersed))
    model.update_parameters(params)
    history, fitter = [], None
    for it in range(iterations):
        fitter = LSFitter(data.data_full, model, good, weighted_fit=True,
                          weight_range=list(weight_range) if weight_range else None)
        fitter.fix_parameter(list(FIXED_ALWAYS) + list(fix))
        fitter.fit(exact_jacobian=True)
        res = getattr(fitter, "results", None)
        step = {"iteration": it, "success": bool(res is not None and res.success),
                "status": None if res is None else int(res.status),
                "nfev": None if res is None else int(res.nfev),
                "message": "solver raised" if res is None else str(res.message)}
        history.append(step)
        if not step["success"]:
            break
        model.update_parameters(fitter.fit_statistics["bestfit_parameters"])
    ok = bool(history) and history[-1]["success"]
    out = {"success": ok, "solver": history, "fixed": list(FIXED_ALWAYS) + list(fix),
           "weight_range": list(weight_range) if weight_range else None,
           "num_good_freq": int(good.sum()),
           "initial_parameters": params, "model_parameters": model.get_parameters_dict()}
    if ok:
        out["fit_statistics"] = fitter.fit_statistics
    return out


def to_json(obj):
    """json.dump default for numpy scalars and arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def compare(sidecar: dict, results: dict) -> list:
    """Rows (parameter, fit, uncertainty, expected, pull) from fitburst results and a sidecar."""
    stats = results.get("fit_statistics", {})
    best = stats.get("bestfit_parameters", results.get("model_parameters", {}))
    unc = stats.get("bestfit_uncertainties", {})
    rows = []
    for p in PARAMETERS:
        if p not in best:
            continue
        fit = float(np.atleast_1d(best[p])[0])
        err = float(np.atleast_1d(unc[p])[0]) if p in unc else float("nan")
        exp = float(sidecar["expected"].get(p, float("nan")))
        pull = (fit - exp) / err if err > 0 else float("nan")
        rows.append((p, fit, err, exp, pull))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="waterfall.npz -> fitburst generic .npz + sidecar")
    ex.add_argument("--waterfall", type=Path, required=True, help="waterfall.npz of 09")
    ex.add_argument("--beam", required=True, help="dft, kl_local, kl_global or kl_range")
    ex.add_argument("--df-mhz", type=float, default=None,
                    help="channel width (default: smallest spacing of the channels)")
    ex.add_argument("--perturb", type=float, default=0.1,
                    help="relative offset of the initial width from the truth (arrival "
                         "offset by perturb x width)")
    ex.add_argument("--dm-perturb", type=float, default=0.01,
                    help="relative offset of the initial DM from the truth")
    ex.add_argument("--margin-s", type=float, default=None,
                    help="keep only the bins from the top-channel arrival - margin to the "
                         "bottom-channel arrival + margin (default: all covered bins)")
    ex.add_argument("--out", type=Path, required=True, help="output .npz (sidecar: .json)")
    fi = sub.add_parser("fit", help="fit an exported file with the fitburst API")
    fi.add_argument("--data", type=Path, required=True, help=".npz written by export")
    fi.add_argument("--iterations", type=int, default=2)
    fi.add_argument("--weight-range", type=int, nargs=2, default=None, metavar=("I0", "I1"),
                    help="time-bin indices for the channel weights (default: the sidecar "
                         "weight_range, else the whole span)")
    fi.add_argument("--fix", nargs="+", default=[],
                    help="further parameters to keep at their initial values")
    fi.add_argument("--out", type=Path, default=None,
                    help="results file (default: results_fitburst.json next to --data)")
    co = sub.add_parser("compare", help="fitburst results vs expected values")
    co.add_argument("--truth", type=Path, required=True, help="sidecar .json of export")
    co.add_argument("--results", type=Path, required=True, help="results_fitburst*.json")
    args = ap.parse_args()

    if args.cmd == "export":
        wf = dict(np.load(args.waterfall, allow_pickle=False))
        try:
            npz, side = build_export(wf, args.beam, args.df_mhz, args.perturb, args.margin_s,
                                     dm_perturb=args.dm_perturb)
        except ValueError as err:
            raise SystemExit(str(err))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **npz)
        args.out.with_suffix(".json").write_text(json.dumps(side, indent=2))
        m = npz["metadata"]
        print(f"{args.out}: {m['num_freq']} channels ({side['n_good_channels']} usable), "
              f"{m['num_time']} bins of {m['res_time'] * 1e6:.1f} us "
              f"({side['span_s'][0]:.4f}-{side['span_s'][1]:.4f} s)")
        print("expected: " + ", ".join(f"{k} {v:.6g}" for k, v in side["expected"].items()))
        wr = side["weight_range"]
        print("fitburst option: " + (f"--weight_range {wr[0]} {wr[1]}" if wr else
                                     "fewer than 10 bins before the burst: no --weight_range"))
    elif args.cmd == "fit":
        wr = args.weight_range
        side_path = args.data.with_suffix(".json")
        if wr is None and side_path.exists():
            wr = json.loads(side_path.read_text()).get("weight_range")
        res = run_fit(args.data, wr, args.iterations, args.fix)
        out = args.out or args.data.parent / "results_fitburst.json"
        out.write_text(json.dumps(res, indent=1, default=to_json))
        for st in res["solver"]:
            print(f"iteration {st['iteration']}: success {st['success']}, status {st['status']}, "
                  f"{st['nfev']} evaluations ({st['message']})")
        print(f"{'fit converged' if res['success'] else 'FIT DID NOT CONVERGE'} -> {out}")
        if not res["success"]:
            raise SystemExit(1)
    else:
        side = json.loads(args.truth.read_text())
        res = json.loads(args.results.read_text())
        if res.get("success") is False:
            raise SystemExit(f"{args.results}: the fit did not converge "
                             f"({res['solver'][-1]['message']})")
        print(f"beam {side['beam']}")
        print(f"{'parameter':<18} {'fit':>14} {'+/-':>11} {'expected':>14} {'pull':>7}")
        for p, fit, err, exp, pull in compare(side, res):
            print(f"{p:<18} {fit:14.7g} {err:11.3g} {exp:14.7g} {pull:7.2f}")
        chi = res.get("fit_statistics", {}).get("chisq_final_reduced")
        if chi is not None:
            print(f"reduced chi^2 {float(chi):.3f}")


if __name__ == "__main__":
    main()
