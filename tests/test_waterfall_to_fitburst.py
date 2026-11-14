"""tools/waterfall_to_fitburst.py: export to the fitburst generic format and comparison."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr.transients import K_DM

TOOL = Path(__file__).resolve().parents[1] / "tools" / "waterfall_to_fitburst.py"
spec = importlib.util.spec_from_file_location("waterfall_to_fitburst", TOOL)
wtf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wtf)

DM, WIDTH, T_TOP, AMP = 20.0, 2e-3, 1.0, 0.2
BIN = 64 * 10e-6 / 3
NOISE = 0.125


def waterfall(path, seed=1, gap=False):
    """Binned beam power of a dispersed Gaussian burst (power sigma = WIDTH / sqrt(2))
    plus chi-square noise of 64 samples per bin; every 4th channel of 300-400.5 MHz."""
    rng = np.random.default_rng(seed)
    f = 300.0 + 0.3 * np.arange(0, 336, 4)
    f_top = 300.0 + 0.3 * 335
    t = np.arange(int(0.9 / BIN), int(1.5 / BIN)) * BIN
    ta = T_TOP + K_DM * DM * (f ** -2.0 - f_top ** -2.0)
    P = NOISE * rng.gamma(64, 1 / 64, (t.size, f.size))
    sub = t[:, None] + np.arange(64) * BIN / 64        # sample times in each bin
    P += AMP ** 2 * np.exp(-(((sub[:, :, None] - ta) / WIDTH) ** 2)).mean(1)
    if gap:
        P[100:110] = np.nan
    flag = np.zeros(f.size, bool)
    flag[[5, 40]] = True
    np.savez(path, freq_mhz=f, flag=flag, dm=DM, width_s=WIDTH, t_top=T_TOP, t_bin=t,
             bin_s=BIN, amp=AMP, f_top=f_top, beams=np.array(["dft"]),
             dft_power=P.astype(np.float32), dft_mu_off=np.full(f.size, NOISE))
    return path


def test_export_format_and_expected_values(tmp_path):
    wf = dict(np.load(waterfall(tmp_path / "w.npz")))
    npz, side = wtf.build_export(wf, "dft", k_fitburst=4149.377593360996)
    m = npz["metadata"]
    assert npz["data_full"].shape == (m["num_freq"], m["num_time"]) == (84, wf["t_bin"].size)
    # contiguous 1.2 MHz grid, lower edge of channel 0, flagged rows all zero
    assert m["res_freq"] == pytest.approx(1.2) and m["freqs_bin0"] == pytest.approx(299.4)
    assert m["bad_chans"] == [5, 40] and m["is_dedispersed"] is False
    assert not npz["data_full"][[5, 40]].any() and npz["data_full"][0].any()
    e = side["expected"]
    assert e["dm"] == pytest.approx(DM * K_DM / 4149.377593360996)
    assert e["burst_width"] == pytest.approx(WIDTH / np.sqrt(2))
    assert e["arrival_time"] == pytest.approx(T_TOP - wf["t_bin"][0] - BIN / 2)
    assert m["times_bin0"] == pytest.approx(wf["t_bin"][0] + BIN / 2)
    assert e["amplitude"] == pytest.approx(np.log10(AMP ** 2))
    bp = npz["burst_parameters"]
    assert bp["scattering_timescale"] == [0.0] and bp["ref_freq"] == [pytest.approx(400.5)]
    assert bp["dm"][0] == pytest.approx(1.01 * DM)
    assert bp["burst_width"][0] == pytest.approx(1.1 * WIDTH / np.sqrt(2))
    assert abs(bp["amplitude"][0] - e["amplitude"]) < 0.3      # guess from the data
    n_pre = int(np.floor((e["arrival_time"] - 5 * e["burst_width"]) / BIN))
    assert side["weight_range"] == [0, n_pre]


def test_export_span_gap_margin_and_grid(tmp_path):
    wf = dict(np.load(waterfall(tmp_path / "w.npz", gap=True)))
    npz, side = wtf.build_export(wf, "dft")
    # longest covered run: after the gap
    assert side["span_s"][0] == pytest.approx(wf["t_bin"][110])
    npz, side = wtf.build_export(wf, "dft", margin_s=0.01)
    sweep = K_DM * DM * (300.0 ** -2 - 400.5 ** -2)
    assert side["span_s"][0] >= T_TOP - 0.01 - 1e-9
    assert side["span_s"][1] <= T_TOP + sweep + 0.01 + 1e-9
    with pytest.raises(ValueError):
        wtf.build_export(wf, "kl_local")
    with pytest.raises(ValueError):
        wtf.channel_grid([300.0, 300.3, 300.75])


def test_compare_rows():
    side = {"expected": {"dm": 10.0, "burst_width": 1e-3}}
    res = {"fit_statistics": {"bestfit_parameters": {"dm": [10.2], "burst_width": [1.1e-3]},
                              "bestfit_uncertainties": {"dm": [0.1], "burst_width": [5e-5]}}}
    rows = {r[0]: r for r in wtf.compare(side, res)}
    assert rows["dm"][4] == pytest.approx(2.0) and rows["burst_width"][4] == pytest.approx(2.0)


def test_fitburst_recovers_the_burst(tmp_path):
    pytest.importorskip("fitburst")
    waterfall(tmp_path / "w.npz")
    out = tmp_path / "fit" / "burst.npz"
    r = subprocess.run([sys.executable, str(TOOL), "export", "--waterfall", str(tmp_path / "w.npz"),
                        "--beam", "dft", "--margin-s", "0.02", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(TOOL), "fit", "--data", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "fit converged" in r.stdout, r.stdout + r.stderr
    side = json.loads(out.with_suffix(".json").read_text())
    res = json.loads((out.parent / "results_fitburst.json").read_text())
    assert res["success"] and res["weight_range"] == side["weight_range"]
    assert [st["success"] for st in res["solver"]] == [True, True]
    rows = {p: (fit, err, pull) for p, fit, err, _, pull in wtf.compare(side, res)}
    for p in ("dm", "arrival_time", "burst_width", "amplitude"):
        assert abs(rows[p][2]) < 4, (p, rows[p])
    r = subprocess.run([sys.executable, str(TOOL), "compare", "--truth", str(out.with_suffix(".json")),
                        "--results", str(out.parent / "results_fitburst.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "pull" in r.stdout


def test_fit_failure_is_reported_not_raised(tmp_path, monkeypatch):
    pytest.importorskip("fitburst")
    from fitburst.analysis.fitter import LSFitter
    wf = dict(np.load(waterfall(tmp_path / "w.npz")))
    npz, side = wtf.build_export(wf, "dft", margin_s=0.01)
    np.savez(tmp_path / "b.npz", **npz)
    monkeypatch.setattr(LSFitter, "fit", lambda self, exact_jacobian=True: None)
    res = wtf.run_fit(tmp_path / "b.npz", side["weight_range"])
    assert res["success"] is False and "fit_statistics" not in res
    assert res["solver"] == [{"iteration": 0, "success": False, "status": None, "nfev": None,
                              "message": "solver raised"}]
    (tmp_path / "r.json").write_text(json.dumps(res, default=wtf.to_json))
    (tmp_path / "b.json").write_text(json.dumps(side))
    r = subprocess.run([sys.executable, str(TOOL), "compare", "--truth", str(tmp_path / "b.json"),
                        "--results", str(tmp_path / "r.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "did not converge" in r.stderr
