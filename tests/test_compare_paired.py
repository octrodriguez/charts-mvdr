"""Tests of tools/compare_paired.py on synthetic 09 and 08 results."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "compare_paired.py"
CH = np.arange(10)


def write09(d, snr, cv=0.072, dd=None, flag=None):
    d.mkdir()
    save = {"channels": CH, "flag": np.zeros(CH.size, bool) if flag is None else flag,
            "beams": np.array(list(snr))}
    for b, s in snr.items():
        save[f"{b}_snr_ch"] = np.asarray(s, float)
        save[f"{b}_mu_off"] = np.ones(CH.size)
        save[f"{b}_sd_off"] = np.full(CH.size, cv)
        save[f"{b}_dd_snr"] = float(dd[b]) if dd else 100.0
    np.savez(d / "waterfall.npz", **save)
    return d


def events(tmp):
    ev = [{"event_id": "frb_01", "event_type": "frb"},
          {"event_id": "r1", "event_type": "rfi_narrow", "channels": [2, 3]},
          {"event_id": "r2", "event_type": "rfi_narrow", "channels": [3, 6]},
          {"event_id": "r3", "event_type": "rfi_narrow", "channels": [8]}]
    p = tmp / "ev.json"
    p.write_text(json.dumps(ev))
    return p


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def test_groups_and_relative_snr(tmp_path):
    sys.path.insert(0, str(TOOL.parent))
    import compare_paired as cp
    clean = np.full(CH.size, 10.0)
    dft = clean.copy()
    dft[[2, 3, 6, 8]] = [5.0, 5.0, 2.5, -1.0]
    kl = clean.copy()
    kl[[2, 3, 6, 8]] = [10.0, 8.0, 7.9, 9.0]
    r = cp.load_result(write09(tmp_path / "r", {"dft": dft, "kl_local": kl},
                               dd={"dft": 80.0, "kl_local": 98.0}))
    c = cp.load_result(write09(tmp_path / "c", {"dft": clean, "kl_local": clean}))
    groups = cp.channel_groups(json.loads(events(tmp_path).read_text()),
                               cp.parse_groups(["A=r1,r2"]))
    assert groups == {2: "A", 3: "A", 6: "A", 8: "r3"}
    res = cp.compare(r, c, groups, 1e-3)
    assert res["paired"] and res["n_clean"] == 6
    assert res["rel"]["dft"][2] == pytest.approx(-3.0103, abs=1e-3)
    assert res["rel"]["kl_local"][2] == pytest.approx(0.0)
    assert np.isnan(res["rel"]["dft"][8])                  # non-positive S/N: lost


def test_cli_and_pairing_failure(tmp_path):
    clean = np.full(CH.size, 10.0)
    dft = clean.copy()
    dft[2] = 5.0
    write09(tmp_path / "r", {"dft": dft, "kl_local": clean})
    write09(tmp_path / "c", {"dft": clean, "kl_local": clean})
    ev = events(tmp_path)
    res = run("--rfi", tmp_path / "r", "--clean", tmp_path / "c", "--events", ev,
              "--csv", tmp_path / "p.csv")
    assert res.returncode == 0, res.stderr
    assert "(paired)" in res.stdout and "r1" in res.stdout and "clean" in res.stdout
    assert "dedispersed S/N: clean dft 100.0" in res.stdout
    assert (tmp_path / "p.csv").read_text().splitlines()[0].startswith("channel,group,flag")
    bad = clean.copy()
    bad[0] = 9.0                                           # a clean channel differs
    write09(tmp_path / "r2", {"dft": bad, "kl_local": clean})
    res = run("--rfi", tmp_path / "r2", "--clean", tmp_path / "c", "--events", ev)
    assert res.returncode == 2 and "NOT PAIRED" in res.stdout


def test_08_results(tmp_path):
    sys.path.insert(0, str(TOOL.parent))
    import compare_paired as cp
    for name, s in (("r", 0.5), ("c", 1.0)):
        d = tmp_path / name
        d.mkdir()
        S = np.ones((CH.size, 2))
        S[2] = [s, 1.0]
        np.savez(d / "transient_snr.npz", channels=CH, weights=np.array(["dft", "off_local"]),
                 S=S, Q=np.ones((CH.size, 2)))
    res = cp.compare(cp.load_result(tmp_path / "r"), cp.load_result(tmp_path / "c"),
                     {2: "r1"}, 1e-3)
    assert res["paired"]
    assert res["rel"]["dft"][2] == pytest.approx(-3.0103, abs=1e-3)
    assert res["rel"]["off_local"][2] == pytest.approx(0.0)
