"""Tests for charts_mvdr.transients and tools/make_frb_events.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr.transients import K_DM, arrival_times_s, dispersion_delay_s, on_pulse_mask

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_frb_events.py"
FREQS = 300.0 + 0.3 * np.arange(336)
EVENT_KEYS = {"event_id", "event_type", "t_start_s", "duration_s", "dm", "nominal_amp",
              "l0", "m0", "drift_dl", "drift_dm", "pulse_period_s", "pulse_width_ms",
              "channels", "is_persistent", "description", "ra_deg", "dec_deg",
              "source_name"}


def test_dispersion_delay_values():
    d = dispersion_delay_s(FREQS, 10.0)
    assert d[-1] == 0.0 and np.all(np.diff(d) < 0)
    # sweep over 300-400.5 MHz: K_DM DM (1/300^2 - 1/400.5^2) ~ 20.2 ms per unit DM
    assert d[0] == pytest.approx(K_DM * 10 * (1 / 300.0 ** 2 - 1 / 400.5 ** 2))
    assert d[0] == pytest.approx(0.2023, abs=2e-4)
    assert dispersion_delay_s(350.0, 5.0, f_ref_mhz=350.0) == 0.0


def test_arrival_and_on_pulse_mask():
    t_arr = arrival_times_s(FREQS, 10.0, 1.0)
    assert t_arr[-1] == 1.0
    frame = 1536 * 10e-6 / 3
    times = np.arange(0.9, 1.3, frame)
    mask = on_pulse_mask(times, FREQS, 10.0, 1.0, 2e-3, n_sigma=3.0, frame_s=frame)
    assert mask.shape == (times.size, FREQS.size)
    assert np.all(mask.sum(axis=0) >= 1)           # every channel has on-pulse frames
    assert np.all(mask.sum(axis=0) <= 4)           # 12 ms extent overlaps at most 4 frames
    k = np.argmax(mask[:, 0])
    assert times[k] <= t_arr[0] + 6e-3 and times[k] + frame >= t_arr[0] - 6e-3


def run_tool(tmp_path, *extra):
    out = tmp_path / "ev.json"
    cmd = [sys.executable, str(TOOL), "--out", str(out), *extra]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res, out


def test_tool_frb_and_rfi(tmp_path):
    res, out = run_tool(
        tmp_path, "--frb", "-0.1", "0.25", "--dm", "10", "--width-ms", "2", "--t-top", "1.0",
        "--rfi", "id=site l=-0.83 m=-0.482 amp=7 t0=0 dur=2 ch=94,133",
        "--rfi", "id=burst_rfi l=0.858 m=0.43 amp=7 t0=burst dur=0.1 ch=192")
    assert res.returncode == 0, res.stderr
    ev = json.loads(out.read_text())
    assert [e["event_id"] for e in ev] == ["frb_01", "site", "burst_rfi"]
    assert all(set(e) == EVENT_KEYS for e in ev)
    frb, site, burst = ev
    assert frb["event_type"] == "frb" and frb["dm"] == 10.0 and frb["pulse_width_ms"] == 2.0
    assert frb["duration_s"] == pytest.approx(0.2023 + 0.012, abs=1e-3)
    assert site["channels"] == [94, 133] and site["is_persistent"] is False
    t192 = float(arrival_times_s(FREQS, 10.0, 1.0)[192])
    assert burst["t_start_s"] + 0.05 == pytest.approx(t192)
    assert "--event-dense-s 0.215" in res.stdout
    assert "RFI overlap   : burst_rfi on channels [192]" in res.stdout


def test_tool_append_and_errors(tmp_path):
    res, out = run_tool(tmp_path, "--frb", "0", "0")
    assert res.returncode == 0
    res, _ = run_tool(tmp_path, "--append", "--rfi", "l=0 m=0 amp=1 t0=0 dur=1 ch=3")
    assert res.returncode == 0 and len(json.loads(out.read_text())) == 2
    res, _ = run_tool(tmp_path, "--append", "--frb", "0", "0")       # duplicated frb_01
    assert res.returncode != 0 and "duplicated" in res.stderr
    res, _ = run_tool(tmp_path, "--rfi", "l=0 m=0 amp=1 t0=0 dur=1")  # missing ch
    assert res.returncode != 0 and "missing" in res.stderr
    res, _ = run_tool(tmp_path, "--rfi", "l=0 m=0 amp=1 t0=0 dur=1 ch=400")
    assert res.returncode != 0 and "outside" in res.stderr
