"""07_beam_response on a small synthetic window, then tools/fig_beam_response.py."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from charts_mvdr.beamforming import grating_lobes, steering_vectors
from test_data_io import write_astron

ROOT = Path(__file__).resolve().parents[1]
S07 = ROOT / "scripts" / "07_beam_response.py"
TOOL = ROOT / "tools" / "fig_beam_response.py"
LOOK = (0.4677, 0.4801)
FMHZ = np.array([399.0, 398.4, 350.1])
K = 1536


def make_window(tmp):
    """4 x 4 grid (0.6 m); channels 0, 1: interferer at 0.03 and 0.15 from the grating
    lobe of the look; channel 2: interferer far from any lobe. Exact covariances."""
    k = np.arange(16)
    x, y = (k % 4) * 0.6, (k // 4) * 0.6
    win, corr = tmp / "win", tmp / "corr"
    win.mkdir(); corr.mkdir()
    ev, R = [], np.zeros((3, 16, 16), complex)
    for c, f in enumerate(FMHZ * 1e6):
        if c < 2:
            p, q, ll, mm = grating_lobes(*LOOK, f, 0.6, 0.6)[0]
            li, mi = ll + (0.03, 0.15)[c], mm
        else:
            li, mi = 0.0, -0.6
        d = steering_vectors(*LOOK, f, x, y)[0]
        a = steering_vectors(li, mi, f, x, y)[0]
        R[c] = 8.0 * np.eye(16) + 16.0 * np.outer(a, a.conj()) + 0.5 * np.outer(d, d.conj())
        ev.append({"event_id": f"r{c}", "event_type": "rfi_narrow", "t_start_s": 0.0,
                   "duration_s": 1.0, "l0": li, "m0": mi, "channels": [c], "nominal_amp": 4.0})
    (win / "w_events.json").write_text(json.dumps(ev))
    fidx = [0, 10, 20]
    with h5py.File(win / "w_meta.h5", "w") as h:
        h["antenna_pos_x_m"], h["antenna_pos_y_m"], h["frequencies_mhz"] = x, y, FMHZ
        h["frames/timestamp_s"] = np.array(fidx) * 5.12e-3
        h["frames/output_file_index"] = np.array(fidx)
    for fi in fidx:
        write_astron(corr / f"corr_{fi:07d}.bin", K * R)
    return win, corr


def test_07_and_composite_figure(tmp_path):
    win, corr = make_window(tmp_path)
    out = tmp_path / "br"
    res = subprocess.run([sys.executable, str(S07), "--corr-dir", str(corr), "--window-dir", str(win),
                          "--channels", "0", "1", "2", "--look", *map(str, LOOK), "--loading", "0", "10",
                          "--grid", "101", "--samples", str(K), "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    z = np.load(out / "beam_response.npz")
    assert z["pos_x"].size == 16 and abs(float(z["ch0_noise_floor"]) - 8.0) < 0.5
    npz = out / "beam_response.npz"
    res = subprocess.run([sys.executable, str(TOOL), "--normal", f"{npz}:2", "--grating", f"{npz}:0",
                          "--events", str(win / "w_events.json"), "--out", str(tmp_path / "f.png")],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "f.png").exists()
    sys.path.insert(0, str(TOOL.parent))
    import fig_beam_response as fb
    rows = fb.sweep_table(dict(np.load(npz)), {0: 4.0, 1: 4.0, 2: 4.0}, None, None)
    off = {r["ch"]: r["offset"] for r in rows}
    assert abs(off[0] - 0.03) < 1e-6 and abs(off[1] - 0.15) < 1e-6
    for r in rows:   # exact covariances: measured equals ideal (noise floor ~ 8)
        for n in ("MVDR d=0", "MVDR d=10"):
            assert abs(r[n][1] - r[n + " ideal"][1]) < 0.3       # WNG
    # closer to the lobe: higher WNG for unloaded MVDR
    assert rows[0]["MVDR d=0"][1] > rows[1]["MVDR d=0"][1]
