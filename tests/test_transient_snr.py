"""End-to-end test of scripts/08_transient_snr.py on a small synthetic window."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from charts_mvdr.beamforming import steering_vectors
from charts_mvdr.np_correlator import correlate
from charts_mvdr.transients import arrival_times_s
from test_data_io import write_astron

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_transient_snr.py"


def make_window(tmp, seed=1):
    """4 x 4 grid, 3 channels, 256-sample frames; burst at (-0.1, 0.25) with DM 10;
    channel 1 has an interferer active only around the burst (longer than the
    off-local window, so it is stationary between on and off-local frames)."""
    rng = np.random.default_rng(seed)
    T, N, nf = 256, 16, 3
    frame = T * 10e-6 / 3
    k = np.arange(N)
    x, y = (k % 4) * 0.6, (k // 4) * 0.6
    fmhz = np.array([330.0, 357.6, 400.5])
    t_arr = arrival_times_s(fmhz, 10.0, 0.1)
    fidx = list(range(0, 120, 6)) + list(range(int(0.09 / frame), int(0.26 / frame)))
    fidx = sorted(set(fidx))
    (tmp / "win").mkdir(); (tmp / "corr").mkdir()
    with h5py.File(tmp / "win" / "w_meta.h5", "w") as h:
        h["antenna_pos_x_m"], h["antenna_pos_y_m"], h["frequencies_mhz"] = x, y, fmhz
        h["frames/timestamp_s"] = np.array(fidx) * frame
        h["frames/output_file_index"] = np.arange(len(fidx))
    rfi_lm, rfi_t = (0.15, 0.10), (t_arr[1] - 0.04, t_arr[1] + 0.04)
    ev = [{"event_id": "frb_01", "event_type": "frb", "t_start_s": 0.1, "duration_s": 0.08,
           "dm": 10.0, "pulse_width_ms": 1.0, "l0": -0.1, "m0": 0.25},
          {"event_id": "r", "event_type": "rfi_narrow", "t_start_s": rfi_t[0],
           "duration_s": 0.08, "l0": rfi_lm[0], "m0": rfi_lm[1], "channels": [1]}]
    (tmp / "win" / "w_events.json").write_text(json.dumps(ev))

    def cn(*s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)

    for o, fi in enumerate(fidx):
        t = fi * frame + np.arange(T) * frame / T
        V = np.zeros((T, nf, N), complex)
        for c in range(nf):
            f = fmhz[c] * 1e6
            v = 2.0 * cn(T, N)
            env = np.exp(-0.5 * ((t - t_arr[c]) / 1e-3) ** 2)
            v += 3.0 * env[:, None] * cn(T, 1) * steering_vectors(-0.1, 0.25, f, x, y)
            if c == 1:
                on = (t >= rfi_t[0]) & (t <= rfi_t[1])
                v += 12.0 * on[:, None] * cn(T, 1) * steering_vectors(*rfi_lm, f, x, y)
            V[:, c, :] = v
        R = correlate(np.round(V * 4) / 4)          # fine rounding, no clipping
        write_astron(tmp / "corr" / f"corr_{o:07d}.bin", np.round(R))
    return tmp


def test_transient_snr_end_to_end(tmp_path):
    make_window(tmp_path)
    out = tmp_path / "out"
    res = subprocess.run([sys.executable, str(SCRIPT), "--corr-dir", str(tmp_path / "corr"),
                          "--window-dir", str(tmp_path / "win"), "--channels", "0", "1",
                          "--samples", "256", "--local-s", "0.03", "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    z = np.load(out / "transient_snr.npz")
    w = list(z["weights"])
    gain = dict(zip(w, z["gain_db"][1]))           # channel 1: interferer only near the burst
    ret = dict(zip(w, z["ret"][0]))
    assert w[0] == "dft" and gain["dft"] == 0.0
    # off-pulse frames far from the burst do not contain the interferer
    assert gain["off_local_d0"] > gain["off_global_d0"] + 3.0
    assert ret["off_global_d0"] > 0.95             # clean channel: KL keeps the signal
    assert (out / "transient_snr.png").exists()
