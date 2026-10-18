"""End-to-end test of scripts/09_beamform_waterfall.py on a small synthetic window."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from charts_mvdr.beamforming import steering_vectors
from charts_mvdr.np_correlator import correlate
from charts_mvdr.transients import arrival_times_s
from test_data_io import write_astron, write_baseband

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "09_beamform_waterfall.py"
T, N = 256, 16
FRAME = T * 10e-6 / 3


def make_window(tmp, rfi_amp, seed=3):
    """4 x 4 grid, 6 channels over 300-400.5 MHz, DM 1 burst (sweep ~20 ms) at
    (-0.1, 0.25); persistent interferer on channel 2 in a DFT sidelobe."""
    rng = np.random.default_rng(seed)
    k = np.arange(N)
    x, y = (k % 4) * 0.6, (k // 4) * 0.6
    fmhz = np.linspace(300.0, 400.5, 6)
    t_arr = arrival_times_s(fmhz, 1.0, 0.1)
    fidx = sorted(set(range(0, 90, 9)) | set(range(int(0.07 / FRAME), int(0.16 / FRAME))))
    d = tmp / "win"
    d.mkdir(); (tmp / "corr").mkdir()
    with h5py.File(d / "w_meta.h5", "w") as h:
        h["antenna_pos_x_m"], h["antenna_pos_y_m"], h["frequencies_mhz"] = x, y, fmhz
        h["frames/timestamp_s"] = np.array(fidx) * FRAME
        h["frames/output_file_index"] = np.array(fidx)
    ev = [{"event_id": "frb_01", "event_type": "frb", "t_start_s": 0.1, "duration_s": 0.03,
           "dm": 1.0, "pulse_width_ms": 0.5, "l0": -0.1, "m0": 0.25},
          {"event_id": "r", "event_type": "rfi_narrow", "t_start_s": 0.0, "duration_s": 1.0,
           "l0": 0.55, "m0": 0.25, "channels": [2]}]
    (d / "w_events.json").write_text(json.dumps(ev))

    def cn(*s):
        return (rng.standard_normal(s) + 1j * rng.standard_normal(s)) / np.sqrt(2)

    for fi in fidx:
        t = fi * FRAME + np.arange(T) * FRAME / T
        V = np.zeros((T, 6, N), complex)
        for c in range(6):
            f = fmhz[c] * 1e6
            v = 2.0 * np.sqrt(2) * cn(T, N)
            env = np.exp(-0.5 * ((t - t_arr[c]) / 0.5e-3) ** 2)
            v += 3.0 * env[:, None] * cn(T, 1) * steering_vectors(-0.1, 0.25, f, x, y)
            if c == 2:
                v += rfi_amp * cn(T, 1) * steering_vectors(0.55, 0.25, f, x, y)
            V[:, c] = np.clip(np.rint(v.real), -7, 7) + 1j * np.clip(np.rint(v.imag), -7, 7)
        write_baseband(d / f"w_{fi:07d}.bin", V)
        write_astron(tmp / "corr" / f"corr_{fi:07d}.bin", correlate(V))
    return d


def run(tmp):
    out = tmp / "out"
    res = subprocess.run([sys.executable, str(SCRIPT), "--window-dir", str(tmp / "win"),
                          "--corr-dir", str(tmp / "corr"), "--samples", str(T), "--tbin", "8",
                          "--local-s", "0.01", "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return np.load(out / "waterfall.npz"), res.stdout


def test_waterfall_end_to_end(tmp_path):
    a, b = tmp_path / "rfi", tmp_path / "clean"
    a.mkdir(); b.mkdir()
    make_window(a, rfi_amp=5.0)
    make_window(b, rfi_amp=0.0)
    za, out = run(a)
    zb, _ = run(b)
    assert "dedispersed S/N" in out and (a / "out" / "waterfall.png").exists()
    # top channel: leading tail cut (event starts at the top-channel arrival)
    assert za["flag"][-1] and not za["flag"][:-1].any()
    # burst detected in every unflagged channel and after dedispersion
    assert np.all(zb["dft_snr_ch"][:-1] > 5)
    assert zb["dft_dd_snr"] > 15
    # interferer channel: DFT loses S/N, KL recovers most of it (paired: same noise)
    loss = zb["dft_snr_ch"][2] / za["dft_snr_ch"][2]
    rec = zb["dft_snr_ch"][2] / za["kl_local_snr_ch"][2]
    assert loss > 2.0 and rec < 1.3
    # without interference KL and DFT agree
    np.testing.assert_allclose(zb["kl_local_snr_ch"][:-1], zb["dft_snr_ch"][:-1], rtol=0.15)
