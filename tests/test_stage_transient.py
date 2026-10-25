import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME_S = 1536 * 10e-6 / 3


def _window(tmp_path, n_frames=120, n_chan=6, seed=0, rfi_chans=(1, 2)):
    """Paired stage outputs: burst at DM 0 in frame 60, noise ~ 7 % power fluctuation;
    the RFI window adds fluctuating interference to the DFT plane in rfi_chans."""
    rng = np.random.default_rng(seed)
    freqs = 380.0 + 0.3 * np.arange(n_chan)
    win = tmp_path / "win"
    win.mkdir()
    times = np.arange(n_frames) * FRAME_S
    with h5py.File(win / "w_meta.h5", "w") as f:
        f["antenna_pos_x_m"] = np.zeros(4)
        f["antenna_pos_y_m"] = np.zeros(4)
        f["frequencies_mhz"] = freqs
        f["frames/timestamp_s"] = times
        f["frames/output_file_index"] = np.arange(n_frames)
    t0 = 60 * FRAME_S + FRAME_S / 2
    (win / "w_events.json").write_text(json.dumps([{
        "event_type": "frb", "event_id": "b", "dm": 0.0, "pulse_width_ms": 0.5,
        "t_start_s": t0, "l0": 0.0, "m0": 0.0}]))
    noise = 1.0 + 0.07 * rng.standard_normal((n_frames, n_chan))
    sig = np.zeros((n_frames, n_chan))
    sig[60] = 0.5
    clean = np.stack([noise + sig, noise + sig])[:, :, :, None]
    rfi = clean.copy()
    for c in rfi_chans:
        rfi[0, :, c, 0] += 0.5 * np.abs(rng.standard_normal(n_frames))
    for name, P in (("clean", clean), ("rfi", rfi)):
        np.savez(tmp_path / f"{name}.npz", power=P.astype(np.float32),
                 channels=np.arange(n_chan), freqs_mhz=freqs)
    return win


def test_paired_snr(tmp_path):
    win = _window(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "14_stage_transient_snr.py"),
                        "--rfi", str(tmp_path / "rfi.npz"), "--clean",
                        str(tmp_path / "clean.npz"), "--window-dir", str(win),
                        "--groups", "R:1-2", "--detail", "1", "--out", str(tmp_path / "t.npz")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = np.load(tmp_path / "t.npz")
    assert t["usable"].all()
    # MVDR plane is identical to the clean one: 0 dB; RFI only degrades the DFT plane
    np.testing.assert_allclose(t["rfi_mvdr_db"], 0.0, atol=1e-5)
    np.testing.assert_allclose(t["clean_mvdr_db"], 0.0, atol=1e-5)
    assert np.all(t["rfi_dft_db"][[1, 2]] < -3.0)
    np.testing.assert_allclose(t["rfi_dft_db"][[0, 3, 4, 5]], 0.0, atol=1e-5)
    ded = t["dedispersed_db"]
    assert abs(ded[0]) < 1e-5 and abs(ded[2]) < 1e-5 and ded[1] < 0
    assert "R " in r.stdout and "other" in r.stdout and "channel 1:" in r.stdout
    dz = t["dedispersed_z_db"]
    assert abs(dz[0]) < 1e-5 and abs(dz[2]) < 1e-5
    # z-score down-weights the RFI channels: a smaller dedispersed loss than equal weight
    assert ded[1] < dz[1] < 0


def test_waterfall(tmp_path):
    win = _window(tmp_path, n_frames=120, n_chan=6, rfi_chans=(1, 2))
    out = tmp_path / "fig"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "15_stage_waterfall.py"),
                        "--rfi", str(tmp_path / "rfi.npz"), "--clean",
                        str(tmp_path / "clean.npz"), "--window-dir", str(win),
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "stage_waterfall.png").stat().st_size > 10_000
    z = np.load(out / "stage_waterfall.npz")
    clean, dft, mvdr = z["peaks"]
    # MVDR plane equals the clean one in the synthetic pair; RFI lowers the DFT peak
    assert abs(mvdr - clean) < 1e-6 and dft < clean
    # the peak sits at zero dedispersed time (DM 0, burst in frame 60)
    tau = z["tau_0"]
    assert abs(tau[np.nanargmax(z["snr_0"])]) <= 1536 * 10e-6 / 3
