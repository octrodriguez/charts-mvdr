"""Tests of tracking.expand_gains, tools/kotekan_gains.py and scripts/18_kotekan_real_mvdr.py.

Synthetic run directory: 64-element correlation frames in the layout of the Kotekan
correlator (S = sum conj(v) v, lower block triangle, metadata header), two frequency
blocks, four signal inputs with complex gains, a look source and an interferer. The
Kotekan stage is stood in by its NumPy reference (stage_reference.run_stage, which
mirrors chartsMVDRBeams) fed exactly as the stage is: raw frames, simulator sign,
gains conjugated on load. 17_real_nulling.py runs on the charts-utils view of the same
frames (logical order, V = S, file gains). compare must find them equal.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from charts_mvdr import tracking
from charts_mvdr.beamforming import steering_vectors
from charts_mvdr.data_io import read_astron_correlation
from charts_mvdr.stage_reference import StageConfig, element_to_antenna, run_stage

ROOT = Path(__file__).resolve().parents[1]
S18 = ROOT / "scripts" / "18_kotekan_real_mvdr.py"
S17 = ROOT / "scripts" / "17_real_nulling.py"
TOOL = ROOT / "tools" / "kotekan_gains.py"

N_EL, NF, N_BLK, T, K = 64, 6, 2, 40, 3000
ANTS = [0, 4, 32, 36]
POS = {0: (0.0, 0.0), 4: (-4.2, 0.0), 32: (0.0, 4.2), 36: (-4.2, 4.2)}
F0, DF = 380.0, 0.3
SAMPLES, ACC = 1000, 3                      # samples_per_data_set, frames accumulated
LOOK, RFI = (0.0, 0.31), (0.45, -0.2)
META = b"\x07" * 24


def run(script, *args):
    res = subprocess.run([sys.executable, str(script), *map(str, args)],
                         capture_output=True, text=True)
    return res


def write_astron(path, R, meta=META, block_size=2):
    S = np.conj(R)
    n_rec = R.shape[1] // block_size
    ry, rx = np.tril_indices(n_rec)
    out = np.zeros((R.shape[0], len(ry), block_size, block_size, 2), dtype="<i4")
    for py in range(block_size):
        for px in range(block_size):
            v = S[:, ry * block_size + py, rx * block_size + px]
            out[..., py, px, 0] = np.rint(v.real)
            out[..., py, px, 1] = np.rint(v.imag)
    with open(path, "wb") as f:
        np.uint32(len(meta)).tofile(f)
        f.write(meta)
        out.tofile(f)


def run_config():
    """Minimal run config with the structure of a CHARTS correlator config."""
    cfg = {"num_elements": N_EL, "num_local_freq": NF, "block_size": 2,
           "samples_per_data_set": SAMPLES, "integration_frames": ACC,
           "cpu_affinity": [6, 7], "main_pool": {"kotekan_metadata_pool": "chartsMetadata"},
           "dpdk": {"kotekan_stage": "dpdkCore", "lcore_cpu_map": [3, 5], "main_lcore_cpu": 4,
                    "handlers": []}, "gpu": {}}
    for b in range(N_BLK):
        cfg["dpdk"]["handlers"].append({"coarse_freq_start": b * NF,
                                        "out_buffer": f"network_capture_buf_{b}"})
        cfg["gpu"][f"gpu_{b}"] = {"kotekan_stage": "cudaProcess",
                                  "in_buffers": {"host_voltage": f"network_capture_buf_{b}"},
                                  "out_buffers": {"host_correlation": f"host_correlation_buffer_{b}"}}
        cfg[f"charts_accumulate_{b}"] = {"kotekan_stage": "chartsAccumulate",
                                         "in_buf": f"host_correlation_buffer_{b}",
                                         "num_frames_to_accumulate": "integration_frames",
                                         "out_buf": f"integrated_correlation_buffer_{b}"}
        cfg[f"integrated_correlation_buffer_{b}"] = {"metadata_pool": "main_pool"}
        cfg[f"raw_file_write_{b}"] = {"kotekan_stage": "rawFileWrite", "file_name": f"corr_freq_{b}",
                                      "in_buf": f"integrated_correlation_buffer_{b}"}
    return cfg


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Run directory with frames, run config/metadata, a 4-column solution and positions."""
    d = tmp_path_factory.mktemp("site")
    run_dir = d / "run"
    run_dir.mkdir()
    rng = np.random.default_rng(7)
    n_ch = N_BLK * NF
    freqs = F0 + DF * np.arange(n_ch)
    # charts-utils file gains (x_obs = g x); the voltages carry conj(g)
    g_file = (1 + 0.3 * rng.standard_normal((n_ch, 4))) * np.exp(1j * rng.uniform(-np.pi, np.pi, (n_ch, 4)))
    g_file[:, 0] = np.abs(g_file[:, 0])                  # reference antenna phase 0
    elem = {a: N_EL - 1 - a for a in ANTS}               # descending order
    x = np.array([POS[a][0] for a in ANTS])
    y = np.array([POS[a][1] for a in ANTS])
    for t in range(T):
        for b in range(N_BLK):
            R = np.zeros((NF, N_EL, N_EL), dtype=np.complex128)
            for c in range(NF):
                ch = b * NF + c
                f = freqs[ch] * 1e6
                a_s = steering_vectors(*LOOK, f, x, y)[0]
                a_r = steering_vectors(*RFI, f, x, y)[0]
                s = np.sqrt(4 / 2) * (rng.standard_normal(K) + 1j * rng.standard_normal(K))
                r = np.sqrt(40 / 2) * (rng.standard_normal(K) + 1j * rng.standard_normal(K))
                v = (rng.standard_normal((N_EL, K)) + 1j * rng.standard_normal((N_EL, K))) * 2
                for i, a in enumerate(ANTS):
                    v[elem[a]] += np.conj(g_file[ch, i]) * (a_s[i] * s + a_r[i] * r)
                R[c] = v @ v.conj().T
            write_astron(run_dir / f"ms-x00_corr_freq_{b}_{t:07d}.bin", R)
    (run_dir / "kotekan_config_20261104T135502Z.json").write_text(json.dumps(run_config()))
    (run_dir / "run_metadata_20261104T135502Z.json").write_text(
        json.dumps({"array": {"index_convention": "charts_logical", "n_antennas": N_EL},
                    "signal_antennas": ANTS}))
    g_sol = g_file.astype(np.complex64)
    g_sol[[3, 8], :] = np.nan                            # flagged channels
    np.save(d / "gains_sol.npy", g_sol)
    pos = np.zeros((N_EL, 3))
    for a in ANTS:
        pos[a, :2] = POS[a]
    np.save(d / "positions.npy", pos)
    return d, run_dir, freqs


def test_expand_gains():
    g = np.arange(8).reshape(2, 4) * (1 + 1j)
    out = tracking.expand_gains(g, [3, 0, 7, 5], 8)
    assert out.shape == (2, 8)
    np.testing.assert_array_equal(out[:, [3, 0, 7, 5]], g)
    assert np.isnan(out[:, [1, 2, 4, 6]]).all()
    for bad in ([0, 0, 1, 2], [0, 1, 2, 8], [0, 1, 2]):
        with pytest.raises(ValueError):
            tracking.expand_gains(g, bad, 8)


def test_kotekan_gains_tool(site, tmp_path):
    d, run_dir, _ = site
    out = tmp_path / "full.npy"
    res = run(TOOL, "--gains", d / "gains_sol.npy", "--gain-antennas", "0,4,32,36",
              "--run-metadata", run_dir / "run_metadata_20261104T135502Z.json",
              "--interp-gains", "--out", out)
    assert res.returncode == 0, res.stderr
    full = np.load(out)
    sol = np.load(d / "gains_sol.npy")
    assert full.shape == (N_BLK * NF, N_EL) and full.dtype == np.complex64
    ok = np.isfinite(sol).all(axis=1)
    np.testing.assert_array_equal(full[ok][:, ANTS], sol[ok])
    assert np.isfinite(full[:, ANTS]).all()               # flagged channels interpolated
    assert np.isnan(np.delete(full, ANTS, axis=1)).all()
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["antennas"] == ANTS and "2 channels interpolated" in side["notes"]
    # without interpolation the flagged channels stay NaN; width mismatch is refused
    res = run(TOOL, "--gains", d / "gains_sol.npy", "--gain-antennas", "0,4,32,36",
              "--num-inputs", N_EL, "--out", tmp_path / "raw.npy")
    assert res.returncode == 0 and np.isnan(np.load(tmp_path / "raw.npy")[[3, 8]]).all()
    res = run(TOOL, "--gains", d / "gains_sol.npy", "--gain-antennas", "0,4,32",
              "--num-inputs", N_EL, "--out", tmp_path / "x.npy")
    assert res.returncode == 1


def prepare(site, tmp_path, first=0, count=T, extra=()):
    d, run_dir, _ = site
    full = tmp_path / "full.npy"
    assert run(TOOL, "--gains", d / "gains_sol.npy", "--gain-antennas", "0,4,32,36",
               "--num-inputs", N_EL, "--interp-gains", "--out", full).returncode == 0
    out = tmp_path / "prep"
    res = run(S18, "prepare", "--input", run_dir, "--first", first, "--count", count,
              "--gains", full, "--positions", d / "positions.npy",
              "--beams", f"{LOOK[0]},{LOOK[1]}", "0.2,0.1", "--half-window", 8, "--guard", 1,
              "--loading", 0.1, "--loading-ref", "noise",
              "--freq-start-mhz", F0, "--delta-freq-mhz", DF, *extra, "--out", out)
    return res, out, full


def test_prepare_window_and_config(site, tmp_path):
    res, out, full = prepare(site, tmp_path, first=5, count=30)
    assert res.returncode == 0, res.stderr
    win = out / "window"
    for b in range(N_BLK):
        links = sorted(win.glob(f"ms-x00_corr_freq_{b}_*.bin"))
        assert len(links) == 30
        assert links[0].resolve().name == f"ms-x00_corr_freq_{b}_0000005.bin"
    assert (win / "kotekan_config_20261104T135502Z.json").exists()
    man = json.loads((out / "manifest.json").read_text())
    assert man["samples_per_frame"] == SAMPLES * ACC
    assert man["gain_antennas"] == ANTS and man["metadata_pool"] == "chartsMetadata"
    assert [man["blocks"][str(b)]["first_channel"] for b in range(N_BLK)] == [0, NF]
    assert not set(man["cpu_cores"]) & {3, 4, 5, 6, 7}
    for b in range(N_BLK):
        y = yaml.safe_load((out / f"block{b}" / "mvdr.yaml").read_text())
        st, rd = y["mvdr_beams"], y["correlation_reader"]
        assert rd["file_name"] == f"ms-x00_corr_freq_{b}" and rd["prefix_hostname"] is False
        assert st["samples_per_frame"] == SAMPLES * ACC
        assert st["gains_first_channel"] == b * NF
        assert st["freq_start_mhz"] == pytest.approx(F0 + DF * b * NF)
        e2a = element_to_antenna(N_EL, "descending")
        for a in ANTS:
            e = int(np.flatnonzero(e2a == a)[0])
            assert (st["positions_x_m"][e], st["positions_y_m"][e]) == POS[a]
        assert y["main_pool"]["kotekan_metadata_pool"] == "chartsMetadata"
        assert y["mvdr_power_dump"]["exit_after_n_files"] == 30


def test_prepare_refuses_narrow_gains_and_missing_frames(site, tmp_path):
    d, run_dir, _ = site
    res = run(S18, "prepare", "--input", run_dir, "--first", 0, "--count", 10,
              "--gains", d / "gains_sol.npy", "--positions", d / "positions.npy",
              "--beams", "0,0", "--freq-start-mhz", F0, "--delta-freq-mhz", DF,
              "--out", tmp_path / "p1")
    assert res.returncode == 1 and "kotekan_gains.py" in res.stderr
    res, _, _ = prepare(site, tmp_path / "p2", first=T - 5, count=10)
    assert res.returncode == 1 and "missing" in res.stderr


def stand_in_stage(out):
    """Write mvdr_power files as chartsMVDRBeams would, from its NumPy reference."""
    man = json.loads((out / "manifest.json").read_text())
    gains_file = np.load(man["gains_file"]).astype(np.complex128)
    pos = np.load(man["positions_file"])
    e2a = element_to_antenna(man["num_elements"], man["antenna_order"])
    cfg = StageConfig(beams=np.asarray(man["beams"]), positions_x_m=pos[e2a, 0],
                      positions_y_m=pos[e2a, 1], samples_per_frame=man["samples_per_frame"],
                      half_window=man["half_window"], guard=man["guard"],
                      min_frames=man["min_frames"], update_every=man["update_every"],
                      loading=man["loading"], loading_ref=man["loading_ref"],
                      antenna_order=man["antenna_order"])
    for b, info in man["blocks"].items():
        files = sorted((out / "window").glob(f"{info['file_name']}_*.bin"))
        R = np.stack([read_astron_correlation(p, man["num_local_freq"], man["num_elements"])
                      for p in files])
        k0 = info["first_channel"]
        f_hz = (info["freq_start_mhz"] + man["delta_freq_mhz"] * np.arange(man["num_local_freq"])) * 1e6
        g_v = np.conj(gains_file[k0:k0 + man["num_local_freq"]])     # load_gains_npy
        res = run_stage(R, f_hz, cfg, gains=g_v)
        mdir = out / f"block{b}" / "mvdr"
        for t in range(R.shape[0]):
            with open(mdir / f"mvdr_power_{t:07d}.bin", "wb") as f:
                np.uint32(len(META)).tofile(f)
                f.write(META)
                res.power[:, t].astype("<f4").tofile(f)


def charts_utils_cube(out, freqs, path):
    """The cube 16_rfi_census.py extract would save: logical order, V = S (raw sums)."""
    man = json.loads((out / "manifest.json").read_text())
    e = [N_EL - 1 - a for a in ANTS]
    blocks = []
    for b, info in sorted(man["blocks"].items()):
        files = sorted((out / "window").glob(f"{info['file_name']}_*.bin"))
        R = np.stack([read_astron_correlation(p, NF, N_EL) for p in files])   # (t, f, i, j)
        S = np.conj(R)[:, :, e][:, :, :, e]
        blocks.append(np.moveaxis(S, (0, 1), (3, 2)))                         # (i, j, f, t)
    vis = np.concatenate(blocks, axis=2).astype(np.complex64)
    np.savez(path, vis=vis, times=np.arange(vis.shape[3]) * 10.24, freqs=freqs,
             ants=np.array(ANTS))


def run_17(site, out, sign, tag, look=LOOK):
    d, _, freqs = site
    cube = out / "vis_window.npz"
    if not cube.exists():
        charts_utils_cube(out, freqs, cube)
    res = run(S17, "--vis", cube, "--gains", d / "gains_sol.npy", "--gain-antennas", "0,4,32,36",
              "--interp-gains", "--positions", d / "positions.npy", "--look", f"{look[0]},{look[1]}",
              "--sign", sign, "--clean", f"{F0}-{F0 + 4}", "--half-window", 8, "--guard", 1,
              "--loading", 0.1, "--loading-ref", "median", "--out", out / tag)
    assert res.returncode == 0, res.stderr
    return res, out / tag


def test_compare_end_to_end(site, tmp_path):
    res, out, _ = prepare(site, tmp_path)
    assert res.returncode == 0, res.stderr
    stand_in_stage(out)
    r17, nul = run_17(site, out, "auto", "nulling")
    assert "-> -1" in r17.stdout                      # charts-utils cube: conjugated steering
    res = run(S18, "compare", "--prepared", out, "--nulling", nul, "--beam", 0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "KOTEKAN REAL MATCH" in res.stdout
    line = next(l for l in res.stdout.splitlines() if l.startswith("scale"))
    assert "x samples per frame = 1 " in line or "x samples per frame = 1)" in line
    # the interferer is rejected: MVDR below DFT
    ratio = next(l for l in res.stdout.splitlines() if l.startswith("stage MVDR/DFT"))
    assert float(ratio.split("channels")[1].split("dB")[0]) < -0.5


def test_compare_detects_wrong_convention_and_inputs(site, tmp_path):
    res, out, _ = prepare(site, tmp_path)
    assert res.returncode == 0, res.stderr
    stand_in_stage(out)
    _, nul = run_17(site, out, "1", "wrong_sign")      # simulator sign on the charts-utils cube
    res = run(S18, "compare", "--prepared", out, "--nulling", nul, "--beam", 0)
    assert res.returncode == 2 and "KOTEKAN REAL MISMATCH" in res.stdout
    _, nul = run_17(site, out, "-1", "other_look", look=(0.2, 0.1))
    res = run(S18, "compare", "--prepared", out, "--nulling", nul, "--beam", 0)
    assert res.returncode == 1 and "is not beam 0" in res.stderr
    res = run(S18, "compare", "--prepared", out, "--nulling", nul, "--beam", 1)
    assert res.returncode == 0 and "KOTEKAN REAL MATCH" in res.stdout
