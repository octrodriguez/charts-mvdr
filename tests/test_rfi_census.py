"""Tests of charts_mvdr.rfi_census and scripts/16_rfi_census.py on synthetic autocorrelations."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr import rfi_census as rc

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "16_rfi_census.py"
F, T = 672, 200
PERSISTENT, INTERMITTENT, VARIABLE = 100, 300, 500


def synthetic_cube(seed=0, n_ant=3):
    rng = np.random.default_rng(seed)
    freqs = 300.0 + 0.3 * np.arange(F)
    band = 1.0 + 0.5 * np.sin(np.linspace(0, np.pi, F))
    vis = np.zeros((n_ant, n_ant, F, T), dtype=np.complex64)
    for a in range(n_ant):
        auto = band[:, None] * (1 + 0.01 * rng.standard_normal((F, T)))
        auto[PERSISTENT] *= 1.3
        on = rng.random(T) < 0.2
        auto[INTERMITTENT, on] *= 1.5
        auto[VARIABLE] *= 1 + 0.2 * rng.standard_normal(T)
        auto[:, 10] *= 0.3                     # heavy packet loss: invalid frame
        auto[:, 20] *= 0.8                     # mild packet loss: normalized away
        vis[a, a] = auto
    return vis, freqs


def test_normalize_frames_marks_packet_loss():
    vis, _ = synthetic_cube()
    a, valid = rc.normalize_frames(vis[0, 0].real)
    assert not valid[10] and valid[20] and valid.sum() == T - 1
    assert np.isnan(a[:, 10]).all()
    assert abs(np.nanmedian(a[:, 20]) - 1.0) < 1e-6


def test_per_block_normalization_detects_loss_in_one_block():
    rng = np.random.default_rng(2)
    auto = 1 + 0.01 * rng.standard_normal((F, T))
    auto[F // 2:] *= 1 - 0.1 * rng.random(T)       # mild time-variable loss in the upper block
    auto[F // 2:, 7] *= 0.3                         # heavy loss in the upper block of frame 7
    _, valid1 = rc.normalize_frames(auto, n_blocks=1)
    a2, valid2 = rc.normalize_frames(auto, n_blocks=2)
    assert valid1[7] and not valid2[7] and valid2.sum() == T - 1
    med = np.nanmedian(a2[F // 2:, valid2], axis=0)
    assert np.allclose(med, 1.0)                    # each block normalized to its own median
    with pytest.raises(ValueError):
        rc.normalize_frames(auto[:-1], n_blocks=2)


def test_running_median_matches_direct():
    x = np.random.default_rng(1).standard_normal((50, 7))
    rm = rc.running_median(x, 5)
    pad = np.pad(x, ((2, 2), (0, 0)), mode="reflect")
    direct = np.array([[np.median(pad[f:f + 5, t]) for t in range(7)] for f in range(50)])
    assert np.allclose(rm, direct)
    with pytest.raises(ValueError):
        rc.running_median(x, 4)


def test_classes_from_occupancy():
    vis, _ = synthetic_cube()
    a, valid = rc.normalize_frames(vis[0, 0].real)
    _, flags = rc.spectral_peaks(a, window=31, k=5.0)
    occ = rc.occupancy(flags, valid)
    cls = rc.classify(occ)
    assert rc.CLASSES[cls[PERSISTENT]] == "persistent"
    assert rc.CLASSES[cls[INTERMITTENT]] == "intermittent"
    assert np.mean(cls == 0) > 0.98            # few false positives at k = 5
    assert 0.1 < occ[INTERMITTENT] < 0.35


def test_array_class_and_cross_counts():
    cls = np.array([[3, 0, 2, 1], [3, 2, 0, 1], [0, 0, 2, 3]])
    assert rc.array_class(cls).tolist() == [3, 0, 2, 1]
    labels = np.array([["source", "none", "stationary", "none"],
                       ["source", "none", "stationary", "source"],
                       ["none", "source", "stationary", "source"]])
    maj = rc.fringe_majority(labels)
    assert maj.tolist() == ["source", "none", "stationary", "source"]
    tie = np.array([["source", "stationary"], ["stationary", "stationary"]])
    assert rc.fringe_majority(tie).tolist() == ["ambiguous", "stationary"]
    t = rc.cross_counts(rc.array_class(cls), maj, ["source", "stationary", "none"])
    assert t.sum() == 4 and t[3, 0] == 1 and t[2, 1] == 1 and t[0, 2] == 1 and t[1, 0] == 1


def load_script():
    spec = importlib.util.spec_from_file_location("rfi_census_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_cube_stride_and_chunks():
    mod = load_script()
    groups = list(range(23))
    calls = []

    def loader(g, ants, nf):
        calls.append(len(g))
        out = np.zeros((len(ants), len(ants), nf, len(g)), dtype=np.complex64)
        out[0, 0, 0] = g                       # frame index as a marker
        return out

    vis, t = mod.extract_cube(groups, np.arange(23) * 10.0, [0, 4], 3, loader, stride=4, chunk=2,
                              log=lambda m: None)
    assert vis.shape == (2, 2, 3, 6) and calls == [2, 2, 2]
    assert vis[0, 0, 0].real.tolist() == [0, 4, 8, 12, 16, 20]
    assert t.tolist() == [0, 40, 80, 120, 160, 200]


def run_census(tmp_path, extra):
    vis, freqs = synthetic_cube()
    np.savez(tmp_path / "vis.npz", vis=vis, times=np.arange(T) * 10.24, freqs=freqs, ants=np.array([0, 4, 5]))
    pairs = 3
    labels = np.full((pairs, F), "none", dtype="<U10")
    labels[:, PERSISTENT] = "stationary"
    labels[:, 200:260] = "source"
    np.savez(tmp_path / "fringe.npz", freqs=freqs, labels=labels)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(SCRIPT), "census", "--vis", str(tmp_path / "vis.npz"),
                        "--fringe", str(tmp_path / "fringe.npz"), "--out", str(out)] + extra,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return np.load(out / "rfi_census.npz"), out, r.stdout


def test_census_end_to_end(tmp_path):
    z, out, stdout = run_census(tmp_path, ["--no-variability"])
    assert (out / "rfi_census.png").exists()
    names = list(z["fringe_names"])
    t = z["table"]
    assert t[3, names.index("stationary")] == 1              # persistent channel is stationary
    assert t.sum() == F
    assert z["array_cls"][INTERMITTENT] == 2
    assert "valid per antenna: 0: 199" in stdout
    assert z["flags"].shape == (3, F, T) and not z["flags"][:, :, 10].any()
    assert z["flags"][0, PERSISTENT].mean() > 0.9 and z["times"].shape == (T,)


def test_census_with_mad_rfi_zapper(tmp_path):
    pytest.importorskip("charts_utils.cal.algorithms")
    z, _, _ = run_census(tmp_path, ["--var-sigma", "3"])
    assert z["array_cls"][VARIABLE] >= 1
    assert np.mean(z["array_cls"] == 0) > 0.95
