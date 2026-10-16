"""Tests for grating lobes, fit_source_powers(fixed=...) and tools/make_grating_lobe_events.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from charts_mvdr.beamforming import fit_source_powers, grating_lobes, steering_vectors

TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_grating_lobe_events.py"
LOOK = (0.47, 0.48)
D = 0.6
K = np.arange(64)
X, Y = (K % 8) * D, (K // 8) * D
# Fields of the simulator events file (<window>_events.json)
EVENT_KEYS = {"event_id", "event_type", "t_start_s", "duration_s", "dm", "nominal_amp",
              "l0", "m0", "drift_dl", "drift_dm", "pulse_period_s", "pulse_width_ms",
              "channels", "is_persistent", "description", "ra_deg", "dec_deg",
              "source_name"}


def test_steering_vector_at_grating_lobe_equals_look():
    f = 399e6
    lobes = grating_lobes(*LOOK, f, D, D)
    assert lobes, "expected visible lobes at 399 MHz with 0.6 m spacing"
    d = steering_vectors(*LOOK, f, X, Y)[0]
    for p, q, l, m in lobes:
        assert l * l + m * m < 1.0
        np.testing.assert_allclose(steering_vectors(l, m, f, X, Y)[0], d, atol=1e-9)


def test_visible_lobes_at_399_mhz():
    orders = {(p, q) for p, q, _, _ in grating_lobes(*LOOK, 399e6, D, D)}
    assert orders == {(-1, 0), (0, -1)}


def test_no_visible_lobe_for_half_wavelength_spacing():
    f = 300e6
    assert grating_lobes(0.0, 0.0, f, 0.5, 0.5) == []


def model_covariance(A, p, noise=1.0):
    R = noise * np.eye(A.shape[1], dtype=complex)
    for a, pk in zip(A, p):
        R += pk * np.outer(a, a.conj())
    return R


def test_fixed_power_matches_joint_fit_when_separable():
    f = 350e6
    A = steering_vectors([LOOK[0], -0.8], [LOOK[1], -0.4], f, X, Y)
    R = model_covariance(A, [0.3, 20.0])
    p_joint, n_joint = fit_source_powers(R, A)
    p_fixed, n_fixed = fit_source_powers(R, A, fixed={0: 0.3})
    np.testing.assert_allclose(p_joint, [0.3, 20.0], rtol=1e-9)
    np.testing.assert_allclose(p_fixed, p_joint, rtol=1e-9)
    np.testing.assert_allclose(n_fixed, n_joint, rtol=1e-9)


def test_fixed_power_resolves_collinear_sources():
    f = 399e6
    l, m = grating_lobes(*LOOK, f, D, D)[0][2:]
    A = steering_vectors([LOOK[0], l], [LOOK[1], m], f, X, Y)
    R = model_covariance(A, [0.4, 16.0], noise=8.0)
    p, noise = fit_source_powers(R, A, fixed={0: 0.4})
    np.testing.assert_allclose(p, [0.4, 16.0], rtol=1e-9)
    np.testing.assert_allclose(noise, 8.0, rtol=1e-9)


def run_tool(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


BASE = ["--look", *LOOK, "--spacing", D, D, "--amp", 4, "--duration", 5,
        "--noise", 8, "--p-look", 0.4]


def test_tool_writes_events_at_requested_offsets(tmp_path):
    out = tmp_path / "events.json"
    r = run_tool(*BASE, "--lobe", -1, 0, "--direction", 1, 0, "--channels", 330, 331,
                 "--offsets", 0, 0.05, "--out", out)
    assert r.returncode == 0, r.stderr
    events = json.loads(out.read_text())
    assert [set(ev) for ev in events] == [EVENT_KEYS] * 2
    assert all(ev["event_type"] == "rfi_narrow" and not ev["is_persistent"] for ev in events)
    for ev, off in zip(events, (0.0, 0.05)):
        ch = ev["channels"][0]
        lobe = {(p, q): (l, m) for p, q, l, m in
                grating_lobes(*LOOK, (300.0 + 0.3 * ch) * 1e6, D, D)}[(-1, 0)]
        assert ev["l0"] == pytest.approx(lobe[0] + off)
        assert ev["m0"] == pytest.approx(lobe[1])


def test_tool_append_and_duplicate_ids(tmp_path):
    out = tmp_path / "events.json"
    assert run_tool(*BASE, "--lobe", -1, 0, "--channels", 330, "--offsets", 0,
                    "--out", out).returncode == 0
    r = run_tool(*BASE, "--lobe", 0, -1, "--direction", 0, 1, "--channels", 320,
                 "--offsets", 0.02, "--out", out, "--append")
    assert r.returncode == 0, r.stderr
    assert len(json.loads(out.read_text())) == 2
    r = run_tool(*BASE, "--lobe", -1, 0, "--channels", 330, "--offsets", 0.01,
                 "--out", out, "--append")
    assert r.returncode != 0 and "duplicated event id" in r.stdout + r.stderr


def test_tool_prediction_at_lobe_is_no_null(tmp_path):
    r = run_tool(*BASE, "--lobe", -1, 0, "--channels", 330, "--offsets", 0,
                 "--out", tmp_path / "e.json")
    row = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("330")][0].split()
    rho2, null_db = float(row[-4]), float(row[-2])
    assert rho2 == pytest.approx(1.0) and abs(null_db) < 0.1


def test_tool_rejects_invisible_lobe(tmp_path):
    r = run_tool(*BASE, "--lobe", -1, -1, "--channels", 330, "--offsets", 0,
                 "--out", tmp_path / "e.json")
    assert r.returncode != 0 and "not visible" in r.stdout + r.stderr
