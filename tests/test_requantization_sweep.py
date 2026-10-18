"""End-to-end test of scripts/10_requantization_sweep.py on a short run."""

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "10_requantization_sweep.py"


def test_sweep_end_to_end(tmp_path):
    out = tmp_path / "rs"
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), "--samples", str(2**15),
                    "--rfi-amp", "7", "--targets", "asis", "3.0", "--models", "gaussian", "tone"],
                   check=True, capture_output=True, text=True)
    assert (out / "requantization_sweep.png").exists()
    with open(out / "requantization_sweep.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    g = {r["target"]: r for r in rows if r["model"] == "gaussian"}
    asis, low = g["asis"], g["3"]
    # Gaussian model reproduces the simulation (ghost level and flux-scale gain)
    assert abs(float(asis["ghost_db"]) - float(asis["ghost_pred_db"])) < 1.0
    assert abs(float(asis["g2_clear"]) - float(asis["g2_sky_pred"])) < 0.03
    # correlated clipping passes more sky than independent clipping
    assert float(asis["g2_clear"]) > float(asis["g2_bussgang"])
    # lowering the level reduces the ghost and its S/N loss in the beam that receives it
    assert float(low["ghost_db"]) < float(asis["ghost_db"]) - 8.0
    assert float(low["snr_ghost_kl_db"]) > float(asis["snr_ghost_kl_db"]) + 3.0
    assert float(g["3"]["gain"]) < 1.0 and float(asis["gain"]) == 1.0
