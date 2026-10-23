#!/bin/bash
#SBATCH --job-name=tracker_gains
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=tracker_gains_%j.out

# Calibration proof of concept for cudaDirectBeamTrackerCommand: known quarter-turn gains are
# applied to a simulated window (tools/apply_test_gains.py), the tracker runs on
#   ref:   clean window, no gains file
#   cal:   gained window, gains file (charts-utils format, v_obs = g v)
#   nocal: gained window, no gains file
# and tools/compare_tracker_beams.py checks cal = ref / A (A = amplitude in the file only).
# Usage (from trig-login01, in $SCRATCH/charts-mvdr):
#   sbatch --export=ALL,KOTEKAN_SRC=$SCRATCH/kotekan-wt/feat-mvdr-stage,\
#   WINDOW_DIR=$SCRATCH/data/sim_test/clean_v2/charts_sim_64ant jobs/tracker_gains_poc.sh
# Optional: N_FRAMES (20), BEAMS ("0.4677 0.4801 -0.10 0.25"), AMPLITUDE (2.0), SEED (1).
set -euo pipefail
SRC="${KOTEKAN_SRC:?set KOTEKAN_SRC to the Kotekan worktree with a built binary}"
WIN_DIR="${WINDOW_DIR:?set WINDOW_DIR to the directory with <window>_%07d.bin frames}"
N_FRAMES="${N_FRAMES:-20}"
BEAMS="${BEAMS:-0.4677 0.4801 -0.10 0.25}"
AMPLITUDE="${AMPLITUDE:-2.0}"
SEED="${SEED:-1}"
OUT="$SCRATCH/data/tracker_gains/$(basename "$(dirname "$WIN_DIR")")_${SLURM_JOB_ID}"
MVDR="$SCRATCH/charts-mvdr"

module load StdEnv/2023 gcc/12.3 cuda/12.6 boost hdf5
source "$SCRATCH/charts-utils/.venv/bin/activate"
export LD_LIBRARY_PATH="$SRC/build/external/n2k:$SRC/build/lib:$SRC/build/kotekan:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
git -C "$SRC" rev-parse --short HEAD > "$OUT/kotekan_commit.txt"
git -C "$MVDR" rev-parse --short HEAD > "$OUT/charts_mvdr_commit.txt"

# 1) gained window and gains file
python "$MVDR/tools/apply_test_gains.py" --window-dir "$WIN_DIR" --out-root "$OUT" \
    --num-frames "$N_FRAMES" --amplitude "$AMPLITUDE" --seed "$SEED"
WINDOW=$(ls "$OUT/clean" | sed -n 's/_[0-9]\{7\}\.bin$//p' | head -n 1)

# 2) three configs (same tracker settings, only input and gains differ)
python - "$SRC" "$OUT" "$WINDOW" "$N_FRAMES" "$BEAMS" << 'PY'
import sys
from pathlib import Path
src, out, window, n, beams = sys.argv[1:]
sys.path.insert(0, str(Path(src) / "test_charts"))
from sim.tracker_gains import create_tracker_gains_yaml
out = Path(out)
b = [float(v) for v in beams.split()]
beams = list(zip(b[0::2], b[1::2]))
for run, win, gains in (("ref", "clean", None), ("cal", "gained", out / "gains_test.npy"),
                        ("nocal", "gained", None)):
    create_tracker_gains_yaml(out / f"{run}.yaml", out / win, window, out / run, "beams",
                              int(n), beams, gains_file=gains,
                              gains_first_channel=0 if gains else None)
print("configs:", [str(out / f"{r}.yaml") for r in ("ref", "cal", "nocal")])
PY

# 3) run the tracker three times from the repo root (kernel_path is relative)
cd "$SRC"
for run in ref cal nocal; do
    timeout 900 "$SRC/build/kotekan/kotekan" -c "$OUT/$run.yaml" -b 127.0.0.1:0 \
        > "$OUT/$run.log" 2>&1 || { echo "Kotekan $run failed:"; tail -n 40 "$OUT/$run.log"; exit 1; }
    echo "$run: $(ls "$OUT/$run"/beams_*.bin | wc -l) beam frames"
done
grep -h "calibration gains" "$OUT/cal.log" || echo "WARNING: no calibration message in cal.log"

# 4) compare
EXPECTED=$(python -c "print(1.0 / $AMPLITUDE)")
echo "== cal vs ref (expected alpha = 1/A = $EXPECTED)"
python "$MVDR/tools/compare_tracker_beams.py" --ref "$OUT/ref" --test "$OUT/cal" \
    --expected-scale "$EXPECTED" | tee "$OUT/comparison.txt"
echo "== nocal vs ref (no calibration: decorrelated, residual near 1)"
python "$MVDR/tools/compare_tracker_beams.py" --ref "$OUT/ref" --test "$OUT/nocal" \
    | tee -a "$OUT/comparison.txt"
