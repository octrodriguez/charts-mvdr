#!/bin/bash
#SBATCH --job-name=kotekan_corr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=kotekan_corr_%j.out

# Correlate a baseband window with the Kotekan GPU correlator (cudaCorrelatorAstron)
# and compare the result with charts_mvdr.np_correlator.
# Usage (from trig-login01, in $SCRATCH/charts-mvdr):
#   sbatch --export=ALL,KOTEKAN_SRC=$SCRATCH/kotekan-wt/integration,WINDOW_DIR=$SCRATCH/data/sim_test/quick_day_gsun \
#          jobs/correlate_kotekan.sh
# Optional: LABEL names the output directory (default: WINDOW), useful when several
# windows share the same frame prefix.
set -euo pipefail
SRC="${KOTEKAN_SRC:?set KOTEKAN_SRC to the Kotekan worktree with a built binary}"
WIN_DIR="${WINDOW_DIR:?set WINDOW_DIR to the directory with <window>_%07d.bin frames}"
WINDOW="${WINDOW:-$(basename "$WIN_DIR")}"
NUM_FREQ="${NUM_FREQ:-336}"
NUM_ANT="${NUM_ANT:-64}"
LABEL="${LABEL:-$WINDOW}"
OUT="$SCRATCH/data/kotekan_corr/${LABEL}_${SLURM_JOB_ID}"
MVDR="$SCRATCH/charts-mvdr"

module load StdEnv/2023 gcc/12.3 cuda/12.6 boost hdf5
source "$SCRATCH/charts-utils/.venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$SRC/build/external/n2k:$SRC/build/lib:$SRC/build/kotekan:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
git -C "$SRC" rev-parse --short HEAD > "$OUT/kotekan_commit.txt"
git -C "$MVDR" rev-parse --short HEAD > "$OUT/charts_mvdr_commit.txt"
N_FRAMES=$(ls "$WIN_DIR"/"${WINDOW}"_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].bin | wc -l)
echo "Window $WINDOW: $N_FRAMES frames -> $OUT"

# 1) Kotekan config (same generator as charts_sim.py)
python - "$SRC" "$WIN_DIR" "$WINDOW" "$OUT" "$N_FRAMES" "$NUM_ANT" "$NUM_FREQ" << 'PY'
import sys
from pathlib import Path
src, win_dir, window, out, n, n_ant, n_freq = sys.argv[1:]
sys.path.insert(0, str(Path(src) / "test_charts"))
from sim.pipeline import create_correlator_yaml
create_correlator_yaml(
    yaml_path=Path(out) / "correlator.yaml",
    baseband_dir=Path(win_dir), baseband_name=window,
    correlator_dir=Path(out), num_frames=int(n),
    num_elements=int(n_ant), num_local_freq=int(n_freq),
)
print("config:", Path(out) / "correlator.yaml")
PY

# 2) Run Kotekan from the repo root (kernel_path is relative)
cd "$SRC"
timeout 1200 "$SRC/build/kotekan/kotekan" -c "$OUT/correlator.yaml" -b 127.0.0.1:0 \
    > "$OUT/kotekan.log" 2>&1 || { echo "Kotekan failed; last lines:"; tail -n 40 "$OUT/kotekan.log"; exit 1; }
echo "Kotekan done: $(ls "$OUT"/corr_*.bin 2>/dev/null | wc -l) correlator dumps"

# 3) Compare with the NumPy reference correlator
python "$MVDR/scripts/04_compare_correlators.py" \
    --baseband-dir "$WIN_DIR" --window "$WINDOW" --corr-dir "$OUT" \
    --num-freq "$NUM_FREQ" --num-ant "$NUM_ANT" | tee "$OUT/comparison.txt"
