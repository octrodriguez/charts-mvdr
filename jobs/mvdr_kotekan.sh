#!/bin/bash
#SBATCH --job-name=kotekan_mvdr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=kotekan_mvdr_%j.out

# Replay a baseband window through the Kotekan correlator and the chartsMVDRBeams stage, then
# compare the stage output with the NumPy reference (scripts/13_compare_kotekan_mvdr.py).
# Usage (from trig-login01, in $SCRATCH/charts-mvdr):
#   sbatch --export=ALL,KOTEKAN_SRC=$SCRATCH/kotekan-wt/feat-mvdr-stage,\
#   WINDOW_DIR=$SCRATCH/data/sim_test/clean_v2/charts_sim_64ant,BEAMS="0.4677 0.4801",LABEL=clean_v2 \
#          jobs/mvdr_kotekan.sh
# Optional: H (10), GUARD (1), UPDATE_EVERY (1), LOADING (10), LOADING_REF (noise), GAINS (.npy).
set -euo pipefail
SRC="${KOTEKAN_SRC:?set KOTEKAN_SRC to the Kotekan worktree with a built binary}"
WIN_DIR="${WINDOW_DIR:?set WINDOW_DIR to the directory with <window>_%07d.bin frames}"
BEAMS="${BEAMS:?set BEAMS to look directions, e.g. \"0.4677 0.4801 -0.10 0.25\"}"
WINDOW="${WINDOW:-$(basename "$WIN_DIR")}"
NUM_FREQ="${NUM_FREQ:-336}"
NUM_ANT="${NUM_ANT:-64}"
H="${H:-10}"
GUARD="${GUARD:-1}"
UPDATE_EVERY="${UPDATE_EVERY:-1}"
LOADING="${LOADING:-10}"
LOADING_REF="${LOADING_REF:-noise}"
GAINS="${GAINS:-}"
LABEL="${LABEL:-$WINDOW}"
OUT="$SCRATCH/data/kotekan_mvdr/${LABEL}_${SLURM_JOB_ID}"
MVDR="$SCRATCH/charts-mvdr"

module load StdEnv/2023 gcc/12.3 cuda/12.6 boost hdf5
source "$SCRATCH/charts-utils/.venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$SRC/build/external/n2k:$SRC/build/lib:$SRC/build/kotekan:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
git -C "$SRC" rev-parse --short HEAD > "$OUT/kotekan_commit.txt"
git -C "$MVDR" rev-parse --short HEAD > "$OUT/charts_mvdr_commit.txt"
N_FRAMES=$(ls "$WIN_DIR"/"${WINDOW}"_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].bin | wc -l)
echo "Window $WINDOW: $N_FRAMES frames, beams $BEAMS, H=$H g=$GUARD U=$UPDATE_EVERY -> $OUT"

# 1) Kotekan config: positions and channel frequencies from the window metadata
python - "$SRC" "$WIN_DIR" "$WINDOW" "$OUT" "$N_FRAMES" "$NUM_ANT" "$NUM_FREQ" "$BEAMS" \
    "$H" "$GUARD" "$UPDATE_EVERY" "$LOADING" "$LOADING_REF" "$GAINS" << 'PY'
import sys
from pathlib import Path
import numpy as np
src, win_dir, window, out, n, n_ant, n_freq, beams, h, g, u, loading, lref, gains = sys.argv[1:]
sys.path.insert(0, str(Path(src) / "test_charts"))
from sim.mvdr import create_mvdr_yaml
from charts_mvdr.data_io import load_window_dir_meta
meta = load_window_dir_meta(win_dir)
f = np.asarray(meta["frequencies_mhz"], dtype=float)[: int(n_freq)]
df = float(f[1] - f[0])
if np.max(np.abs(np.diff(f) - df)) > 1e-6:
    raise SystemExit("channel frequencies are not uniformly spaced")
b = [float(v) for v in beams.split()]
create_mvdr_yaml(
    yaml_path=Path(out) / "mvdr.yaml", baseband_dir=Path(win_dir), baseband_name=window,
    output_dir=Path(out), num_frames=int(n),
    positions_x_m=meta["antenna_pos_x_m"], positions_y_m=meta["antenna_pos_y_m"],
    beams=list(zip(b[0::2], b[1::2])), freq_start_mhz=float(f[0]), delta_freq_mhz=df,
    num_elements=int(n_ant), num_local_freq=int(n_freq), cov_half_window=int(h),
    cov_guard=int(g), update_every=int(u), loading=float(loading), loading_ref=lref,
    gains_file=Path(gains) if gains else None,
)
print("config:", Path(out) / "mvdr.yaml", f"f0 {f[0]} MHz, df {df} MHz")
PY

# 2) Run Kotekan from the repo root (kernel_path is relative)
cd "$SRC"
timeout 1800 "$SRC/build/kotekan/kotekan" -c "$OUT/mvdr.yaml" -b 127.0.0.1:0 \
    > "$OUT/kotekan.log" 2>&1 || { echo "Kotekan failed; last lines:"; tail -n 40 "$OUT/kotekan.log"; exit 1; }
echo "Kotekan done: $(ls "$OUT"/corr/corr_*.bin 2>/dev/null | wc -l) correlation frames," \
     "$(ls "$OUT"/mvdr/mvdr_power_*.bin 2>/dev/null | wc -l) power frames," \
     "$(ls "$OUT"/mvdr/mvdr_weights_*.bin 2>/dev/null | wc -l) weight updates"
grep -n "WNG above\|not positive definite\|Flushed" "$OUT/kotekan.log" | head -n 20 || true

# 3) Compare with the NumPy reference on the same correlation frames
GAINS_ARG=()
[ -n "$GAINS" ] && GAINS_ARG=(--gains "$GAINS")
python "$MVDR/scripts/13_compare_kotekan_mvdr.py" --run-dir "$OUT" --window-dir "$WIN_DIR" \
    --beams $BEAMS --num-freq "$NUM_FREQ" --num-ant "$NUM_ANT" --half-window "$H" \
    --guard "$GUARD" --update-every "$UPDATE_EVERY" --loading "$LOADING" \
    --loading-ref "$LOADING_REF" "${GAINS_ARG[@]}" --out "$OUT/stage_output.npz" \
    | tee "$OUT/comparison.txt"
