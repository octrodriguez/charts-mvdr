#!/bin/bash
#SBATCH --job-name=mvdr_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=mvdr_analyze_%j.out

# Run 05_mvdr_correlations (loadings 0 and 10 sigma_n^2) and 06_saturation_analysis
# on a window correlated with Kotekan, for a list of runs given in a text file.
# Usage (in $SCRATCH/charts-mvdr; sbatch, or bash inside a debugjob):
#   sbatch --export=ALL,WINDOW_DIR=<window_dir>,CORR_DIR=<dump_dir>,OUT_DIR=<out_dir>,\
#          LOOK="L M",RUNS_FILE=<runs.txt> jobs/analyze_window.sh
#
# RUNS_FILE: one run per line, "name | channels | t0 t1 | l m"; the time range
# (frame timestamps, s) and the look direction are optional (empty = whole window
# and LOOK). Lines starting with # are ignored. Script 06 runs once per distinct
# time range, with LOOK. Example:
#   persistent | 94 95 133 147 |         |
#   transient  | 40 41 192     | 4.02 5.0 |
#   gridlook   | 94 95         |         | 0.463 0.475
set -euo pipefail
: "${WINDOW_DIR:?set WINDOW_DIR}" "${CORR_DIR:?set CORR_DIR}" "${OUT_DIR:?set OUT_DIR}"
: "${LOOK:?set LOOK to the look direction, e.g. LOOK=\"0.468 0.480\"}"
: "${RUNS_FILE:?set RUNS_FILE}"
LOADINGS="${LOADINGS:-0 10}"
MVDR="${MVDR:-$SCRATCH/charts-mvdr}"
RUNS_FILE=$(realpath "$RUNS_FILE"); OUT_DIR=$(realpath -m "$OUT_DIR")

source "$SCRATCH/charts-utils/.venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$MVDR"
mkdir -p "$OUT_DIR"
git rev-parse --short HEAD > "$OUT_DIR/charts_mvdr_commit.txt"
cp "$RUNS_FILE" "$OUT_DIR/runs.txt"

trim() { local s="$*"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
declare -A TRANGES=()

while IFS='|' read -r name chans trange look || [ -n "$name" ]; do
  name=$(trim "$name"); chans=$(trim "$chans"); trange=$(trim "${trange:-}"); look=$(trim "${look:-}")
  [ -z "$name" ] || [[ "$name" == \#* ]] && continue
  look="${look:-$LOOK}"
  targs=(); [ -n "$trange" ] && targs=(--t-range $trange)
  TRANGES["${trange:-full}"]=1
  for d in $LOADINGS; do
    tag="05_${name}_d${d}"
    echo "== $tag"
    python scripts/05_mvdr_correlations.py --corr-dir "$CORR_DIR" --window-dir "$WINDOW_DIR" \
        --channels $chans --look $look "${targs[@]}" --loading "$d" --loading-ref noise \
        --out "$OUT_DIR/$tag" > "$OUT_DIR/$tag.txt" 2>&1 || echo "   FAILED (see $OUT_DIR/$tag.txt)"
  done
done < "$RUNS_FILE"

for trange in "${!TRANGES[@]}"; do
  targs=(); [ "$trange" != full ] && targs=(--t-range $trange)
  tag="06_${trange// /-}"
  echo "== $tag"
  python scripts/06_saturation_analysis.py --window-dir "$WINDOW_DIR" --corr-dir "$CORR_DIR" \
      --look $LOOK "${targs[@]}" --out "$OUT_DIR/$tag" > "$OUT_DIR/$tag.txt" 2>&1 \
      || echo "   FAILED (see $OUT_DIR/$tag.txt)"
done
echo "outputs -> $OUT_DIR"
