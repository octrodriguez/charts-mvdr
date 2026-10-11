#!/bin/bash
#SBATCH --job-name=mvdr_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --output=mvdr_validate_%j.out
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SCRATCH/charts-utils/.venv/bin/activate"

# Small matrices: avoid BLAS oversubscription (4 threads per sweep, 4 sweeps)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg

OUT="$SCRATCH/mvdr_validation/full_${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse --short HEAD > "$OUT/commit.txt"

for sw in position inr k loading; do
    python scripts/03_validate_mvdr.py --sweep "$sw" --out "$OUT" > "$OUT/$sw.log" 2>&1 &
done
wait
echo "All sweeps finished -> $OUT"
