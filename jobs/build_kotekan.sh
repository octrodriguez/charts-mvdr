#!/bin/bash
#SBATCH --job-name=kotekan_build
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=kotekan_build_%j.out

# Build Kotekan with CUDA on a Trillium GPU node.
# Usage (from trig-login01, in $SCRATCH):
#   sbatch --export=ALL,KOTEKAN_SRC=$SCRATCH/kotekan-wt/integration jobs/build_kotekan.sh
#   sbatch --export=ALL,KOTEKAN_SRC=...,BUILD_TYPE=Test jobs/build_kotekan.sh   # asserts + debug logging
# Changing BUILD_TYPE forces a full rebuild; otherwise builds are incremental.
set -euo pipefail
SRC="${KOTEKAN_SRC:?set KOTEKAN_SRC to the Kotekan worktree to build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

module load StdEnv/2023 gcc/12.3 cuda/12.6 cmake boost hdf5   # fail loudly if missing
source "$SCRATCH/charts-utils/.venv/bin/activate"              # python deps for config parsing

echo "Building $(git -C "$SRC" rev-parse --short HEAD) ($BUILD_TYPE) in $SRC on $(hostname)"
mkdir -p "$SRC/build" && cd "$SRC/build"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90" \
         -DUSE_NUMA=ON -DWERROR=OFF
make -j"$SLURM_CPUS_PER_TASK" kotekan

# CMake puts /usr/lib64 first in the RPATH, which loads the OS glibc instead of the
# CVMFS (StdEnv/2023) one. Remove system and empty entries from every ELF in the build.
find "$SRC/build" -type f \( -name "*.so*" -o -perm -u+x \) | while read -r f; do
    rp=$(patchelf --print-rpath "$f" 2>/dev/null) || continue
    [ -z "$rp" ] && continue
    new=$(echo "$rp" | tr ':' '\n' | grep -v -E '^(/usr/lib64|/lib64|)$' | paste -sd:)
    if [ "$new" != "$rp" ]; then
        patchelf --set-rpath "$new" "$f" && echo "RPATH fixed: $f"
    fi
done

export LD_LIBRARY_PATH="$SRC/build/external/n2k:$SRC/build/lib:$SRC/build/kotekan:${LD_LIBRARY_PATH:-}"
if ldd "$SRC/build/kotekan/kotekan" | grep -q "not found\|GLIBC"; then
    ldd "$SRC/build/kotekan/kotekan" | grep "not found\|GLIBC"
    exit 1
fi
"$SRC/build/kotekan/kotekan" --help > /dev/null
echo "Smoke test OK: $SRC/build/kotekan/kotekan"
