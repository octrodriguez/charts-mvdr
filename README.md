# charts-mvdr

General-purpose pipeline for spatial RFI mitigation (MVDR beamforming) and
calibration on CHARTS (Canadian–Chilean Array for Radio Transient Studies)
data, validated on simulations at two levels of fidelity.

> **Status:** work in progress. Methods are validated first on simulated data
> with known ground truth, then applied to real observations.

## Scope

- Spatial RFI mitigation with an MVDR (Capon) beamformer, following the
  two-stage architecture of Argyriou & Andrew (2025).
- Point-source gain calibration (rank-1 WALS) via `charts_utils.cal.solve_gain`,
  as described in the CHARTS point-source calibration memo.
- Diagnostic and visualization tools for Kotekan GPU correlation dumps and
  baseband frames.
- Two simulation tiers (see below). Every analysis script consumes a covariance
  cube `R[freq, ant, ant]`, regardless of whether it comes from a real
  observation, a CPU simulation or the Kotekan GPU correlator.

## Simulation tiers

| Tier | Runs on | Pipeline | Purpose |
|------|---------|----------|---------|
| **CPU (light)** | login / `debugjob` | Analytic covariance `R = sum_k p_k a_k a_k^H + N`, or voltage-level simulation (complex Gaussian sources, optional 4-bit quantization) followed by a NumPy reference correlator | Unit tests of the beamformer, finite-sample effects, diagonal loading, saturation studies |
| **Kotekan (full)** | GPU nodes | `test_charts/charts_sim.py` (baseband generator) → Kotekan replay → GPU tensor-core correlator | End-to-end validation in the production data format |

The NumPy reference correlator is also used to cross-check Kotekan output:
both compute integer sums, so they must agree exactly. This pins down the
baseline indexing and conjugation conventions.

## Dependencies

### charts-utils (required)

[`charts-utils`](https://github.com/charts-experiment/charts-utils) is
**not** included here. Install it in editable mode:

```bash
git clone https://github.com/charts-experiment/charts-utils.git
cd charts-utils
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Tested against `charts-utils` commit `ba2aa16` (2026-10-06).
`charts_constants` is resolved from site-packages (pip install from GitHub).

### Kotekan (optional, full tier)

```bash
git clone -b kotekan_charts https://github.com/charts-experiment/kotekan.git $SCRATCH/kotekan
source $SCRATCH/charts-utils/.venv/bin/activate
pip install -r $SCRATCH/kotekan/test_charts/requirements.txt
```

Generating baseband needs only Python. Correlating requires a compiled Kotekan
binary, built on a GPU node with `jobs/build_kotekan.sh` (see Usage). On
Trillium the build removes `/usr/lib64` from the RPATH of the binaries: CMake
puts it first, which loads the OS glibc instead of the StdEnv/2023 one and the
binary fails with `GLIBC_2.35 not found`.

Reference commit: `kotekan_charts` at `d88295e4` (Release build). With it the
GPU correlator reproduces `np_correlator` exactly on a 99-frame simulated window.

## Installation

Install this repository as an editable package in the same environment as
`charts-utils`. Run this on a login node: the build step needs internet access.

```bash
source $SCRATCH/charts-utils/.venv/bin/activate
cd $SCRATCH/charts-mvdr
pip install -e .
```

The package is then importable from any directory as `charts_mvdr`.

## Environment

Developed and tested on Trillium (Digital Research Alliance of Canada / SciNet).
On other systems, replace `debugjob` / `sbatch` with the local scheduler.

- Light tests and CPU-tier work on login nodes; heavier runs via `debugjob`.
- Kotekan is built and run on GPU nodes (submit from `trig-login01`).
- Compute nodes have no internet access: install packages from a login node.
- Data live under `$SCRATCH` and are **not** tracked in this repository.
- **Always set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS`** (e.g. 4). Without
  them NumPy starts one BLAS thread per core (192 on a Trillium node) for every
  small matrix operation, which makes the scripts orders of magnitude slower.

## Repository layout

```
pyproject.toml
charts_mvdr/                    # importable package (pip install -e .)
  data_io.py                    # baseband frames, window metadata, events, frame selection, correlation dumps
  np_correlator.py              # NumPy (CPU) reference correlator
  beamforming.py                # steering vectors, DFT/MVDR weights, beam response, grating lobes, (l, m) grid
  checks.py                     # pre-flight checks of MVDR assumptions
  quantization.py               # 4-bit saturation model (Bussgang gain, input sigma)
  ghosts.py                     # intermodulation ghosts of 4-bit clipping: levels (Gaussian model), directions, beam hits
  transients.py                 # dispersion delays, arrival times, on-pulse masks
  fringes.py                    # coherence with a geometric fringe model: baseline fit, channel classification
  stage_reference.py            # NumPy reference of the Kotekan MVDR stage (blind local covariance window, gains)
  sim_cpu.py                    # CPU tier: analytic covariance, voltage simulation
scripts/
  00_check_assumptions.py       # runs all checks on a frame, summary per check
  01_inspect.py                 # dataset diagnostics (format, antennas, frames)
  02_snapshot_view.py           # per-source snapshot visualization (in progress)
  03_validate_mvdr.py           # SINR sweeps on CPU-tier data with ground truth
  04_compare_correlators.py     # Kotekan GPU correlator vs np_correlator
  05_mvdr_correlations.py       # DFT localization (refined) + MVDR on correlation dumps, power budget
  06_saturation_analysis.py     # 4-bit saturation per channel vs Bussgang prediction
  07_beam_response.py           # DFT vs MVDR beam response |w^H a(l, m)|^2 maps from correlation dumps
  08_transient_snr.py           # dispersed burst: S/N through DFT and MVDR (on, all, off-pulse covariances)
  09_beamform_waterfall.py      # voltage-domain DFT vs KL beamforming: waterfalls, per-channel and dedispersed S/N
  10_requantization_sweep.py    # CPU tier: per-channel gain before 4-bit requantization (flux gain, ghosts, S/N)
  11_fringe_coherence.py        # real frames: baseline fit per band and source/conjugate/stationary channel classes
  12_stage_reference.py         # reference of the Kotekan MVDR stage: beam power per frame/channel/beam, weights
  13_compare_kotekan_mvdr.py    # Kotekan chartsMVDRBeams output vs the reference on the same frames
jobs/
  validate_mvdr.sh              # Slurm: 03_validate_mvdr sweeps (CPU node)
  build_kotekan.sh              # Slurm: build Kotekan with CUDA (GPU node)
  correlate_kotekan.sh          # Slurm: Kotekan correlator on a window + comparison
  analyze_window.sh             # Slurm: 05 and 06 on a correlated window (runs from a text file)
  mvdr_kotekan.sh               # Slurm: correlator + chartsMVDRBeams on a window, comparison with 13
tests/                          # pytest suite (no external data needed)
tools/
  export_md.py                  # export tracked source files as one Markdown bundle
  export_github_md.py           # export GitHub metadata (PRs, issues, commits)
  diagnose_astron_layout.py     # layout/convention diagnosis of correlator dumps
  make_grating_lobe_events.py   # events file with interferers near a grating lobe (+ MVDR prediction)
  make_frb_events.py            # events file with a dispersed FRB + RFI (capture and off-pulse windows, frame estimate vs window)
  compare_paired.py             # paired windows (with / without RFI): S/N relative to the clean DFT by RFI group
  fig_beam_response.py          # composite beam-response figure (maps; null depth and WNG vs distance to a grating lobe, ideal model)
legacy/                         # early exploratory scripts, kept for reference
```

## Roadmap

Planned scripts:

- `10_mvdr_real.py`: MVDR on real observations with a configurable window.
- `11_fitburst_inject.py`: synthetic FRB injection into real data.
- `12_paper_figures.py`: figures for memos and reports.

Longer term: port the beamformer to the real-time correlator pipeline.

## Usage

```bash
source $SCRATCH/charts-utils/.venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $SCRATCH/charts-mvdr

python -m pytest -q tests/                                         # unit tests
python scripts/00_check_assumptions.py <frame.bin>                 # MVDR pre-flight checks
python scripts/00_check_assumptions.py <frame.bin> --multi-frame    # + #9 stationarity over the window
python scripts/00_check_assumptions.py <frame.bin> --look L M --on T0 T1 --off T0 T1   # + #10
python scripts/00_check_assumptions.py <frame.bin> --look L M --multi-frame          # + #11 on the window mean
python scripts/01_inspect.py --input <data_dir> --pattern "corr_*_0000000.bin" --summary
sbatch jobs/validate_mvdr.sh                                       # SINR sweeps
```

Full tier (Kotekan):

```bash
# Generate a baseband window (CPU, from a Kotekan worktree)
python test_charts/charts_sim.py generate --duration-s 5 --antennas 64 --num-freq 336 \
    --profile day --start-time 2026-03-20T15:00:00 --scratch-dir $SCRATCH/data/sim_test/<name>

# Scenario with interferers at increasing offsets from a grating lobe of the look
# direction (one channel each); pass the file to the generator with --events-json
python tools/make_grating_lobe_events.py --look L M --spacing DX DY --lobe P Q \
    --channels <ch ...> --offsets <offset ...> --amp 4 --out <events.json>

# Paired windows (same seed and schedule, RFI amplitude 0 in one): run 09 (or 08) on
# both, then compare per RFI group; exits with status 2 if the windows are not paired
python tools/compare_paired.py --rfi <rfi_out> --clean <clean_out> --window-dir <rfi_window_dir> \
    [--group NAME=id1,id2 ...] [--csv paired.csv]

# Composite beam-response figure from 07 outputs (normal case, grating-lobe case, sweep)
python tools/fig_beam_response.py --normal <normal_out>/beam_response.npz:<ch> \
    --grating <grating_out>/beam_response.npz:<ch> --events <grating_events.json> --out <fig.png>

# Per-channel digital gain before the 4-bit requantization (CPU tier, ~2 min):
# flux-scale gain, k = -3 ghost level and S/N of a beam with and without the ghost,
# Gaussian vs tone emitter, as a function of G sigma_in
python scripts/10_requantization_sweep.py --out <out_dir>

# Real GPU correlation frames: coherence with the Sun's fringe model (no gain calibration needed).
# Baseline vector fitted per band vs the layout, its negative (conjugated data) and zero
# (stationary signal); per-channel classes saved to <out_dir>/fringe_coherence.npz
python scripts/11_fringe_coherence.py --input <frame_dir> --antennas <a,b,c> --positions <positions.npy> \
    --time-start <ISO UTC> --time-stop <ISO UTC> --out <out_dir> [--bands 300-410 431-501.6] [--exclude 360-381.5]

# Reference of the Kotekan MVDR stage on a correlated window: covariance from frames
# guard < |k - t| <= half-window around each frame, DFT and MVDR (loading 10 x noise floor)
# beam power per frame, channel and beam, weights at each update (npz)
python scripts/12_stage_reference.py --corr-dir <corr_dir> --window-dir <window_dir> \
    --beams <l1> <m1> [<l2> <m2> ...] --out <out.npz> [--half-window 10 --guard 1 --gains <gains.npy>]

# Kotekan MVDR stage on a window and comparison with the reference (GPU, from trig-login01);
# outputs in $SCRATCH/data/kotekan_mvdr/<label>_<jobid>/{corr,mvdr}, verdict in comparison.txt
sbatch --export=ALL,KOTEKAN_SRC=<kotekan_worktree>,WINDOW_DIR=<window_dir>,BEAMS="<l1> <m1> ...",LABEL=<label> \
    jobs/mvdr_kotekan.sh

# Build Kotekan and correlate a window (GPU, from trig-login01)
sbatch --export=ALL,KOTEKAN_SRC=<kotekan_worktree> jobs/build_kotekan.sh
sbatch --export=ALL,KOTEKAN_SRC=<kotekan_worktree>,WINDOW_DIR=<window_dir> jobs/correlate_kotekan.sh

# MVDR and saturation analysis of the correlated window (CPU; run list in a text file)
sbatch --export=ALL,WINDOW_DIR=<window_dir>,CORR_DIR=<dump_dir>,OUT_DIR=<out_dir>,LOOK="L M",RUNS_FILE=<runs.txt> \
    jobs/analyze_window.sh
# Interferers near a grating lobe: look-source power from the clean channels
#   (add EXTRA_05="--p-look-ref clean" to the --export list)
```

Dataset paths are always passed as command-line arguments; no path is
hard-coded in the scripts.

Library usage:

```python
from charts_mvdr.data_io import read_baseband_frame, read_astron_correlation
from charts_mvdr.np_correlator import correlate

V = read_baseband_frame(path, num_freq=336, num_ant=64)       # (time, freq, antenna)
R = correlate(V, normalize=True)                               # (freq, antenna, antenna)
Rk = read_astron_correlation(corr_path, num_freq=336, num_ant=64)  # same convention, sums
```

## Data formats

**Baseband frames** (`<window>_%07d.bin`): one `uint32` prefix followed by
`samples × freq × antennas` bytes, each byte a 4-bit complex sample (real in the
low nibble, imaginary in the high nibble, two's complement). The F-engine
produces values in [-7, +7]. Companion files: `<window>_meta.h5` (frequencies,
antenna positions) and `<window>_events.json` (injected events).

**Correlation dumps** (Kotekan `cudaCorrelatorAstron` → `rawFileWrite`,
`corr_%07d.bin`): `uint32` metadata size, metadata, then `int32` (re, im) pairs
with shape `(freq, blocks, 2, 2, 2)`. Blocks of 2×2 elements cover the lower
block triangle in row-major order. Read with `data_io.read_astron_correlation`.

Raw data, intermediate products (`.npy`), Slurm logs and generated bundles are
excluded via `.gitignore`.

## Conventions

- Visibilities: `R_ij[f] = sum_t v_i[t, f] * conj(v_j[t, f])` (`np_correlator`).
- The Kotekan correlator stores the conjugate, `sum_t conj(v_i) v_j`;
  `read_astron_correlation` conjugates, so both sources share one convention.
  Verified with `04_compare_correlators.py`: exact agreement for data in
  [-7, +7].
- Samples equal to -8 (outside the F-engine range) are **not** correlated
  exactly: the kernel negates one operand in 4-bit arithmetic, so -(-8) wraps
  to -8 and the imaginary part of the affected cross-correlations is wrong
  (always by multiples of 16). Simulated data must be clipped to [-7, +7].
- Steering vector: `a_n = exp(-j 2 pi f (l x_n + m y_n) / c)`.
- Voltages in ADC units (LSB).
- Baseband frame: 1536 samples × 10/3 µs = 5.12 ms. One simulated correlation
  dump per frame; the production correlator integrates 10.24 s.
- Frequency grid: 0.3 MHz channels starting at 300 MHz; each dump holds 336
  of the 672 channels.
- Array geometry is taken from `charts_constants` or from the dataset metadata;
  no layout is hard-coded in the scripts.

## References

- Argyriou & Andrew (2025), *ApJS*.
- Pollarolo & Cassanelli, CHARTS internal memo on point-source calibration.
- CHARTS Simulation Framework in Kotekan (internal presentation).

## Author

Octavio Rodríguez ([@octrodriguez](https://github.com/octrodriguez)) —
CHARTS team, AstroLab.