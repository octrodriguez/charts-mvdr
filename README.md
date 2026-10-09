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

Tested against `kotekan_charts` commit `db331bb7` (2026-10-03). The simulator
picks up the installed `charts_constants` package (check
`sim.constants.USING_CHARTS_CONSTANTS_PACKAGE`). Running the correlator
requires a compiled Kotekan binary (path passed with `--kotekan-bin`).

## Environment

Developed and tested on Trillium (Digital Research Alliance of Canada / SciNet).
On other systems, replace `debugjob` / `sbatch` with the local scheduler.

- Light tests and CPU-tier work on login nodes; heavier runs via `debugjob`.
- Baseband generation and Kotekan correlation on GPU nodes via Slurm.
- Long jobs are submitted with `sbatch`.
- Compute nodes have no internet access: install packages from a login node.
- Data live under `$SCRATCH` and are **not** tracked in this repository.

## Repository layout

```
scripts/
  01_inspect.py              # dataset diagnostics (format, antennas, frames)
  02_snapshot_view.py        # per-source snapshot visualization (in progress)
legacy/                      # early exploratory scripts, kept for reference
```

## Roadmap

Planned scripts:

- `03_validate_mvdr.py`: MVDR validation on simulated data with known ground truth.
- `04_saturation_analysis.py`: effect of ADC saturation on calibration and MVDR.
- `05_mvdr_real.py`: MVDR on real observations with a configurable window.
- `06_fitburst_inject.py`: synthetic FRB injection into real data.
- `07_paper_figures.py`: figures for memos and reports.

Planned shared package (`charts_mvdr/`), so that scripts do not duplicate code:

- Baseband I/O: 4-bit complex unpacking and frame reader.
- NumPy reference correlator.
- CPU-tier simulations (analytic and voltage-level).
- DFT and MVDR beamformer weights.

Longer term: port the beamformer to the real-time correlator pipeline.

## Usage

```bash
source $SCRATCH/charts-utils/.venv/bin/activate
cd $SCRATCH/charts-mvdr/scripts

# Dataset diagnostics
python 01_inspect.py --input <data_dir> --pattern "corr_*_0000000.bin" --summary

# Snapshot visualization
python 02_snapshot_view.py --input <data_dir> --source frb --mode all --compare
```

Full-tier simulation (from `$SCRATCH/kotekan`):

```bash
# Generate baseband only (no Kotekan binary needed)
python test_charts/charts_sim.py pipeline --preset quick --profile day --dry-run \
    --scratch-dir $SCRATCH/sim_test

# Correlate a saved reference window (GPU node + Kotekan binary)
python test_charts/charts_sim.py correlate --reference <tag>
```

Dataset paths are always passed as command-line arguments; no path is
hard-coded in the scripts.

## Data formats

**Correlation dumps** (Kotekan `cudaCorrelatorAstron` → `rawFileWrite`):

```python
load_gpu_visibilities(path, n_freq=336, num_antennas=64, frame_format="prefixed")
```

**Baseband frames** (`<window>_%07d.bin`): one `uint32` prefix followed by
`samples × freq × antennas` bytes, each byte a 4-bit complex sample
(real in the low nibble, imaginary in the high nibble, two's complement,
clipped to [-7, +7]). Companion files: `<window>_meta.h5` (frame indices,
timestamps, antenna positions) and `<window>_events.json` (injected events).

Raw data, intermediate products (`.npy`) and figures are excluded via
`.gitignore`.

## Conventions

- Visibilities: `R_ij[f] = sum_t v_i[t, f] * conj(v_j[t, f])`. Agreement with
  the Kotekan output convention is to be verified against the NumPy
  reference correlator.
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
CHARTS team, Universidad de Chile.
