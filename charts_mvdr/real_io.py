"""Real-data products at the boundary with charts-utils: visibility cubes and gains files.

charts-utils changed its conventions with gains ``format_version`` 2:

* version 1 (up to 2026-11-06): the readers returned the values stored by the GPU correlator
  as V_ab, and the gains were solved on them. In the raw-voltage convention of the Kotekan
  stages (R = v v^H), V = conj(R) and the file gains are conj(g_v).
* version 2: the readers conjugate the stored values, V_ab = v_a conj(v_b) with
  v ~ exp(-j k r.s), and the gains are the voltage gains, V_ab = g_a M_ab conj(g_b).

A version-2 cube is the conjugate of a version-1 cube of the same frames, and a version-2
gains array is the conjugate of the version-1 one. Dividing a cube by g_a conj(g_b) is
valid within one version only; mixing versions doubles the gain phase error.

charts-mvdr keeps version 1 internally (charts_mvdr.nulling, charts_mvdr.tracking and the
scripts on real cubes: conjugated steering, sign = -1). This module converts at the
boundary, so both generations of files can be used and mixed:

* `loader_convention`: the version of the cubes that the installed charts-utils returns.
* `read_cube`: a cube saved by 16_rfi_census.py extract, converted to the internal version
  (the cube carries `vis_convention`; cubes without it are version 1).
* `read_gains`: a charts-utils gain file (.npz, version from `format_version`) or a bare
  .npy array (version from its sidecar JSON, else it must be given), converted to the
  internal version.
* `add_gains_arguments` / `gains_from_args`: the command-line options shared by the
  scripts and the selection of the gain columns of a list of antennas.

Gains files for the Kotekan loaders (chartsMvdr::load_gains_npy conjugates on load) are
version-1 arrays; tools/kotekan_gains.py writes them with `format_version` 1 in the sidecar.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INTERNAL_CONVENTION = 1
CONVENTIONS = (1, 2)


def loader_convention(io_module=None) -> int:
    """Version of the cubes returned by the charts-utils readers: 2 if they conjugate the
    stored GPU values (charts_utils.core.io.corr_from_astron_storage exists), else 1."""
    if io_module is None:
        from charts_utils.core import io as io_module
    return 2 if hasattr(io_module, "corr_from_astron_storage") else 1


def to_internal(values, convention: int):
    """Cube or gains array of the given version in the internal version."""
    if int(convention) not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
    values = np.asarray(values)
    return values if int(convention) == INTERNAL_CONVENTION else np.conj(values)


def read_cube(path):
    """Cube of 16_rfi_census.py extract as a dict: vis (M, M, n_freq, n_frames) in the
    internal version, freqs [MHz], times, ants (list), look or None, convention (as stored)."""
    z = np.load(path)
    conv = int(z["vis_convention"]) if "vis_convention" in z.files else 1
    return {"vis": to_internal(z["vis"], conv), "freqs": np.asarray(z["freqs"], float),
            "times": np.asarray(z["times"], float), "ants": [int(a) for a in z["ants"]],
            "look": np.asarray(z["look"], float) if "look" in z.files else None,
            "convention": conv}


@dataclass
class GainsFile:
    gains: np.ndarray            # (n_freq, n_cols) complex128, internal version
    antennas: list | None        # logical antenna of each column (None: not stored)
    interpolated: np.ndarray     # bool (n_freq, n_cols): filled in by the calibration
    convention: int              # version of the file
    key: str                     # array read from a .npz


def _sidecar_version(path: Path):
    side = path.with_suffix(".json")
    if not side.exists():
        return None
    try:
        v = json.loads(side.read_text()).get("format_version")
    except (OSError, ValueError):
        return None
    return None if v is None else int(v)


def read_gains(path, fmt=None, key: str = "gains") -> GainsFile:
    """Read a gains file and return it in the internal version.

    .npz (charts-utils gain file): array `key` ("gains": cleaned and interpolated, the one
    charts-utils applies; "gains_solved": raw solution, NaN where flagged), version from
    `format_version` (1 when missing), antennas from the file.
    .npy (bare array): version `fmt`, else `format_version` of the sidecar JSON next to it;
    without either a ValueError explains what to pass, because the two versions cannot be
    told apart from the data. `fmt` overrides the stored version of either kind of file.
    Time-resolved arrays (n_freq, n_ant, n_time) are rejected.
    """
    path = Path(path)
    if fmt is not None and int(fmt) not in CONVENTIONS:
        raise ValueError(f"gains format must be one of {CONVENTIONS}, got {fmt!r}")
    antennas, interp = None, None
    if path.suffix == ".npz":
        z = np.load(path)
        if key not in z.files:
            raise ValueError(f"{path}: no array '{key}' (has {sorted(z.files)})")
        g = np.asarray(z[key])
        stored = int(z["format_version"]) if "format_version" in z.files else 1
        if "antennas" in z.files:
            antennas = [int(a) for a in np.atleast_1d(z["antennas"])]
        if key == "gains" and "interpolated" in z.files:
            interp = np.asarray(z["interpolated"], dtype=bool)
    else:
        g = np.load(path)
        stored = _sidecar_version(path)
    conv = int(fmt) if fmt is not None else stored
    if conv is None:
        raise ValueError(
            f"{path}: a bare .npy carries no gains format version. Pass --gains-format 1 for "
            "arrays solved with charts-utils before format 2 (and for Kotekan gains files), "
            "--gains-format 2 for arrays written by charts-utils with format 2 "
            "(voltage gains), or use the .npz gain file.")
    if not np.iscomplexobj(g) or g.ndim != 2:
        raise ValueError(f"{path}: expected a complex (n_freq, n_ant) array, got {g.dtype} {g.shape}")
    if antennas is not None and len(antennas) != g.shape[1]:
        raise ValueError(f"{path}: {len(antennas)} antennas listed for {g.shape[1]} columns")
    g = to_internal(g.astype(np.complex128), conv)
    if interp is None or interp.shape != g.shape:
        interp = np.zeros(g.shape, dtype=bool)
    return GainsFile(g, antennas, interp, conv, key)


def add_gains_arguments(ap, required: bool = True, antennas_required: bool = False):
    """--gains, --gain-antennas, --gains-format, --gains-key on an argparse parser."""
    ap.add_argument("--gains", required=required, default=None,
                    help="Gains: charts-utils gain file (.npz) or a bare .npy (n_freq, n_cols).")
    ap.add_argument("--gain-antennas", required=antennas_required, default=None,
                    help="Logical antennas, e.g. 0,4,32,36: the columns of a .npy, or the "
                         "antennas to take from a .npz (default: all those in the file).")
    ap.add_argument("--gains-format", type=int, choices=CONVENTIONS, default=None,
                    help="Gains format version of a bare .npy without sidecar: 1 (before "
                         "charts-utils format 2; Kotekan gains files) or 2 (voltage gains).")
    ap.add_argument("--gains-key", default="gains", choices=("gains", "gains_solved"),
                    help=".npz only: the cleaned, interpolated gains (default) or the raw solution.")
    return ap


def gains_from_args(args, n_freq: int | None = None):
    """(gains (n_freq, n) internal version, antennas, interpolated mask) for parsed options.

    The antennas are --gain-antennas, else those stored in the file. With a .npz the
    columns are matched by antenna; with a .npy --gain-antennas names every column.
    Raises ValueError with a message for the user.
    """
    gf = read_gains(args.gains, fmt=args.gains_format, key=args.gains_key)
    want = [int(a) for a in args.gain_antennas.split(",")] if args.gain_antennas else None
    if gf.antennas is None:
        if want is None:
            raise ValueError(f"{args.gains} does not list its antennas: give --gain-antennas")
        if len(want) != gf.gains.shape[1]:
            raise ValueError(f"{args.gains} has {gf.gains.shape[1]} columns, --gain-antennas "
                             f"lists {len(want)}")
        g, interp, ants = gf.gains, gf.interpolated, want
    else:
        ants = want if want is not None else list(gf.antennas)
        missing = [a for a in ants if a not in gf.antennas]
        if missing:
            raise ValueError(f"antennas {missing} not in {args.gains} (has {gf.antennas})")
        cols = [gf.antennas.index(a) for a in ants]
        g, interp = gf.gains[:, cols], gf.interpolated[:, cols]
    if n_freq is not None and g.shape[0] != n_freq:
        raise ValueError(f"gains of {args.gains} have {g.shape[0]} channels, the data {n_freq}")
    return g, ants, interp
