#!/usr/bin/env python
"""Kotekan chartsMVDRBeams on real correlation frames, compared with 17_real_nulling.py.

The stage reads the raw Kotekan correlation sums S_ij = sum conj(v_i) v_j in correlator
element order and works in the voltage domain (simulator steering sign, ENU positions,
gains conjugated on load). 17_real_nulling.py reads the same frames through charts-utils
(logical order, V = S, steering sign -1, gains as in the file). Both must give the same
beam power up to a global scale (the cube holds sums over the samples of a frame, the
stage power per sample): this checks the element order, the steering sign and the gains
conjugation of the stage on real data.

prepare  Links a window of frames of a run directory (both frequency blocks, renumbered
         from 0 because rawFileRead starts at file 0 and stops at the first missing file)
         into OUT/window, together with the run's kotekan_config_*.json and
         run_metadata_*.json, so that Kotekan and charts-utils read exactly the same
         frames. Writes one Kotekan config per frequency block,
         OUT/block<b>/mvdr.yaml (rawFileRead -> chartsMVDRBeams -> rawFileWrite of the beam
         power and weights into OUT/block<b>/mvdr), and OUT/manifest.json.
         From the run config: elements, channels per block, block size, the metadata pool
         type, the samples per correlation frame (samples_per_data_set times the frames of
         a chartsAccumulate stage, if any) and the first channel of each block (traced from
         its rawFileWrite stage back to its network handler, or --block-start).
         Positions: logical file, written in correlator element order with
         --antenna-order. Gains: full-width file (tools/kotekan_gains.py); narrower files
         are refused because chartsMVDRBeams would map them onto a fixed antenna set.
         CPU cores: four cores outside those of the run's own pipeline (cpu_affinity,
         dpdk lcores), or --cpu-cores.

compare  Reads the stage power of every block and the real_nulling.npz of a
         17_real_nulling.py run on a cube extracted from OUT/window (stride 1), checks that
         17 used the same frames, look direction (--beam), window, guard and loading, fits
         the global scale on the DFT power and reports the relative residual of the DFT
         and MVDR power. Prints "KOTEKAN REAL MATCH" when the 99th percentile of the
         residual is below --tol in both, else "KOTEKAN REAL MISMATCH". The scale times the
         samples per frame is 1 when the cube holds raw sums.

Example (on the node with the data; the Kotekan command is printed by prepare):
  python scripts/18_kotekan_real_mvdr.py prepare --input RUN_DIR --first 860 --count 240 \\
      --gains gains_full.npy --positions positions.npy --beams 0.0,0.31 \\
      --loading 0.1 --loading-ref noise --out OUT
  python scripts/16_rfi_census.py extract --input OUT/window --antennas 0,4,32,36 \\
      --stride 1 --out OUT/vis_window.npz
  python scripts/17_real_nulling.py --vis OUT/vis_window.npz --gains gains.npy \\
      --gain-antennas 0,4,32,36 --interp-gains --positions positions.npy --look 0.0,0.31 \\
      --sign auto --clean <MHz ranges> --half-window 40 --loading 0.1 --out OUT/nulling
  python scripts/18_kotekan_real_mvdr.py compare --prepared OUT --nulling OUT/nulling
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from charts_mvdr.stage_reference import element_to_antenna, read_stage_power

LOADING_REF_17 = {"noise": "median"}   # stage loading_ref -> 17_real_nulling.py --loading-ref


# --------------------------------------------------------------------------- run config
def _resolve(cfg: dict, value):
    """Numeric config value; a string naming a root key is looked up once."""
    if isinstance(value, str) and value in cfg:
        value = cfg[value]
    return int(value)


def _stages(cfg: dict):
    """(name, dict) of every stage-like block of a Kotekan config, nested gpu blocks included."""
    for name, val in cfg.items():
        if isinstance(val, dict):
            if "kotekan_stage" in val:
                yield name, val
            for sub_name, sub in val.items():
                if isinstance(sub, dict) and "kotekan_stage" in sub:
                    yield f"{name}.{sub_name}", sub


def samples_per_frame(cfg: dict) -> int:
    """Samples summed in one stored correlation frame."""
    n = _resolve(cfg, cfg["samples_per_data_set"])
    for _, st in _stages(cfg):
        if st.get("kotekan_stage") == "chartsAccumulate":
            return n * _resolve(cfg, st.get("num_frames_to_accumulate", 1))
    return n


def block_writers(cfg: dict) -> dict:
    """{block index: rawFileWrite stage} for writers whose file_name ends in _<index>."""
    out = {}
    for _, st in _stages(cfg):
        if st.get("kotekan_stage") == "rawFileWrite":
            m = re.search(r"_(\d+)$", str(st.get("file_name", "")))
            if m:
                out[int(m.group(1))] = st
    return out


def block_first_channel(cfg: dict, buffer: str, max_steps: int = 10) -> int:
    """First F-engine channel of the data in `buffer`, traced back to its network handler."""
    for _ in range(max_steps):
        for h in cfg.get("dpdk", {}).get("handlers", []):
            if h.get("out_buffer") == buffer:
                return _resolve(cfg, h["coarse_freq_start"])
        nxt = None
        for _, st in _stages(cfg):
            if st.get("out_buf") == buffer and "in_buf" in st:
                nxt = st["in_buf"]
                break
            outs = st.get("out_buffers")
            if isinstance(outs, dict) and buffer in outs.values():
                nxt = next(iter(st.get("in_buffers", {}).values()), None)
                break
        if nxt is None:
            break
        buffer = nxt
    raise ValueError(f"cannot trace the first channel of buffer {buffer!r}; use --block-start")


def metadata_pool_type(cfg: dict, buffer: str) -> str:
    pool = cfg.get(buffer, {}).get("metadata_pool")
    if not pool or "kotekan_metadata_pool" not in cfg.get(pool, {}):
        raise ValueError(f"buffer {buffer!r} has no metadata pool in the run config")
    return cfg[pool]["kotekan_metadata_pool"]


def pipeline_cores(cfg: dict) -> set:
    used = set(int(c) for c in cfg.get("cpu_affinity", []))
    dpdk = cfg.get("dpdk", {})
    used |= set(int(c) for c in dpdk.get("lcore_cpu_map", []))
    if "main_lcore_cpu" in dpdk:
        used.add(int(dpdk["main_lcore_cpu"]))
    return used


def free_cores(used: set, n: int = 4, n_cpu: int | None = None) -> list:
    n_cpu = n_cpu or os.cpu_count() or 1
    cores = [c for c in range(n_cpu - 1, -1, -1) if c not in used][:n]
    if not cores:
        raise ValueError("no CPU core outside the run pipeline; use --cpu-cores")
    return sorted(cores)


def channel_frequencies(n_total: int, f0_mhz, df_mhz):
    """F-engine channel frequencies [MHz]: charts_constants.FREQ, or f0 + df k if given."""
    if f0_mhz is not None and df_mhz is not None:
        return float(f0_mhz) + float(df_mhz) * np.arange(n_total)
    try:
        from charts_constants import FREQ
    except ImportError:
        raise SystemExit("error: charts_constants not available; give --freq-start-mhz and "
                         "--delta-freq-mhz") from None
    f = np.asarray(FREQ, dtype=float)
    if f.size < n_total:
        raise SystemExit(f"error: charts_constants.FREQ has {f.size} channels, need {n_total}")
    return f


# --------------------------------------------------------------------------- yaml
def expected_weight_updates(n_frames, half_window, guard, min_frames, update_every) -> int:
    """Weight updates of chartsMVDRBeams on n_frames frames (rule of the stage)."""
    count, has = 0, False
    for t in range(n_frames):
        lo, hi = max(0, t - half_window), min(n_frames - 1, t + half_window)
        n_win = sum(1 for k in range(lo, hi + 1) if abs(k - t) > guard)
        if n_win >= min_frames and (not has or t % update_every == 0):
            count += 1
            has = True
    return count


def stage_yaml(*, window_dir, file_name, out_dir, n_frames, num_elements, num_local_freq,
               block_size, samples, pool_type, freq_start_mhz, delta_freq_mhz, pos_x, pos_y,
               beams, half_window, guard, min_frames, update_every, loading, loading_ref,
               gains_file, gains_first_channel, antenna_order, cores) -> str:
    n_rec = num_elements // block_size
    num_blocks = n_rec * (n_rec + 1) // 2
    n_beam = len(beams)
    corr_frames = 2 * half_window + 4
    weights_size = 16 + num_local_freq * n_beam * num_elements * 8 + num_local_freq * n_beam * 4
    n_updates = expected_weight_updates(n_frames, half_window, guard, min_frames, update_every)
    if n_updates == 0:
        raise ValueError("no weight update would happen: window too short for cov_min_frames")
    js = json.dumps
    return f"""######################################################################
# chartsMVDRBeams on real correlation frames (replay)
# Source: {Path(window_dir).as_posix()}/{file_name}_%07d.bin ({n_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {js(cores)}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples}
num_data_sets: 1
block_size: {block_size}
num_blocks: {num_blocks}
sizeof_int: 4

main_pool:
  kotekan_metadata_pool: {pool_type}
  num_metadata_objects: {2 * corr_frames + 20}

host_correlation_buffer:
  kotekan_buffer: standard
  num_frames: {corr_frames}
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

mvdr_power_buffer:
  kotekan_buffer: standard
  num_frames: 4
  frame_size: {2 * num_local_freq * n_beam * 4}
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

mvdr_weights_buffer:
  kotekan_buffer: standard
  num_frames: 4
  frame_size: {weights_size}
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

correlation_reader:
  kotekan_stage: rawFileRead
  buf: host_correlation_buffer
  base_dir: "{Path(window_dir).as_posix()}"
  file_name: "{file_name}"
  file_ext: "bin"
  prefix_hostname: false
  end_interrupt: false

mvdr_beams:
  kotekan_stage: chartsMVDRBeams
  in_buf: host_correlation_buffer
  out_buf: mvdr_power_buffer
  weights_buf: mvdr_weights_buffer
  samples_per_frame: {samples}
  freq_start_mhz: {float(freq_start_mhz)!r}
  delta_freq_mhz: {float(delta_freq_mhz)!r}
  positions_x_m: {js([float(v) for v in pos_x])}
  positions_y_m: {js([float(v) for v in pos_y])}
  beams: {js([[float(l), float(m)] for l, m in beams])}
  cov_half_window: {half_window}
  cov_guard: {guard}
  cov_min_frames: {min_frames}
  update_every: {update_every}
  loading: {float(loading)!r}
  loading_ref: {loading_ref}
  flush_after_frames: {n_frames}
  gains_file: "{Path(gains_file).as_posix()}"
  gains_first_channel: {gains_first_channel}
  antenna_order: {antenna_order}

mvdr_power_dump:
  kotekan_stage: rawFileWrite
  in_buf: mvdr_power_buffer
  base_dir: "{Path(out_dir).as_posix()}"
  file_name: "mvdr_power"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false
  exit_after_n_files: {n_frames}

mvdr_weights_dump:
  kotekan_stage: rawFileWrite
  in_buf: mvdr_weights_buffer
  base_dir: "{Path(out_dir).as_posix()}"
  file_name: "mvdr_weights"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false
  exit_after_n_files: {n_updates}
"""


# --------------------------------------------------------------------------- prepare
def parse_beam(text: str):
    l, m = (float(v) for v in text.split(","))
    return l, m


def source_files(input_dir: Path, file_name: str) -> tuple:
    """(prefix, {index: path}) of the frames <prefix>_<file_name>_%07d.bin of one block."""
    rx = re.compile(rf"^(?:(?P<prefix>.+)_)?{re.escape(file_name)}_(?P<idx>\d{{7}})\.bin$")
    files, prefixes = {}, set()
    for p in input_dir.glob(f"*{file_name}_*.bin"):
        m = rx.match(p.name)
        if m:
            files[int(m.group("idx"))] = p
            prefixes.add(m.group("prefix") or "")
    if len(prefixes) > 1:
        raise ValueError(f"several file prefixes for {file_name} in {input_dir}: {sorted(prefixes)}")
    return (prefixes.pop() if prefixes else ""), files


def cmd_prepare(args) -> int:
    inp = Path(args.input)
    configs = sorted(inp.glob("kotekan_config_*.json"))
    if not configs:
        print(f"error: no kotekan_config_*.json in {inp}", file=sys.stderr)
        return 1
    cfg = json.loads(configs[-1].read_text())
    n_el = _resolve(cfg, cfg["num_elements"])
    n_f = _resolve(cfg, cfg["num_local_freq"])
    bs = _resolve(cfg, cfg.get("block_size", 2))
    samples = samples_per_frame(cfg)
    writers = block_writers(cfg)
    blocks = ([int(b) for b in args.blocks.split(",")] if args.blocks else sorted(writers))
    missing_w = [b for b in blocks if b not in writers]
    if missing_w:
        print(f"error: no rawFileWrite stage for blocks {missing_w} in {configs[-1].name}",
              file=sys.stderr)
        return 1
    starts = {}
    if args.block_start:
        for item in args.block_start.split(","):
            b, s = item.split(":")
            starts[int(b)] = int(s)
    for b in blocks:
        if b not in starts:
            starts[b] = block_first_channel(cfg, writers[b]["in_buf"])
    pool = metadata_pool_type(cfg, writers[blocks[0]]["in_buf"])
    freqs = channel_frequencies(max(starts.values()) + n_f, args.freq_start_mhz,
                                args.delta_freq_mhz)
    df = float(freqs[1] - freqs[0])

    gains = np.load(args.gains)
    if gains.ndim != 2 or gains.shape[1] != n_el:
        print(f"error: {args.gains} has shape {gains.shape}; chartsMVDRBeams needs one column "
              f"per correlator input ({n_el}). Expand it with tools/kotekan_gains.py.",
              file=sys.stderr)
        return 1
    if gains.shape[0] < max(starts.values()) + n_f:
        print(f"error: {args.gains} has {gains.shape[0]} channels, blocks need "
              f"{max(starts.values()) + n_f}", file=sys.stderr)
        return 1
    used_ants = [int(a) for a in np.flatnonzero(np.isfinite(gains).any(axis=0))]
    pos = np.load(args.positions)
    if pos.ndim != 2 or pos.shape[0] < n_el or pos.shape[1] < 2:
        print(f"error: positions {pos.shape}; need ({n_el}, >= 2), logical order",
              file=sys.stderr)
        return 1
    e2a = element_to_antenna(n_el, args.antenna_order)
    pos_x, pos_y = pos[e2a, 0], pos[e2a, 1]
    beams = [parse_beam(b) for b in args.beams]
    min_frames = args.min_frames if args.min_frames is not None else args.half_window - args.guard
    cores = ([int(c) for c in args.cpu_cores.split(",")] if args.cpu_cores
             else free_cores(pipeline_cores(cfg)))

    out = Path(args.out)
    win = out / "window"
    win.mkdir(parents=True, exist_ok=True)
    if any(win.iterdir()):
        print(f"error: {win} is not empty", file=sys.stderr)
        return 1
    sel = range(args.first, args.first + args.count)
    links, names = {}, {}
    for b in blocks:
        fname = writers[b]["file_name"]
        prefix, files = source_files(inp, fname)
        absent = [i for i in sel if i not in files]
        if absent:
            print(f"error: block {b}: frames {absent[:5]}{'...' if len(absent) > 5 else ''} "
                  f"missing in {inp}", file=sys.stderr)
            return 1
        stem = f"{prefix}_{fname}" if prefix else fname
        names[b] = stem
        links[b] = [(files[i], win / f"{stem}_{k:07d}.bin") for k, i in enumerate(sel)]
    for b in blocks:
        for src, dst in links[b]:
            os.symlink(src.resolve(), dst)
    for p in list(configs) + sorted(inp.glob("run_metadata_*.json")):
        os.symlink(p.resolve(), win / p.name)

    manifest = {
        "input": str(inp.resolve()), "run_config": configs[-1].name,
        "first_frame": args.first, "count": args.count, "blocks": {},
        "num_elements": n_el, "num_local_freq": n_f, "block_size": bs,
        "samples_per_frame": samples, "metadata_pool": pool, "delta_freq_mhz": df,
        "beams": beams, "half_window": args.half_window, "guard": args.guard,
        "min_frames": min_frames, "update_every": args.update_every, "loading": args.loading,
        "loading_ref": args.loading_ref, "antenna_order": args.antenna_order,
        "gains_file": str(Path(args.gains).resolve()), "gain_antennas": used_ants,
        "positions_file": str(Path(args.positions).resolve()), "cpu_cores": cores,
    }
    for b in blocks:
        bdir = out / f"block{b}"
        mdir = bdir / "mvdr"
        mdir.mkdir(parents=True, exist_ok=True)
        yaml_text = stage_yaml(
            window_dir=win.resolve(), file_name=names[b], out_dir=mdir.resolve(),
            n_frames=args.count, num_elements=n_el, num_local_freq=n_f, block_size=bs,
            samples=samples, pool_type=pool, freq_start_mhz=float(freqs[starts[b]]),
            delta_freq_mhz=df, pos_x=pos_x, pos_y=pos_y, beams=beams,
            half_window=args.half_window, guard=args.guard, min_frames=min_frames,
            update_every=args.update_every, loading=args.loading, loading_ref=args.loading_ref,
            gains_file=Path(args.gains).resolve(), gains_first_channel=starts[b],
            antenna_order=args.antenna_order, cores=cores)
        (bdir / "mvdr.yaml").write_text(yaml_text)
        manifest["blocks"][str(b)] = {"file_name": names[b], "first_channel": starts[b],
                                      "freq_start_mhz": float(freqs[starts[b]])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"run config {configs[-1].name}: {n_el} elements, {n_f} channels per block, "
          f"{samples} samples per frame, metadata {pool}")
    print(f"window: frames {args.first}-{args.first + args.count - 1} -> {win} "
          f"(blocks {blocks}, first channels {[starts[b] for b in blocks]})")
    print(f"gains: inputs with a solution {used_ants}; CPU cores {cores}")
    for b in blocks:
        bdir = (out / f"block{b}").resolve()
        print(f"block {b}: kotekan -c {bdir / 'mvdr.yaml'} -b 127.0.0.1:0 "
              f"> {bdir / 'kotekan.log'} 2>&1")
    return 0


# --------------------------------------------------------------------------- compare
def load_nulling(path: Path) -> dict:
    p = path / "real_nulling.npz" if path.is_dir() else path
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def cmd_compare(args) -> int:
    prep = Path(args.prepared)
    man = json.loads((prep / "manifest.json").read_text())
    ref = load_nulling(Path(args.nulling))
    n_beam = len(man["beams"])
    if not 0 <= args.beam < n_beam:
        print(f"error: --beam {args.beam} outside 0..{n_beam - 1}", file=sys.stderr)
        return 1
    errors = []
    h, g, loading = (float(v) for v in ref["params"])
    if (int(h), int(g)) != (man["half_window"], man["guard"]) or not np.isclose(loading, man["loading"]):
        errors.append(f"17 window/guard/loading {int(h)}/{int(g)}/{loading:g} vs stage "
                      f"{man['half_window']}/{man['guard']}/{man['loading']:g}")
    lref = str(ref["loading_ref"])
    if LOADING_REF_17.get(man["loading_ref"]) != lref:
        errors.append(f"17 loading_ref {lref} vs stage {man['loading_ref']}")
    look = np.asarray(ref["look"], float).reshape(-1, 2)
    if not np.allclose(look, np.asarray(man["beams"][args.beam])[None, :], atol=1e-9):
        errors.append(f"17 look {look[0].tolist()} is not beam {args.beam} "
                      f"{man['beams'][args.beam]}")
    if sorted(int(a) for a in ref["ants"]) != sorted(man["gain_antennas"]):
        errors.append(f"17 antennas {ref['ants'].tolist()} vs gains {man['gain_antennas']}")
    n_ref = ref["power"].shape[1]
    if n_ref != man["count"]:
        errors.append(f"17 has {n_ref} frames, the window {man['count']} (frames skipped by "
                      f"charts-utils?)")
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1

    fref = np.asarray(ref["freqs"], float)
    df = man["delta_freq_mhz"]
    st_list, rf_list, f_list = [], [], []
    for b, info in sorted(man["blocks"].items(), key=lambda kv: int(kv[0])):
        p = read_stage_power(prep / f"block{b}" / "mvdr", man["num_local_freq"], n_beam)
        n = min(p.shape[1], n_ref)
        if p.shape[1] != man["count"]:
            print(f"warning: block {b}: stage wrote {p.shape[1]} of {man['count']} frames; "
                  f"comparing {n}")
        f_st = info["freq_start_mhz"] + df * np.arange(man["num_local_freq"])
        j = np.abs(fref[None, :] - f_st[:, None]).argmin(axis=1)
        ok = np.abs(fref[j] - f_st) < 0.1 * df
        st_list.append(p[:, :n, ok, args.beam])
        rf_list.append(np.asarray(ref["power"], float)[:, :n][:, :, j[ok]])
        f_list.append(f_st[ok])
    n = min(a.shape[1] for a in st_list)
    st = np.concatenate([a[:, :n] for a in st_list], axis=2).astype(float)
    rf = np.concatenate([a[:, :n] for a in rf_list], axis=2)

    fin_st, fin_rf = np.isfinite(st), np.isfinite(rf)
    nan_mismatch = int(np.count_nonzero(fin_st != fin_rf))
    both = fin_st & fin_rf & (np.abs(rf) > 0)
    if not both[0].any():
        print("error: no channel with finite DFT power in both", file=sys.stderr)
        return 1
    scale = float(np.median(st[0][both[0]] / rf[0][both[0]]))
    scale_mvdr = float(np.median(st[1][both[1]] / rf[1][both[1]])) if both[1].any() else np.nan
    print(f"frames {n}, channels {st.shape[2]}, beam {args.beam} {man['beams'][args.beam]}")
    print(f"scale stage / 17: {scale:.6g} (x samples per frame = "
          f"{scale * man['samples_per_frame']:.6g}); MVDR plane {scale_mvdr:.6g}")
    verdict = nan_mismatch == 0
    for k, name in ((0, "DFT"), (1, "MVDR")):
        if not both[k].any():
            print(f"{name}: no common finite values")
            verdict = False
            continue
        rel = np.abs(st[k][both[k]] - scale * rf[k][both[k]]) / np.abs(scale * rf[k][both[k]])
        p50, p99, mx = np.percentile(rel, 50), np.percentile(rel, 99), rel.max()
        print(f"{name} relative residual: median {p50:.2e}, p99 {p99:.2e}, max {mx:.2e}")
        verdict &= bool(p99 <= args.tol)
    print(f"NaN pattern mismatches: {nan_mismatch}")
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.nanmedian(10 * np.log10(st[1] / st[0]), axis=0)
    print(f"stage MVDR/DFT median over frames: median over channels {np.nanmedian(ratio):+.2f} dB, "
          f"min {np.nanmin(ratio):+.2f} dB")
    print("KOTEKAN REAL MATCH" if verdict else "KOTEKAN REAL MISMATCH")
    return 0 if verdict else 2


# --------------------------------------------------------------------------- cli
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="window links, Kotekan configs and manifest")
    p.add_argument("--input", required=True, help="Run directory (frames, kotekan_config_*.json).")
    p.add_argument("--first", type=int, required=True, help="First frame index of the window.")
    p.add_argument("--count", type=int, required=True, help="Frames in the window.")
    p.add_argument("--blocks", default=None, help="Frequency blocks, e.g. 0,1 (default: all).")
    p.add_argument("--block-start", default=None,
                   help="First channel per block, e.g. 0:0,1:336 (default: from the run config).")
    p.add_argument("--gains", required=True, help="Full-width gains (tools/kotekan_gains.py).")
    p.add_argument("--positions", required=True, help="Positions .npy (n, 3), ENU m, logical order.")
    p.add_argument("--beams", required=True, nargs="+", help="Look directions l,m (one per beam).")
    p.add_argument("--half-window", type=int, default=40)
    p.add_argument("--guard", type=int, default=1)
    p.add_argument("--min-frames", type=int, default=None)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--loading", type=float, default=10.0)
    p.add_argument("--loading-ref", default="noise", choices=("noise", "trace", "absolute"))
    p.add_argument("--antenna-order", default="descending", choices=("descending", "ascending"))
    p.add_argument("--freq-start-mhz", type=float, default=None,
                   help="Frequency of channel 0 (default: charts_constants.FREQ).")
    p.add_argument("--delta-freq-mhz", type=float, default=None)
    p.add_argument("--cpu-cores", default=None, help="Cores for Kotekan, e.g. 20,21,22,23.")
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="stage power vs 17_real_nulling.py")
    c.add_argument("--prepared", required=True, help="OUT directory of prepare.")
    c.add_argument("--nulling", required=True, help="17_real_nulling.py output dir or npz.")
    c.add_argument("--beam", type=int, default=0, help="Stage beam that 17 used as --look.")
    c.add_argument("--tol", type=float, default=1e-4,
                   help="Maximum 99th percentile of the relative residual.")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "prepare":
        if args.count < 1 or args.first < 0:
            print("error: need --first >= 0 and --count >= 1", file=sys.stderr)
            return 1
        return cmd_prepare(args)
    return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
