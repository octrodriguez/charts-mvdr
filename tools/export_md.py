#!/usr/bin/env python
"""Export the tracked source files of a git repository as one Markdown bundle.

Each file becomes a ``## FILE: <path>`` section with a fenced code block. The
fence is longer than any backtick run inside the file, so Markdown files that
contain their own code blocks do not break the bundle. Only files tracked by
git are considered (``git ls-files``), so ``.gitignore`` is respected.

Examples
--------
  # charts-mvdr: package, scripts, tests, jobs, README
  python tools/export_md.py . -o charts_mvdr_bundle.md \
      -i 'charts_mvdr/*' 'scripts/*' 'tests/*' 'jobs/*' README.md pyproject.toml

  # Kotekan: CHARTS simulator only
  python tools/export_md.py $SCRATCH/kotekan -o kotekan_sim_bundle.md \
      -i 'test_charts/*' 'lib/stages/chartsFEngineSim*' -x '*.png' '*.h5'
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import re
import subprocess
from pathlib import Path

LANG = {
    ".py": "python", ".cpp": "cpp", ".hpp": "cpp", ".h": "cpp", ".cu": "cuda",
    ".c": "c", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".md": "markdown", ".txt": "text", ".cfg": "ini",
    ".ini": "ini", ".cmake": "cmake", ".tex": "latex",
}
BINARY_EXT = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".h5", ".hdf5", ".npz",
              ".npy", ".bin", ".so", ".o", ".a", ".zip", ".gz", ".tar", ".mp4"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-i", "--include", nargs="+", default=["*"],
                   help="glob patterns relative to the repo root (default: all)")
    p.add_argument("-x", "--exclude", nargs="*", default=[])
    p.add_argument("--max-kb", type=float, default=300.0,
                   help="skip files larger than this (default: 300 kB)")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    repo = args.repo.resolve()
    commit = git(repo, "rev-parse", "--short", "HEAD")
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    date = git(repo, "log", "-1", "--format=%cs")
    remote = ""
    try:
        remote = git(repo, "remote", "get-url", "origin")
    except subprocess.CalledProcessError:
        pass
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=no"))

    files, skipped = [], []
    for rel in git(repo, "ls-files").splitlines():
        if "__pycache__" in rel or not matches(rel, args.include) or matches(rel, args.exclude):
            continue
        path = repo / rel
        if path.suffix.lower() in BINARY_EXT:
            skipped.append((rel, "binary"))
            continue
        if path.stat().st_size > args.max_kb * 1024:
            skipped.append((rel, f"{path.stat().st_size / 1024:.0f} kB"))
            continue
        try:
            files.append((rel, path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            skipped.append((rel, "not UTF-8"))

    title = args.title or f"{repo.name} @ {commit} ({date})"
    out = [f"# {title}", "",
           f"- Repository: `{remote or repo}`",
           f"- Branch: `{branch}`, commit `{commit}` ({date})"
           + (" — **working tree has uncommitted changes**" if dirty else ""),
           f"- Exported: {dt.datetime.now():%Y-%m-%d %H:%M}",
           f"- Files: {len(files)}" + (f" (skipped {len(skipped)})" if skipped else ""),
           "", "## Contents", ""]
    out += [f"- `{rel}` ({len(txt.splitlines())} lines)" for rel, txt in files]
    if skipped:
        out += ["", "Skipped: " + ", ".join(f"`{r}` ({why})" for r, why in skipped)]
    for rel, txt in files:
        f = fence_for(txt)
        out += ["", f"## FILE: {rel}", f"{f}{LANG.get(Path(rel).suffix.lower(), '')}",
                txt.rstrip("\n"), f]
    args.output.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(files)} files -> {args.output} "
          f"({args.output.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()