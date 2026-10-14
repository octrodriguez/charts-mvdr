#!/usr/bin/env python
"""Export the GitHub-side information of a repository as Markdown.

Collects what is visible on the GitHub web page but not in a clone:
repository metadata, branches, recent commits of a branch, pull requests
(description, changed files, review comments), issues (with comments) and
releases. The README of the selected branch is included as well.

Authentication: a token is read from $GITHUB_TOKEN (fine-grained token with
read access to "Contents", "Pull requests", "Issues" and "Metadata" of the
repository). Without a token only public repositories work, limited to
60 requests per hour.

Examples
--------
  export GITHUB_TOKEN=github_pat_...
  python tools/export_github_md.py OWNER/REPO --branch kotekan_charts -o github_info.md
  python tools/export_github_md.py OWNER/REPO --prs 20 --issues 20 --commits 50 -o info.md
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"


class GitHub:
    def __init__(self, token: str | None):
        self.headers = {"Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                        "User-Agent": "charts-mvdr-export"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.calls = 0

    def get(self, path: str, **params):
        url = f"{API}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=self.headers)
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            raise SystemExit(f"GitHub API error {e.code} on {path}: {msg}")

    def paged(self, path: str, limit: int, **params):
        out, page = [], 1
        while len(out) < limit:
            batch = self.get(path, per_page=min(100, limit), page=page, **params)
            if not batch:
                break
            out += batch
            page += 1
        return out[:limit]


def quote(text: str | None) -> str:
    return (text or "_(no description)_").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", help="OWNER/REPO")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--branch", default=None, help="branch for README and commits (default: repo default)")
    p.add_argument("--commits", type=int, default=30)
    p.add_argument("--prs", type=int, default=15)
    p.add_argument("--issues", type=int, default=15)
    p.add_argument("--no-comments", action="store_true", help="skip comments (fewer API calls)")
    args = p.parse_args()

    gh = GitHub(os.environ.get("GITHUB_TOKEN"))
    r = f"/repos/{args.repo}"
    meta = gh.get(r)
    branch = args.branch or meta["default_branch"]
    L = [f"# GitHub: {meta['full_name']}", "",
         f"- URL: {meta['html_url']}",
         f"- Description: {meta.get('description') or '-'}",
         f"- Visibility: {'private' if meta['private'] else 'public'}; default branch `{meta['default_branch']}`",
         f"- Fork of: {meta['parent']['full_name']}" if meta.get("fork") and meta.get("parent") else "- Fork: no",
         f"- Last push: {meta['pushed_at']}; exported {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"- Selected branch: `{branch}`", ""]

    # Branches
    branches = gh.paged(f"{r}/branches", 100)
    L += ["## Branches", ""] + [f"- `{b['name']}` @ `{b['commit']['sha'][:8]}`" for b in branches] + [""]

    # Commits
    commits = gh.paged(f"{r}/commits", args.commits, sha=branch)
    L += [f"## Recent commits on `{branch}`", "",
          "| SHA | Date | Author | Message |", "|---|---|---|---|"]
    for c in commits:
        msg = c["commit"]["message"].splitlines()[0].replace("|", "\\|")
        L.append(f"| `{c['sha'][:8]}` | {c['commit']['author']['date'][:10]} | "
                 f"{c['commit']['author']['name']} | {msg} |")
    L.append("")

    # Pull requests
    prs = gh.paged(f"{r}/pulls", args.prs, state="all", sort="updated", direction="desc")
    L += ["## Pull requests", ""]
    for pr in prs:
        state = "merged" if pr.get("merged_at") else pr["state"]
        L += [f"### #{pr['number']} {pr['title']} ({state})", "",
              f"- `{pr['head']['ref']}` → `{pr['base']['ref']}`, by {pr['user']['login']}, "
              f"updated {pr['updated_at'][:10]}", f"- {pr['html_url']}", "", quote(pr["body"]), ""]
        files = gh.paged(f"{r}/pulls/{pr['number']}/files", 300)
        if files:
            L += ["Changed files:", ""] + [
                f"- `{f['filename']}` ({f['status']}, +{f['additions']}/-{f['deletions']})" for f in files] + [""]
        if not args.no_comments:
            comments = (gh.paged(f"{r}/issues/{pr['number']}/comments", 100)
                        + gh.paged(f"{r}/pulls/{pr['number']}/comments", 100))
            for cm in sorted(comments, key=lambda c: c["created_at"]):
                where = f" on `{cm['path']}`" if cm.get("path") else ""
                L += [f"> **{cm['user']['login']}** ({cm['created_at'][:10]}){where}:", ">",
                      *[f"> {line}" for line in quote(cm["body"]).splitlines()], ""]

    # Issues (the issues endpoint also returns PRs; filter them out)
    issues = [i for i in gh.paged(f"{r}/issues", args.issues * 2, state="all",
                                  sort="updated", direction="desc")
              if "pull_request" not in i][:args.issues]
    L += ["## Issues", ""] if issues else ["## Issues", "", "_(none)_", ""]
    for it in issues:
        labels = ", ".join(l["name"] for l in it.get("labels", []))
        L += [f"### #{it['number']} {it['title']} ({it['state']})", "",
              f"- by {it['user']['login']}, updated {it['updated_at'][:10]}"
              + (f"; labels: {labels}" if labels else ""), f"- {it['html_url']}", "",
              quote(it["body"]), ""]
        if not args.no_comments and it.get("comments"):
            for cm in gh.paged(f"{r}/issues/{it['number']}/comments", 100):
                L += [f"> **{cm['user']['login']}** ({cm['created_at'][:10]}):", ">",
                      *[f"> {line}" for line in quote(cm["body"]).splitlines()], ""]

    # Releases
    rel = gh.paged(f"{r}/releases", 10)
    if rel:
        L += ["## Releases", ""]
        for x in rel:
            L += [f"### {x['tag_name']} — {x.get('name') or ''} ({x['published_at'][:10] if x.get('published_at') else 'draft'})",
                  "", quote(x.get("body")), ""]

    # README of the selected branch
    try:
        readme = gh.get(f"{r}/readme", ref=branch)
        text = base64.b64decode(readme["content"]).decode("utf-8", errors="replace")
        fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
        L += [f"## README ({readme['path']} on `{branch}`)", "", f"{fence}markdown", text.rstrip(), fence, ""]
    except SystemExit:
        L += ["## README", "", "_(not found)_", ""]

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"{args.output}: {len(branches)} branches, {len(commits)} commits, {len(prs)} PRs, "
          f"{len(issues)} issues ({gh.calls} API calls)")


if __name__ == "__main__":
    main()