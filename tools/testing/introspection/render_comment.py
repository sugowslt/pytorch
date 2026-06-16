"""Render a `testintro diff --json` result into a sticky PR-comment markdown body.

Used by the Stage-2 comment workflow:
    python -m tools.testing.introspection.render_comment < diff.json > comment.md

Reads the diff result (the object printed by `testintro diff --json`) from stdin or a
file argument and prints markdown. Tests are grouped by the set of platforms they were
added/removed on (the same grouping the CLI uses), collapsed in <details>, with a hard
size cap so the comment stays under GitHub's ~65k limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.testing.introspection import diff as diff_mod


MARKER = "<!-- testintro-diff -->"
_MAX_PER_SECTION = 300  # test lines per added/removed section before truncating
_MAX_CHARS = 60000  # stay under GitHub's ~65k comment limit


def _short(ref: str) -> str:
    return (
        ref[:7] if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref) else ref
    )


def _plats(fs: frozenset[str], all_jobs: set[str]) -> str:
    return "all platforms" if set(fs) == all_jobs else ", ".join(sorted(fs))


def _section(title: str, m: dict[str, set[str]], all_jobs: set[str]) -> list[str]:
    if not m:
        return []
    lines = [f"<details>\n<summary>{title} ({len(m)})</summary>\n"]
    shown = 0
    for fs, tests in diff_mod.group_by_platform_set(m, all_jobs):
        lines.append(f"\n**{_plats(fs, all_jobs)}**")
        for t in tests:
            if shown >= _MAX_PER_SECTION:
                break
            lines.append(f"- `{t}`")
            shown += 1
        if shown >= _MAX_PER_SECTION:
            lines.append(f"\n_... {len(m) - shown} more (see the CI job artifact)_")
            break
    lines.append("\n</details>")
    return lines


def render(res: dict) -> str:
    added, removed, job_names = diff_mod.invert_per_job(res)
    all_jobs = set(job_names)
    header = (
        f"{MARKER}\n"
        f"### Test changes ([testintro](https://github.com/pytorch/pytorch/tree/main/tools/testing/introspection))\n\n"
        f"`{_short(res['from'])}..{_short(res['to'])}` across {len(job_names)} platforms "
        f"— **+{len(added)} added, −{len(removed)} removed**"
    )
    if not added and not removed:
        return header + "\n\nNo tests added or removed."

    parts = [header, ""]
    if res.get("broad"):
        parts += [
            "> ⚠️ This PR changes the test-generation surface (e.g. `common_*` / op_db "
            "/ codegen), which can change tests anywhere; comparison is limited to "
            "`linux-cpu`.",
            "",
        ]
    parts += _section("➕ Added", added, all_jobs)
    parts += _section("➖ Removed", removed, all_jobs)

    # Scope + uncomparable footnote (scope_reason is identical across jobs).
    per_job = res.get("per_job", {})
    reasons = {jr.get("scope_reason", "") for jr in per_job.values()}
    uncomparable = sorted(
        {f for jr in per_job.values() for f in jr.get("uncomparable", [])}
    )
    notes = []
    if reasons:
        notes.append(f"scope: {'; '.join(sorted(r for r in reasons if r))}")
    if uncomparable:
        notes.append(
            f"{len(uncomparable)} file(s) could not be compared (collection error at a ref)"
        )
    if notes:
        parts += ["", "> " + " · ".join(notes)]

    body = "\n".join(parts)
    if len(body) > _MAX_CHARS:
        body = (
            body[:_MAX_CHARS]
            + "\n\n_(truncated; see the CI job artifact for the full list)_"
        )
    return body


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = Path(argv[0]).read_text() if argv else sys.stdin.read()
    print(render(json.loads(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
