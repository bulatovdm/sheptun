#!/usr/bin/env python3
"""Build replacements.check.yaml from a verdicts JSON.

Emits ONLY the freshly-added rules (working copy vs HEAD via `git diff`), not the
whole file — the check file should show just what this review pass touched. Every
rule marked WRONG is commented out and prefixed with `# >>> WRONG: <reason>`; the
rest stay as-is. The result is valid YAML (commented rules are inert).

Usage:
    python3 make_check.py VERDICTS.json [REPLACEMENTS.yaml] [OUT.yaml]

VERDICTS.json is a list of objects with at least {"line": int, "reason": str} for
the WRONG rules (OK/FIX rules may be present too; OK is ignored, FIX is written
in place by hand afterwards). `line` refers to the current file's line numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEFAULT_SRC = "src/sheptun/config/replacements.yaml"
DEFAULT_OUT = "replacements.check.yaml"


def _fresh_line_numbers(src_path: Path) -> set[int]:
    """1-based line numbers of rules added in the working copy vs HEAD.

    Parses `git diff` hunks: added ('+') lines advance the new-file counter and are
    collected; context lines advance it too; removed ('-') lines do not.
    """
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", str(src_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fresh: set[int] = set()
    new_lineno = 0
    for line in diff.splitlines():
        if line.startswith("@@"):
            # @@ -old,cnt +new,cnt @@  → take the new-file start
            new_lineno = int(line.split("+")[1].split(",")[0].split(" ")[0]) - 1
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            new_lineno += 1
            if line[1:].startswith('"'):
                fresh.add(new_lineno)
        elif not line.startswith("-"):
            new_lineno += 1
    return fresh


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    verdicts = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    src_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_SRC)
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(DEFAULT_OUT)

    wrong = {
        v["line"]: v.get("reason", "")
        for v in verdicts
        if str(v.get("verdict", "WRONG")).upper() == "WRONG"
    }

    fresh = _fresh_line_numbers(src_path)
    src = src_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = [
        "# replacements.check.yaml — РЕВЬЮ свежих автозамен (сгенерировано)",
        "# Только НЕЗАКОММИЧЕННЫЕ правила (working copy vs HEAD) — что добавил последний проход.",
        "# Ошибочные ЗАКОММЕНТИРОВАНЫ и помечены '# >>> WRONG: причина'. FIX вписаны на месте.",
        "# Пройдись сам: раскомментируй/поправь спорные — применяется по содержимому этого файла.",
        "",
    ]
    for i, line in enumerate(src, start=1):
        if i not in fresh:
            continue
        if i in wrong:
            out.append(f"# >>> WRONG: {wrong[i]}")
            out.append(f"# {line}")
        else:
            out.append(line)

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{out_path}: свежих={len(fresh)}, WRONG={len(wrong)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
