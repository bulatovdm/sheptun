#!/usr/bin/env python3
"""Build replacements.check.yaml from a verdicts JSON.

Copies src/sheptun/config/replacements.yaml; every rule whose line is marked
WRONG is commented out and prefixed with `# >>> WRONG: <reason>`. Correct rules
stay as-is. The result remains valid YAML (commented rules are inert).

Usage:
    python3 make_check.py VERDICTS.json [REPLACEMENTS.yaml] [OUT.yaml]

VERDICTS.json is a list of objects with at least: {"line": int, "reason": str}
for the WRONG rules (OK rules may be present too and are ignored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_SRC = "src/sheptun/config/replacements.yaml"
DEFAULT_OUT = "replacements.check.yaml"


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

    src = src_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = [
        "# replacements.check.yaml — РЕВЬЮ автозамен (сгенерировано)",
        "# Ошибочные правила ЗАКОММЕНТИРОВАНЫ и помечены '# >>> WRONG: причина'.",
        "# Корректные — как есть. Пройдись сам: раскомментируй/поправь спорные.",
        "",
    ]
    for i, line in enumerate(src, start=1):
        if i in wrong:
            out.append(f"# >>> WRONG: {wrong[i]}")
            out.append(f"# {line}")
        else:
            out.append(line)

    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{out_path}: WRONG={len(wrong)}, всего строк={len(src)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
