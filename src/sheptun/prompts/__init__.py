"""Prompt templates, stored as editable Markdown files in this package.

Add a new prompt by dropping a `<name>.md` file here and calling
`load_prompt("<name>")`. Keeps prompt text out of the code so it can be tuned
without touching Python.
"""

from __future__ import annotations

from importlib import resources

_PACKAGE = "sheptun.prompts"


def load_prompt(name: str) -> str:
    """Load a prompt by file stem (without the .md extension)."""
    return resources.files(_PACKAGE).joinpath(f"{name}.md").read_text(encoding="utf-8").strip()
