"""Filesystem-backed prompt registry.

Prompts ship as package data under ``gopro_garmin_pipeline/prompts/<name>/
<version>.md`` with TOML
frontmatter (``+++`` delimiters) describing the version and a body
that is the raw prompt template with ``{placeholder}`` substitutions
for ``str.format``.

Each prompt evolves explicitly on disk — git log is the change history,
and cache files can key off the version string so a prompt bump
invalidates cached outputs automatically.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_FM_DELIM = "+++\n"


def _prompt_path(name: str, version: str) -> Path:
    return _PROMPTS_DIR / name / f"{version}.md"


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str) -> tuple[dict, str]:
    """Return (metadata, body) for a named, versioned prompt."""
    path = _prompt_path(name, version)
    raw = path.read_text(encoding='utf-8')
    if not raw.startswith(_FM_DELIM):
        raise ValueError(f"{path}: missing TOML frontmatter (+++)")
    _, fm, body = raw.split(_FM_DELIM, 2)
    metadata = tomllib.loads(fm)
    return metadata, body.lstrip("\n")


def prompt_body(name: str, version: str) -> str:
    """Shortcut when you only need the template body."""
    return load_prompt(name, version)[1]


def list_versions(name: str) -> list[str]:
    """Available versions for a prompt, sorted lexically (v2, v3, ...)."""
    d = _PROMPTS_DIR / name
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.md"))
