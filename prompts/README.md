# Prompt registry

Each file under `<name>/<version>.md` is one immutable version of a prompt
used by the pipeline. TOML frontmatter (`+++`) records metadata; the body
is a `str.format`-compatible template.

## Layout

```
prompts/
  gemini_scan/            # _SYSTEM_INSTRUCTION for the vision scan
    v2.md ... v10.md
  narrative_select/       # composer.py narrative clip picker
    v1.md v2.md
```

Live at runtime: `gemini_scan` fine pass uses `v10` (`_PROMPT_VERSION` in
`gemini_scan.py`), coarse pass uses `v5` (`_COARSE_PROMPT_VERSION` in
`gemini_scan.py`); `narrative_select` uses `v2` (`_NARRATIVE_PROMPT_VERSION`
in `composer.py`). Older versions stay on disk for history/diffing but
aren't loaded by the pipeline.

## Versioning rules

- Bump the version **any time** you change the prompt body. Never edit
  an existing version file in place — create a new one and update the
  `_PROMPT_VERSION` / `_NARRATIVE_PROMPT_VERSION` constant in code.
- The version string is embedded in Gemini cache filenames
  (`<clip>_<version>.json`), so bumping naturally invalidates stale
  responses without manual cache clearing.
- `supersedes` in the frontmatter gives you the immediate predecessor.
  Use `diff prompts/<name>/vN-1.md prompts/<name>/vN.md` to see exactly
  what changed.

## Frontmatter fields

| field | required | meaning |
|---|---|---|
| `name` | yes | must match directory name |
| `version` | yes | must match filename |
| `model` | yes | Gemini model the prompt was authored for |
| `date` | yes | when the version was created |
| `commit` | yes | git SHA (or "uncommitted") where this version landed |
| `supersedes` | no | previous version; omit on the first |
| `placeholders` | yes | `{…}` fields the caller must substitute |
| `rationale` | yes | why this version exists — what changed and why |

## Python API

```python
from gopro_garmin_pipeline.prompt_registry import load_prompt, list_versions

meta, body = load_prompt("gemini_scan", "v5")
list_versions("gemini_scan")  # ['v2', 'v3', 'v4', 'v5']
```
