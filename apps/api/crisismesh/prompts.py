"""Versioned prompt templates, loaded from /prompts.

Prompts live in files so that:
  - the exact text used for any event can be reconstructed from
    (prompt_version, model_label) in the audit log,
  - a judge can read what we asked the model without reading Python,
  - extraction_v1 -> extraction_v2 is a reviewable diff rather than a git blame.

extraction_v2 is the validated version. v1 scored ~50% event-type recall on the
spike set because it had no taxonomy definitions and no worked examples; v2
reached 92% on the same set. Both are kept because the evaluation report cites
the delta.
"""

from __future__ import annotations

from functools import lru_cache

from .config import PROMPTS_DIR


@lru_cache(maxsize=32)
def load_prompt(name: str, version: str) -> str:
    """Load prompts/<name>_<version>.md.

    The version may be given as plain "v2" or as the full "extraction_v2". The
    long form is what we store in the audit log, where the stage name is useful
    context, so accepting both avoids a mismatch between the recorded value and
    the file on disk.
    """
    version = version.strip()
    prefix = f"{name}_"
    if version.startswith(prefix):
        version = version[len(prefix):]
    path = PROMPTS_DIR / f"{name}_{version}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt {name}@{version} not found at {path}. "
            f"Available: {list_prompts()}"
        )
    return path.read_text(encoding="utf-8")


def list_prompts() -> list[str]:
    return sorted(p.name for p in PROMPTS_DIR.glob("*.md"))


def extraction_prompt(version: str) -> str:
    return load_prompt("extraction", version)


def conflict_prompt(version: str) -> str:
    return load_prompt("conflict", version)


def normalization_prompt(version: str) -> str:
    return load_prompt("normalization", version)


def place_prompt(version: str) -> str:
    return load_prompt("place", version)