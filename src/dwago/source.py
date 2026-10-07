"""Resolve source reads within the selected project, preserving logical names."""
from __future__ import annotations

from pathlib import Path


def resolve_source_file(root: Path, path: str | Path) -> Path | None:
    """Return an in-project regular file, or None for unavailable source.

    Symlinks are allowed when their target stays inside the canonical root.
    Callers keep the original path for identity/citations and read the returned
    path. This is not atomic protection against concurrent filesystem changes.
    """
    try:
        root = root.resolve()
        target = (root / path).resolve(strict=True)
        target.relative_to(root)
        return target if target.is_file() else None
    except (OSError, RuntimeError, ValueError):
        # Includes missing files, permissions, invalid paths and symlink loops
        # (RuntimeError on older supported Python versions).
        return None


def read_source_lines(root: Path, path: str | Path) -> list[str]:
    """Read local source for enrichment/context, treating blocked paths as absent."""
    target = resolve_source_file(root, path)
    if target is None:
        return []
    try:
        return target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
