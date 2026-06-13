from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fastapi import HTTPException


def safe_file_under(root: Path, relpath: str) -> Path:
    """Resuelve `relpath` dentro de `root` evitando path traversal."""
    root = root.resolve()
    rel = (relpath or "").replace("\\", "/").strip("/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside allowed root") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return candidate


def file_to_public_url(path: Path, roots: Iterable[tuple[Path, str]]) -> str | None:
    """Mapea un path local a una URL publica si cae bajo alguno de los roots servidos."""
    try:
        p = Path(path).resolve()
    except OSError:
        return None
    for base, prefix in roots:
        try:
            rel = p.relative_to(Path(base).resolve())
            return f"{prefix}/{rel.as_posix()}"
        except ValueError:
            continue
    return None
