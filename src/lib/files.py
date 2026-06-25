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
        raw = Path(path)
    except OSError:
        return None
    for base, prefix in roots:
        base_path = Path(base).resolve()
        candidates: list[Path] = []
        try:
            if raw.is_absolute():
                candidates.append(raw.resolve())
            else:
                rel = Path(str(path).replace("\\", "/"))
                candidates.append((base_path / rel).resolve())
                if rel.parts and rel.parts[0] == base_path.name:
                    candidates.append((base_path.parent / rel).resolve())
        except OSError:
            continue
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                relpath = candidate.relative_to(base_path)
                return f"{prefix}/{relpath.as_posix()}"
            except ValueError:
                continue
    return None
