"""Construye la base vectorial indexando el dataset con extract_embedding (Etapa 1).

Requiere haber implementado SimilarityService.extract_embedding (o, si
EMBEDDING_MODEL != baseline, ClassifierService.extract_custom_embedding).
Si USE_PGVECTOR=true, la base de datos debe estar corriendo
(`docker compose up postgres -d`).

Uso:
    python scripts/build_index.py [--split train] [--limit 0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4
from lib.bootstrap import build_classifier, build_similarity, build_store
from lib.config import settings
from lib.schemas import EmbeddingRecord

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _store_relative_path(image_path: Path, data_root: Path) -> str:
    resolved = image_path.resolve()
    try:
        return resolved.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _normalize_indexed_path(raw_path: str, data_root: Path) -> str:
    candidate = Path(str(raw_path).replace("\\", "/"))
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            return str(candidate.resolve())
    if candidate.parts and candidate.parts[0] == data_root.name:
        candidate = Path(*candidate.parts[1:])
    return candidate.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Subcarpeta del dataset a indexar (train/valid/test).")
    parser.add_argument("--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas).")
    args = parser.parse_args()

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    root = settings.dataset_path / args.split
    if not root.is_dir():
        sys.exit(f"No existe {root}. Descarga el dataset con scripts/download_dataset.py")

    current_model = settings.embedding_model
    data_root = settings.data_path
    print("Consultando la base de datos para evitar duplicados...")
    existentes = store.all()
    claves_indexadas = {
        (
            _normalize_indexed_path(record.path, data_root),
            str(record.metadata.get("model", "baseline")).strip() or "baseline",
        )
        for record in existentes
    }
    ya_cargadas_modelo = sum(1 for _, model in claves_indexadas if model == current_model)
    print(f"Se encontraron {ya_cargadas_modelo} imagenes ya cargadas para el modelo {current_model}.")
    soporta_bulk = hasattr(store, "extend") and callable(getattr(store, "extend"))
    nuevos_registros: list[EmbeddingRecord] = []

    total = 0
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = 0
        saltadas = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit and count >= args.limit:
                break
            current_path = str(image_path)
            stored_path = _store_relative_path(image_path, data_root)

            current_key = (stored_path, current_model)
            if current_key in claves_indexadas:
                saltadas += 1
                continue

            metadata = {"split": args.split, "model": current_model}
            image = similarity._load_image(current_path)
            embedding = similarity.extract_embedding(image)
            record = EmbeddingRecord(
                id_imagen=str(uuid4()),
                embedding=embedding,
                path=stored_path,
                breed=breed_dir.name,
                metadata=metadata,
            )
            if soporta_bulk:
                nuevos_registros.append(record)
            else:
                store.append(record)
            claves_indexadas.add(current_key)
            count += 1
            total += 1
        print(f"{breed_dir.name}: {count} nuevas indexadas (omitidas: {saltadas})")

    if soporta_bulk and nuevos_registros:
        store.extend(nuevos_registros)

    print(f"Total: {total} embeddings NUEVOS almacenados (modelo: {current_model})")

if __name__ == "__main__":
    main()
