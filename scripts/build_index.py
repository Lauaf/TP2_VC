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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", help="Subcarpeta del dataset a indexar (train/valid/test).")
    parser.add_argument("--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas).")
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity, build_store
    from lib.config import settings

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    root = ROOT / "data" / "dataset" / args.split
    if not root.is_dir():
        sys.exit(f"No existe {root}. Descarga el dataset con scripts/download_dataset.py")

    print("Consultando la base de datos para evitar duplicados...")
    existentes = store.all()
    paths_indexados = {record.path for record in existentes}
    print(f"Se encontraron {len(paths_indexados)} imagenes ya cargadas en la base.")

    total = 0
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = 0
        saltadas = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit and (count + saltadas) >= args.limit:
                break
            current_path = str(image_path)
            
            if current_path in paths_indexados:
                saltadas += 1
                continue

            similarity.index_image(
                current_path,
                breed_dir.name,
                {"split": args.split, "model": settings.embedding_model},
            )
            count += 1
            total += 1
        print(f"{breed_dir.name}: {count} nuevas indexadas (Omitidas: {saltadas})")

    print(f"Total: {total} embeddings NUEVOS almacenados (modelo: {settings.embedding_model})")

if __name__ == "__main__":
    main()