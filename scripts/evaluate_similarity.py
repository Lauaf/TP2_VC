"""Evalua busqueda por similitud sobre un subconjunto del dataset.

Uso:
    python scripts/evaluate_similarity.py --model baseline --limit-per-class 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="baseline",
        choices=("baseline", "resnet18_finetuned", "cnn_custom"),
        help="Modelo de embeddings a evaluar.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=("train", "valid", "test"),
        help="Split a usar como consultas.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=3,
        help="Cantidad maxima de imagenes por raza para evaluar.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Cantidad de vecinos usados en la prediccion.",
    )
    parser.add_argument(
        "--embeddings-path",
        default="",
        help="Ruta opcional al store JSON de embeddings. Si se omite usa la configuracion actual.",
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity
    from lib.config import settings
    from lib.storage.embedding_store import EmbeddingStore

    if args.embeddings_path:
        candidate = Path(args.embeddings_path)
        embeddings_path = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
    else:
        embeddings_path = settings.embeddings_path
    store = EmbeddingStore(embeddings_path)
    similarity = build_similarity(settings, store)
    similarity.model_name = args.model

    if args.model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(args.model)
        extractor = classifier.extract_custom_embedding

    query_root = settings.dataset_path / args.split
    rows: list[dict[str, object]] = []
    total = 0
    correct = 0
    per_class_total: Counter[str] = Counter()
    per_class_correct: Counter[str] = Counter()

    for breed_dir in sorted(p for p in query_root.iterdir() if p.is_dir()):
        breed = breed_dir.name
        seen = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit_per_class > 0 and seen >= args.limit_per_class:
                break
            image = similarity._load_image(str(image_path))
            embedding = extractor(image)
            neighbors = similarity.search_similar_images(embedding, args.top_k, args.model)
            predicted_breed, score = similarity.predict_breed_from_neighbors(neighbors)
            is_correct = predicted_breed == breed
            rows.append(
                {
                    "path": str(image_path),
                    "true_breed": breed,
                    "predicted_breed": predicted_breed,
                    "score": round(float(score), 6),
                    "correct": is_correct,
                }
            )
            total += 1
            correct += int(is_correct)
            per_class_total[breed] += 1
            per_class_correct[breed] += int(is_correct)
            seen += 1

    accuracy = (correct / total) if total else 0.0
    per_class_accuracy = {
        breed: round(per_class_correct[breed] / count, 4)
        for breed, count in sorted(per_class_total.items())
        if count
    }
    payload = {
        "model": args.model,
        "split": args.split,
        "top_k": args.top_k,
        "limit_per_class": args.limit_per_class,
        "embeddings_path": str(embeddings_path),
        "queries": total,
        "correct": correct,
        "accuracy": round(float(accuracy), 4),
        "per_class_accuracy": per_class_accuracy,
        "predictions": rows,
    }

    output_path = settings.output_path / f"{args.model}_similarity_metrics.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("model", "queries", "correct", "accuracy")}, ensure_ascii=False, indent=2))
    print(f"Detailed results written to: {output_path}")


if __name__ == "__main__":
    main()
