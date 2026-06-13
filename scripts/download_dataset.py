"""Descarga el dataset `70 Dog Breeds Image Dataset` (Kaggle) en data/dataset.

Requiere kagglehub (incluido en requirements.txt). Alternativamente, descargar
manualmente desde:
https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
y descomprimir el contenido (train/ valid/ test/) dentro de data/dataset.

Uso:
    python scripts/download_dataset.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "dataset"

KAGGLE_DATASET = "gpiosenka/70-dog-breedsimage-data-set"


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub no esta instalado. Instala las dependencias (requirements.txt) "
            "o descarga el dataset manualmente desde Kaggle."
        )

    print(f"Descargando {KAGGLE_DATASET} ...")
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    DEST.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = DEST / item.name
        if target.exists():
            print(f"Ya existe, se omite: {target}")
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"Dataset disponible en {DEST}")


if __name__ == "__main__":
    main()
