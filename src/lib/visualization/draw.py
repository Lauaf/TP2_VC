"""Herramientas de visualizacion (provistas por la catedra)."""
from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

_BOX_COLOR_BGR = (80, 220, 80)
_TEXT_COLOR_BGR = (80, 220, 80)


def _get(det: Any, key: str, default: Any = None) -> Any:
    """Soporta detecciones como dicts o como modelos pydantic."""
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


def draw_detections(image_bgr: np.ndarray, detections: Iterable[Any]) -> np.ndarray:
    """Dibuja bounding boxes, raza predicha y scores sobre una copia de la imagen.

    Cada deteccion debe tener: bbox [x1,y1,x2,y2], breed, det_score, breed_score
    (formato de lib.schemas.DogDetection).
    Retorna la imagen anotada en BGR.
    """
    vis = image_bgr.copy()
    for det in detections:
        bbox = _get(det, "bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        breed = _get(det, "breed", "?")
        det_score = float(_get(det, "det_score", 0.0) or 0.0)
        breed_score = float(_get(det, "breed_score", 0.0) or 0.0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), _BOX_COLOR_BGR, 2)
        label = f"{breed} det:{det_score:.2f} cls:{breed_score:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _TEXT_COLOR_BGR,
            2,
        )
    return vis


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte BGR (OpenCV) a RGB (matplotlib / gradio)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
