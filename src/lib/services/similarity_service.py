from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ej: ResNet50, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.

        Sugerencias:
          - Preprocesar la imagen (resize a self.image_size, normalizacion ImageNet).
          - Usar torchvision.models o timm con pesos pre-entrenados.
          - Recordar que la imagen llega en BGR (OpenCV).
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        if image is None or image.size == 0:
            raise ValueError("Image is empty; cannot extract embedding.")

        model, preprocess, device, torch = self._baseline_components()
        pil_image = self._to_pil_rgb(image)
        tensor = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.inference_mode():
            features = model(tensor)

        embedding = np.asarray(features.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding /= norm
        return embedding.tolist()

    def search_similar_images(
        self,
        embedding: list[float],
        top_k: int,
        model_name: Optional[str] = None,
    ) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        Sugerencias:
          - Con pgvector: self.store.search(embedding, top_k).
          - Con JSON: iterar self.store.all() y usar self.similarity(...).
          - Respetar SIMILARITY_METRIC (cosine | l2).
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        query = list(embedding)
        k = max(int(top_k), 1)
        target_model = model_name or self.model_name
        can_use_store_search = (
            hasattr(self.store, "search")
            and callable(getattr(self.store, "search"))
            and self.similarity_metric.lower() != "l2"
        )
        if can_use_store_search:
            search_fn = getattr(self.store, "search")
            try:
                records = list(search_fn(query, k, target_model))
            except TypeError:
                records = list(search_fn(query, k))
        else:
            records = self._records_for_model(target_model)

        neighbors = []
        for record in records:
            neighbors.append(
                Neighbor(
                    path=record.path,
                    breed=record.breed,
                    score=float(self.similarity(query, record.embedding)),
                )
            )

        neighbors.sort(key=lambda item: item.score, reverse=True)
        return neighbors[:k]

    def _records_for_model(self, model_name: str) -> list[EmbeddingRecord]:
        records = list(self.store.all())
        matching_records = [
            record
            for record in records
            if str(record.metadata.get("model", "")).strip() == model_name
        ]
        if matching_records:
            return matching_records

        legacy_records = [record for record in records if "model" not in record.metadata]
        if legacy_records:
            logger.warning(
                "No indexed records tagged with model '%s'; using %d legacy records without model metadata.",
                model_name,
                len(legacy_records),
            )
            return legacy_records

        logger.warning("No indexed records found for embedding model '%s'.", model_name)
        return []

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados (ej: voto
        mayoritario, opcionalmente ponderado por score).

        Si el mejor score esta por debajo de self.similarity_threshold se
        considera "unknown". Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0

        best_score = float(results[0].score)
        if best_score < self.similarity_threshold:
            return "unknown", best_score

        votes: dict[str, float] = {}
        for rank, neighbor in enumerate(results, start=1):
            weight = max(float(neighbor.score), 0.0) / rank
            votes[neighbor.breed] = votes.get(neighbor.breed, 0.0) + weight

        if not votes:
            return "unknown", best_score

        breed = max(votes.items(), key=lambda item: item[1])[0]
        breed_score = max(float(item.score) for item in results if item.breed == breed)
        return breed, breed_score

    def _baseline_components(self) -> tuple[Any, Any, Any, Any]:
        if not hasattr(self, "_baseline_model"):
            import torch
            from torchvision import transforms
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = resnet18(weights=weights)
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(device)

            preprocess = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

            self._baseline_model = model
            self._baseline_preprocess = preprocess
            self._baseline_device = device
            self._baseline_torch = torch

        return (
            self._baseline_model,
            self._baseline_preprocess,
            self._baseline_device,
            self._baseline_torch,
        )

    @staticmethod
    def _to_pil_rgb(image: np.ndarray):
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        effective_model_name = model_name or self.model_name
        neighbors = [
            self._with_url(n)
            for n in self.search_similar_images(embedding, k, effective_model_name)
        ]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=effective_model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
