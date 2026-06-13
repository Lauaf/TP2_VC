from __future__ import annotations

import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

from lib.evaluation.metrics import specificity as specificity_score

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetBreedClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)
        self.embedding_dim = in_features

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        if return_features:
            return features
        return self.classifier(features)


class CustomCNNClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(256, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(512, num_classes)
        self.embedding_dim = 512

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        features = torch.relu(self.embedding(x))
        if return_features:
            return features
        return self.classifier(self.dropout(features))


@dataclass
class TorchInferenceBundle:
    model: nn.Module
    class_names: list[str]


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        if self.active_checkpoint.suffix.lower() != ".pth":
            raise ValueError(
                "train_classifier guarda checkpoints PyTorch (.pth). "
                "Usa una ruta .pth y exporta a ONNX aparte si lo necesitas."
            )

        self._set_random_seed()
        train_dataset = self._build_dataset("train", train=True)
        valid_dataset = self._build_dataset("valid", train=False)
        self._ensure_compatible_classes(train_dataset.classes, valid_dataset.classes)

        device = self._device()
        batch_size = self._env_int("TRAIN_BATCH_SIZE", 32)
        epochs = self._default_epochs()
        learning_rate = self._default_learning_rate()
        weight_decay = self._env_float("TRAIN_WEIGHT_DECAY", 1e-4)
        num_workers = self._env_int("TRAIN_NUM_WORKERS", 0)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        model = self._build_torch_model(
            self.active_model_name,
            num_classes=len(train_dataset.classes),
            pretrained=self.active_model_name == "resnet18_finetuned",
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(self._env_int("TRAIN_LR_PATIENCE", 2), 1),
        )

        best_state: dict[str, Any] | None = None
        best_val_accuracy = -1.0
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            train_loss, train_accuracy = self._run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                training=True,
            )
            valid_loss, valid_accuracy = self._run_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                training=False,
            )
            scheduler.step(valid_accuracy)

            epoch_summary = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "valid_loss": float(valid_loss),
                "valid_accuracy": float(valid_accuracy),
            }
            history.append(epoch_summary)
            logger.info(
                "[%s] epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                self.active_model_name,
                epoch + 1,
                epochs,
                train_loss,
                train_accuracy,
                valid_loss,
                valid_accuracy,
            )

            if valid_accuracy > best_val_accuracy:
                best_val_accuracy = valid_accuracy
                best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Training finished without producing a checkpoint.")

        model.load_state_dict(best_state)
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "model_name": self.active_model_name,
            "num_classes": len(train_dataset.classes),
            "class_names": list(train_dataset.classes),
            "image_size": self.image_size,
            "state_dict": model.state_dict(),
            "history": history,
            "best_val_accuracy": float(best_val_accuracy),
        }
        torch.save(checkpoint_payload, self.active_checkpoint)

        self.output_path.mkdir(parents=True, exist_ok=True)
        history_path = self.output_path / f"{self.active_model_name}_history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=True, indent=2), encoding="utf-8")

        self._loaded.pop(self.active_model_name, None)
        if hasattr(self, "_torch_bundles"):
            self._torch_bundles.pop(self.active_model_name, None)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en la notebook.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        test_dataset = self._build_dataset("test", train=False)
        batch_size = self._env_int("EVAL_BATCH_SIZE", self._env_int("TRAIN_BATCH_SIZE", 32))
        num_workers = self._env_int("TRAIN_NUM_WORKERS", 0)
        device = self._device()

        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        class_names = self._class_names_for_inference()
        self._ensure_compatible_classes(class_names, test_dataset.classes)

        y_true: list[int] = []
        y_pred: list[int] = []
        for inputs, labels in loader:
            logits = self._predict_logits(inputs)
            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions)

        labels_idx = list(range(len(class_names)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels_idx)
        accuracy = float(np.trace(matrix) / np.sum(matrix)) if np.sum(matrix) else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels_idx,
            average="macro",
            zero_division=0,
        )

        total = int(np.sum(matrix))
        specificities: list[float] = []
        for idx in labels_idx:
            tp = int(matrix[idx, idx])
            fp = int(matrix[:, idx].sum() - tp)
            fn = int(matrix[idx, :].sum() - tp)
            tn = total - tp - fp - fn
            specificities.append(float(specificity_score(tn, fp)))
        specificity = float(np.mean(specificities)) if specificities else 0.0

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(specificity, 4),
            "f1": round(float(f1), 4),
        }

        self.output_path.mkdir(parents=True, exist_ok=True)
        metrics_path = self.output_path / f"{self.active_model_name}_metrics.json"
        payload = {
            "model": self.active_model_name,
            "metrics": metrics,
            "class_names": class_names,
            "confusion_matrix": matrix.tolist(),
        }
        metrics_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        if image is None or image.size == 0:
            raise ValueError("Image is empty; cannot extract embedding.")

        batch = self._preprocess_bgr(image)
        loaded = self.load_model()

        if isinstance(loaded, onnxruntime.InferenceSession):
            outputs = self._run_onnx(loaded, batch)
            output_names = [item.name for item in loaded.get_outputs()]
            features = None
            for name, output in zip(output_names, outputs):
                lowered = name.lower()
                if "embed" in lowered or "feature" in lowered:
                    features = output
                    break
            if features is None:
                features = outputs[0]
            vector = np.asarray(features, dtype=np.float32).reshape(-1)
        else:
            bundle = self._torch_bundle()
            device = self._device()
            with torch.inference_mode():
                features = self._forward_features(bundle.model, batch.to(device))
            vector = np.asarray(features.squeeze(0).detach().cpu().numpy(), dtype=np.float32)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    # ------------------------------------------------------------------
    # Helpers de entrenamiento / inferencia
    # ------------------------------------------------------------------

    def _predict_from_bgr(self, image: np.ndarray) -> tuple[str, float]:
        if image is None or image.size == 0:
            return "unknown", 0.0

        batch = self._preprocess_bgr(image)
        class_names = self._class_names_for_inference()
        logits = self._predict_logits(batch)
        probabilities = torch.softmax(logits, dim=1)
        score, index = torch.max(probabilities, dim=1)
        predicted_idx = int(index.item())
        breed = class_names[predicted_idx] if predicted_idx < len(class_names) else str(predicted_idx)
        return breed, float(score.item())

    def _predict_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        loaded = self.load_model()
        if isinstance(loaded, onnxruntime.InferenceSession):
            outputs = self._run_onnx(loaded, inputs.cpu())
            output_names = [item.name for item in loaded.get_outputs()]
            logits_output = None
            for name, output in zip(output_names, outputs):
                lowered = name.lower()
                if "logit" in lowered or lowered.endswith("output") or "prob" in lowered:
                    logits_output = output
                    break
            if logits_output is None:
                for name, output in zip(output_names, outputs):
                    lowered = name.lower()
                    if "embed" not in lowered and "feature" not in lowered:
                        logits_output = output
                        break
            if logits_output is None:
                logits_output = outputs[0]
            logits = np.asarray(logits_output, dtype=np.float32)
            if logits.ndim == 1:
                logits = np.expand_dims(logits, axis=0)
            return torch.from_numpy(logits)

        bundle = self._torch_bundle()
        device = self._device()
        with torch.inference_mode():
            logits = self._forward_logits(bundle.model, inputs.to(device))
        return logits.cpu()

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
        training: bool,
    ) -> tuple[float, float]:
        model.train(mode=training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                logits = self._forward_logits(model, inputs)
                loss = criterion(logits, labels)
                if training and optimizer is not None:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            predictions = torch.argmax(logits, dim=1)
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, total_correct / total_samples

    def _forward_logits(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        outputs = model(inputs)
        if isinstance(outputs, dict):
            if "logits" in outputs:
                outputs = outputs["logits"]
            else:
                outputs = next(iter(outputs.values()))
        elif isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        return outputs

    def _forward_features(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        try:
            features = model(inputs, return_features=True)
        except TypeError:
            if hasattr(model, "backbone"):
                features = model.backbone(inputs)
            elif hasattr(model, "features") and hasattr(model, "embedding"):
                maps = model.features(inputs)
                maps = torch.flatten(maps, 1)
                features = torch.relu(model.embedding(maps))
            elif hasattr(model, "fc"):
                backbone = nn.Sequential(*list(model.children())[:-1]).to(inputs.device)
                features = torch.flatten(backbone(inputs), 1)
            else:
                features = self._forward_logits(model, inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features

    def _torch_bundle(self, name: str | None = None) -> TorchInferenceBundle:
        key = name or self.active_model_name
        bundles: dict[str, TorchInferenceBundle] = getattr(self, "_torch_bundles", {})
        if key in bundles:
            return bundles[key]

        loaded = self.load_model(key)
        class_names = self._extract_class_names(loaded)
        if isinstance(loaded, dict):
            model_name = str(loaded.get("model_name") or key)
            num_classes = int(loaded.get("num_classes") or len(class_names))
            state_dict = loaded.get("state_dict") or loaded.get("model_state_dict")
            if state_dict is None:
                raise ValueError(f"Checkpoint {self.checkpoints[key]} has no state_dict.")
            model = self._build_torch_model(model_name, num_classes=num_classes, pretrained=False)
            model.load_state_dict(state_dict)
        elif isinstance(loaded, nn.Module):
            model = loaded
        else:
            raise TypeError(f"Model '{key}' is not a PyTorch checkpoint.")

        model.eval()
        model.to(self._device())
        bundle = TorchInferenceBundle(model=model, class_names=class_names)
        bundles[key] = bundle
        self._torch_bundles = bundles
        return bundle

    def _build_torch_model(self, name: str, num_classes: int, pretrained: bool) -> nn.Module:
        if name == "resnet18_finetuned":
            return ResNetBreedClassifier(num_classes=num_classes, pretrained=pretrained)
        if name == "cnn_custom":
            return CustomCNNClassifier(num_classes=num_classes)
        raise ValueError(f"Unsupported classifier architecture '{name}'.")

    def _build_dataset(self, split: str, train: bool) -> datasets.ImageFolder:
        directory = self.dataset_path / split
        if not directory.is_dir():
            raise ValueError(
                f"Dataset split not found: {directory}. "
                "Descarga/descomprime el dataset en data/dataset con las carpetas train/valid/test."
            )
        return datasets.ImageFolder(directory, transform=self._transform(train=train))

    def _transform(self, train: bool) -> transforms.Compose:
        if train:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(12),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _preprocess_bgr(self, image: np.ndarray) -> torch.Tensor:
        from PIL import Image

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._transform(train=False)(pil).unsqueeze(0)
        return tensor.to(torch.float32)

    def _run_onnx(self, session: onnxruntime.InferenceSession, batch: torch.Tensor) -> list[np.ndarray]:
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: batch.cpu().numpy()})

    def _extract_class_names(self, loaded: Any) -> list[str]:
        if isinstance(loaded, dict):
            for key in ("class_names", "classes"):
                value = loaded.get(key)
                if isinstance(value, list) and value:
                    return [str(item) for item in value]
            idx_to_class = loaded.get("idx_to_class")
            if isinstance(idx_to_class, dict) and idx_to_class:
                return [str(name) for _, name in sorted(idx_to_class.items(), key=lambda item: int(item[0]))]
            class_to_idx = loaded.get("class_to_idx")
            if isinstance(class_to_idx, dict) and class_to_idx:
                return [str(name) for name, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))]
        if isinstance(loaded, nn.Module):
            value = getattr(loaded, "class_names", None)
            if isinstance(value, list) and value:
                return [str(item) for item in value]
        return self._class_names_from_dataset()

    def _class_names_for_inference(self) -> list[str]:
        loaded = self.load_model()
        if isinstance(loaded, onnxruntime.InferenceSession):
            return self._onnx_class_names(self.active_checkpoint)
        return self._torch_bundle().class_names

    def _class_names_from_dataset(self) -> list[str]:
        train_dir = self.dataset_path / "train"
        if not train_dir.is_dir():
            raise ValueError(
                f"Cannot infer class names because {train_dir} does not exist. "
                "Descarga el dataset o usa checkpoints con metadata de clases."
            )
        return sorted(item.name for item in train_dir.iterdir() if item.is_dir())

    def _onnx_class_names(self, checkpoint_path: Path) -> list[str]:
        for candidate in (
            checkpoint_path.with_suffix(".json"),
            checkpoint_path.with_name(f"{checkpoint_path.stem}.meta.json"),
            checkpoint_path.with_name(f"{checkpoint_path.stem}.labels.json"),
        ):
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for key in ("class_names", "classes"):
                value = payload.get(key)
                if isinstance(value, list) and value:
                    return [str(item) for item in value]
        return self._class_names_from_dataset()

    @staticmethod
    def _ensure_compatible_classes(expected: list[str], observed: list[str]) -> None:
        if list(expected) != list(observed):
            raise ValueError(
                "Class ordering differs between splits/checkpoint and dataset. "
                "Verifica que train/valid/test tengan las mismas carpetas de razas."
            )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        return int(os.getenv(name, default))

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        return float(os.getenv(name, default))

    def _default_learning_rate(self) -> float:
        if self.active_model_name == "resnet18_finetuned":
            return self._env_float("TRAIN_LR", 3e-4)
        return self._env_float("TRAIN_LR", 1e-3)

    def _default_epochs(self) -> int:
        if self.active_model_name == "resnet18_finetuned":
            return self._env_int("TRAIN_EPOCHS", 6)
        return self._env_int("TRAIN_EPOCHS", 10)

    @staticmethod
    def _device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _set_random_seed() -> None:
        seed = int(os.getenv("TRAIN_SEED", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
