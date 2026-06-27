"""Genera plots y un resumen para el informe a partir de los JSON de salida."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
ASSETS = ROOT / "report_assets"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_class_distribution() -> None:
    dataset = ROOT / "data" / "dataset" / "train"
    counts = sorted(
        [(d.name, len(list(d.iterdir()))) for d in dataset.iterdir() if d.is_dir()],
        key=lambda x: x[1],
    )
    names = [c[0] for c in counts]
    values = [c[1] for c in counts]
    avg = sum(values) / len(values)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(names, values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(y=avg, color="red", linestyle="--", linewidth=1.2, label=f"Promedio ({avg:.1f})")
    ax.set_xlabel("Raza", fontsize=11)
    ax.set_ylabel("Cantidad de imágenes", fontsize=11)
    ax.set_title("Distribución de imágenes por raza — split train", fontsize=13)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_history(resnet_history: dict, cnn_history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for payload, label, color in (
        (resnet_history, "ResNet18 fine-tuned", "#1f77b4"),
        (cnn_history, "CNN custom", "#d62728"),
    ):
        history = payload["history"]
        epochs = [int(item["epoch"]) for item in history]
        train_acc = [item["train_accuracy"] for item in history]
        valid_acc = [item["valid_accuracy"] for item in history]
        train_loss = [item["train_loss"] for item in history]
        valid_loss = [item["valid_loss"] for item in history]

        axes[0].plot(epochs, train_acc, label=f"{label} train", color=color, linestyle="-")
        axes[0].plot(epochs, valid_acc, label=f"{label} valid", color=color, linestyle="--")
        axes[1].plot(epochs, train_loss, label=f"{label} train", color=color, linestyle="-")
        axes[1].plot(epochs, valid_loss, label=f"{label} valid", color=color, linestyle="--")

    axes[0].set_title("Accuracy por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Loss por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS / "classifier_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_classifier_metrics(resnet_metrics: dict, cnn_metrics: dict) -> None:
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    resnet_values = [resnet_metrics["metrics"][key] for key in metric_names]
    cnn_values = [cnn_metrics["metrics"][key] for key in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, resnet_values, width=width, label="ResNet18 fine-tuned", color="#1f77b4")
    ax.bar(x + width / 2, cnn_values, width=width, label="CNN custom", color="#d62728")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Comparacion de clasificadores (test)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for offset, values in ((-width / 2, resnet_values), (width / 2, cnn_values)):
        for idx, value in enumerate(values):
            ax.text(idx + offset, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS / "classifier_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_metrics(baseline_metrics: dict, resnet_metrics: dict, cnn_metrics: dict) -> None:
    labels = ["Baseline", "ResNet18 fine-tuned", "CNN custom"]
    values = [
        baseline_metrics["accuracy"],
        resnet_metrics["accuracy"],
        cnn_metrics["accuracy"],
    ]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Etapa 1: accuracy de similitud (valid, 3 imagenes/raza)")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values, strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(ASSETS / "similarity_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(payload: dict, output_name: str, title: str) -> None:
    matrix = np.asarray(payload["confusion_matrix"], dtype=np.float32)
    class_names = payload["class_names"]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Clase real")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=4, rotation=90)
    ax.set_yticklabels(class_names, fontsize=4)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSETS / output_name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    baseline_similarity: dict,
    resnet_similarity: dict,
    cnn_similarity: dict,
    resnet_metrics: dict,
    cnn_metrics: dict,
) -> None:
    summary = {
        "classification_test": {
            "resnet18_finetuned": resnet_metrics["metrics"],
            "cnn_custom": cnn_metrics["metrics"],
        },
        "similarity_valid": {
            "baseline": {
                "accuracy": baseline_similarity["accuracy"],
                "queries": baseline_similarity["queries"],
            },
            "resnet18_finetuned": {
                "accuracy": resnet_similarity["accuracy"],
                "queries": resnet_similarity["queries"],
            },
            "cnn_custom": {
                "accuracy": cnn_similarity["accuracy"],
                "queries": cnn_similarity["queries"],
            },
        },
    }
    (ASSETS / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    resnet_history = _load_json(OUTPUT / "resnet18_finetuned_history.json")
    cnn_history = _load_json(OUTPUT / "cnn_custom_history.json")
    resnet_metrics = _load_json(OUTPUT / "resnet18_finetuned_metrics.json")
    cnn_metrics = _load_json(OUTPUT / "cnn_custom_metrics.json")
    baseline_similarity = _load_json(OUTPUT / "baseline_similarity_metrics.json")
    resnet_similarity = _load_json(OUTPUT / "resnet18_finetuned_similarity_metrics.json")
    cnn_similarity = _load_json(OUTPUT / "cnn_custom_similarity_metrics.json")

    _plot_class_distribution()
    _plot_history(resnet_history, cnn_history)
    _plot_classifier_metrics(resnet_metrics, cnn_metrics)
    _plot_similarity_metrics(baseline_similarity, resnet_similarity, cnn_similarity)
    _plot_confusion_matrix(
        resnet_metrics,
        "resnet_confusion_matrix.png",
        "ResNet18 fine-tuned - matriz de confusion normalizada",
    )
    _plot_confusion_matrix(
        cnn_metrics,
        "cnn_confusion_matrix.png",
        "CNN custom - matriz de confusion normalizada",
    )
    _write_summary(
        baseline_similarity,
        resnet_similarity,
        cnn_similarity,
        resnet_metrics,
        cnn_metrics,
    )
    print(f"Assets written to: {ASSETS}")


if __name__ == "__main__":
    main()