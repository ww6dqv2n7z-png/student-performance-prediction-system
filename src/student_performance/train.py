"""Train, evaluate, and safely persist the student-performance ANN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "student-performance-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .data import canonicalize_dataframe, load_csv, validate_training_data
from .model import build_ann
from .preprocessing import SafePreprocessor
from .schema import FEATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _plot_history(history: dict[str, list[float]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    metric = next((key for key in history if key not in {"loss", "val_loss"} and not key.startswith("val_")), None)
    if metric:
        axes[1].plot(history[metric], label="train")
        axes[1].plot(history[f"val_{metric}"], label="validation")
        axes[1].set(title=metric.upper(), xlabel="Epoch", ylabel=metric)
        axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_confusion(matrix: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    raw = load_csv(args.data)
    canonical, missing = canonicalize_dataframe(raw, school_days=args.school_days)
    clean = validate_training_data(canonical)
    features = [name for name in FEATURES if name not in missing]
    if len(features) < 3:
        raise ValueError("Dataset must provide at least three usable input features")

    threshold = args.pass_threshold
    if threshold is None:
        threshold = 10.0 if clean["final_grade"].max() <= 20 else 50.0

    y = clean["final_grade"].to_numpy(dtype=np.float32)
    if args.task == "classification":
        y = (y >= threshold).astype(np.float32)
        counts = np.bincount(y.astype(int), minlength=2)
        if counts.min() < 5:
            raise ValueError("Classification requires at least five pass and five fail examples")

    indices = np.arange(len(clean))
    stratify = y if args.task == "classification" else None
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed, stratify=stratify)
    stratify_train = y[train_idx] if args.task == "classification" else None
    train_idx, val_idx = train_test_split(train_idx, test_size=args.validation_size, random_state=args.seed, stratify=stratify_train)

    preprocessor = SafePreprocessor(features)
    x_train = preprocessor.fit_transform(clean.iloc[train_idx])
    x_val = preprocessor.transform(clean.iloc[val_idx])
    x_test = preprocessor.transform(clean.iloc[test_idx])

    model = build_ann(preprocessor.output_size, args.task, args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(2, args.patience // 2), factor=0.5),
    ]
    class_weight = None
    if args.task == "classification":
        train_counts = np.bincount(y[train_idx].astype(int), minlength=2)
        class_weight = {index: len(train_idx) / (2 * count) for index, count in enumerate(train_counts)}
    history = model.fit(
        x_train,
        y[train_idx],
        validation_data=(x_val, y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=args.verbose,
    )
    predictions = model.predict(x_test, verbose=0).reshape(-1)

    if args.task == "classification":
        labels = (predictions >= 0.5).astype(int)
        metrics: dict[str, Any] = {
            "accuracy": accuracy_score(y[test_idx], labels),
            "precision": precision_score(y[test_idx], labels, zero_division=0),
            "recall": recall_score(y[test_idx], labels, zero_division=0),
            "f1_score": f1_score(y[test_idx], labels, zero_division=0),
            "confusion_matrix": confusion_matrix(y[test_idx], labels, labels=[0, 1]),
            "majority_baseline_accuracy": float(np.max(np.bincount(y[test_idx].astype(int))) / len(test_idx)),
        }
    else:
        error_margin = 1.0 if clean["final_grade"].max() <= 20 else 5.0
        metrics = {
            "mae": mean_absolute_error(y[test_idx], predictions),
            "rmse": mean_squared_error(y[test_idx], predictions) ** 0.5,
            "r2_score": r2_score(y[test_idx], predictions),
            "within_margin_accuracy": np.mean(np.abs(y[test_idx] - predictions) <= error_margin),
            "accuracy_margin": error_margin,
            "mean_baseline_mae": mean_absolute_error(y[test_idx], np.full(len(test_idx), y[train_idx].mean())),
        }

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.keras"
    preprocessor_path = output / "preprocessor.json"
    metadata_path = output / "metadata.json"
    model.save(model_path)
    preprocessor.save(preprocessor_path)
    serializable_metrics = {key: _json_value(value) for key, value in metrics.items()}
    metadata = {
        "format_version": 1,
        "task": args.task,
        "dataset_label": args.dataset_label,
        "synthetic_data": args.synthetic_data,
        "features": features,
        "pass_threshold": threshold if args.task == "classification" else None,
        "decision_threshold": 0.5 if args.task == "classification" else None,
        "training_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "test_rows": len(test_idx),
        "metrics": serializable_metrics,
        "seed": args.seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")
    _plot_history(history.history, output / "training_history.png")
    if args.task == "classification":
        _plot_confusion(np.asarray(metrics["confusion_matrix"]), output / "confusion_matrix.png")

    manifest = {
        "format_version": 1,
        "files": {
            "model.keras": _sha256(model_path),
            "preprocessor.json": _sha256(preprocessor_path),
            "metadata.json": _sha256(metadata_path),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification")
    parser.add_argument("--dataset-label", default="Unspecified training dataset", help="Human-readable provenance label")
    parser.add_argument("--synthetic-data", action="store_true", help="Mark artifacts as trained on synthetic data")
    parser.add_argument("--output", default="artifacts/model", help="Artifact output directory")
    parser.add_argument("--pass-threshold", type=float, default=None, help="Grade at or above which a student passes")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fraction of the non-test split")
    parser.add_argument("--school-days", type=int, default=200, help="Used to derive attendance from absences")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.05 <= args.test_size <= 0.4 or not 0.05 <= args.validation_size <= 0.4:
        raise SystemExit("Split fractions must be between 0.05 and 0.4")
    if not 1 <= args.epochs <= 10_000 or not 1 <= args.batch_size <= 4_096:
        raise SystemExit("epochs or batch size is outside the safe range")
    metadata = train(args)
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
