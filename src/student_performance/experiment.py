"""Reproducible model comparison, fairness audit, and research report generator."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "student-performance-matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import canonicalize_dataframe, load_csv, validate_training_data
from .model import build_ann
from .preprocessing import SafePreprocessor
from .schema import FEATURES


def _metrics(actual: np.ndarray, labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(actual, labels)),
        "precision": float(precision_score(actual, labels, zero_division=0)),
        "recall": float(recall_score(actual, labels, zero_division=0)),
        "f1_score": float(f1_score(actual, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "confusion_matrix": confusion_matrix(actual, labels, labels=[0, 1]).tolist(),
    }


def _fairness_rows(model: str, actual: np.ndarray, labels: np.ndarray, frame, test_idx: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in ("gender", "family_support"):
        groups = frame.iloc[test_idx][feature].astype(str).to_numpy()
        for group in sorted(set(groups)):
            mask = groups == group
            group_actual, group_labels = actual[mask], labels[mask]
            matrix = confusion_matrix(group_actual, group_labels, labels=[0, 1])
            tn, fp, fn, tp = matrix.ravel()
            rows.append({
                "model": model, "sensitive_feature": feature, "group": group, "sample_size": int(mask.sum()),
                "accuracy": float(accuracy_score(group_actual, group_labels)),
                "positive_prediction_rate": float(group_labels.mean()),
                "recall": float(tp / (tp + fn)) if tp + fn else None,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
            })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(results: dict[str, dict[str, Any]], output: Path) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    models = list(results)
    x = np.arange(len(metric_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(11, 5.5))
    colors = ["#12815a", "#dcae38", "#315b7d"]
    for index, model in enumerate(models):
        axis.bar(x + (index - 1) * width, [results[model][metric] for metric in metric_names], width, label=model, color=colors[index])
    axis.set_ylim(0, 1.05)
    axis.set_xticks(x, [name.replace("_", " ").title() for name in metric_names])
    axis.set_ylabel("Score")
    axis.set_title("UCI Student Performance — Held-out Model Comparison")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_confusions(results: dict[str, dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (model, metrics) in zip(axes, results.items(), strict=True):
        matrix = np.asarray(metrics["confusion_matrix"])
        axis.imshow(matrix, cmap="Greens", vmin=0, vmax=max(1, matrix.max()))
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontweight="bold")
        axis.set(title=model, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
    figure.suptitle("Confusion Matrices on the Shared UCI Test Split")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_fairness(rows: list[dict[str, Any]], output: Path) -> None:
    ann = [row for row in rows if row["model"] == "ANN"]
    labels = [f"{row['sensitive_feature']}={row['group']}" for row in ann]
    rates = [row["positive_prediction_rate"] for row in ann]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, rates, color=["#12815a", "#6eb58f", "#dcae38", "#eed684"][:len(labels)])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Positive prediction rate")
    axis.set_title("ANN Group Prediction-Rate Audit (Descriptive, Not Causal)")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run_experiment(data_path: str, output_path: str, dataset_name: str, threshold: float = 10, seed: int = 42, epochs: int = 100) -> dict[str, Any]:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    raw = load_csv(data_path)
    frame, missing = canonicalize_dataframe(raw)
    frame = validate_training_data(frame)
    features = [name for name in FEATURES if name not in missing]
    target = (frame["final_grade"].to_numpy(dtype=float) >= threshold).astype(int)
    counts = np.bincount(target, minlength=2)
    if counts.min() < 10:
        raise ValueError("Experiment requires at least ten pass and fail examples")
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=target)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed, stratify=target[train_idx])
    processor = SafePreprocessor(features)
    x_train = processor.fit_transform(frame.iloc[train_idx])
    x_val = processor.transform(frame.iloc[val_idx])
    x_test = processor.transform(frame.iloc[test_idx])

    logistic = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1)
    logistic.fit(x_train, target[train_idx])
    forest.fit(x_train, target[train_idx])

    ann = build_ann(processor.output_size, "classification")
    train_counts = np.bincount(target[train_idx], minlength=2)
    class_weight = {index: len(train_idx) / (2 * count) for index, count in enumerate(train_counts)}
    ann.fit(x_train, target[train_idx], validation_data=(x_val, target[val_idx]), epochs=epochs, batch_size=32,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)],
            class_weight=class_weight, verbose=0)

    probabilities = {
        "ANN": ann.predict(x_test, verbose=0).reshape(-1),
        "Logistic Regression": logistic.predict_proba(x_test)[:, 1],
        "Random Forest": forest.predict_proba(x_test)[:, 1],
    }
    results: dict[str, dict[str, Any]] = {}
    fairness: list[dict[str, Any]] = []
    for name, scores in probabilities.items():
        labels = (scores >= 0.5).astype(int)
        results[name] = _metrics(target[test_idx], labels, scores)
        fairness.extend(_fairness_rows(name, target[test_idx], labels, frame, test_idx))

    comparison = [{"model": model, **{key: value for key, value in metrics.items() if key != "confusion_matrix"}} for model, metrics in results.items()]
    fairness_gaps: dict[str, dict[str, float]] = {}
    for model in results:
        fairness_gaps[model] = {}
        for feature in ("gender", "family_support"):
            group_rows = [row for row in fairness if row["model"] == model and row["sensitive_feature"] == feature]
            for metric in ("accuracy", "positive_prediction_rate", "recall", "false_positive_rate", "false_negative_rate"):
                values = [row[metric] for row in group_rows if row[metric] is not None]
                fairness_gaps[model][f"{feature}_{metric}_gap"] = round(max(values) - min(values), 5) if len(values) > 1 else 0.0

    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "model_comparison.csv", comparison)
    _write_csv(output / "fairness_audit.csv", fairness)
    _plot_comparison(results, output / "model_comparison.png")
    _plot_confusions(results, output / "confusion_matrices.png")
    _plot_fairness(fairness, output / "fairness_audit.png")
    best_model = max(results, key=lambda name: results[name]["f1_score"])
    summary = {
        "format_version": 1, "dataset_name": dataset_name, "source_rows": len(frame), "features": features,
        "dropped_unavailable_features": missing, "pass_threshold": threshold, "split_seed": seed,
        "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows": len(test_idx),
        "class_distribution": {"fail": int(counts[0]), "pass": int(counts[1])}, "models": results,
        "best_model_by_f1": best_model, "fairness_gaps": fairness_gaps,
        "mtu_validation_status": "awaiting_approved_dataset" if "UCI" in dataset_name.upper() else "completed_on_approved_dataset",
    }
    (output / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ann_metrics = results["ANN"]
    discussion = f"""# Experiment Results and Discussion

## Dataset and protocol

The experiment used **{dataset_name}** with {len(frame)} usable records. A fixed, stratified split produced {len(train_idx)} training, {len(val_idx)} validation, and {len(test_idx)} held-out test records. A final grade of {threshold:g} or above was classified as pass. The same split and training-only preprocessing were used for every model.

## Model comparison

The strongest held-out F1-score was produced by **{best_model}**. The ANN achieved accuracy {ann_metrics['accuracy']:.3f}, precision {ann_metrics['precision']:.3f}, recall {ann_metrics['recall']:.3f}, F1 {ann_metrics['f1_score']:.3f}, and ROC-AUC {ann_metrics['roc_auc']:.3f}. These scores describe this held-out UCI split and must not be presented as validated MTU performance.

## Bias audit

Gender and family-support groups were audited for accuracy, positive prediction rate, recall, false-positive rate, and false-negative rate. Differences are descriptive signals, not proof of discrimination or causation. Small test-group sizes make the estimates uncertain. Any material gap should trigger review, repeated cross-validation, confidence intervals, consultation with affected groups, and consideration of excluding sensitive features.

## Limitations and MTU transfer

The UCI records concern Portuguese secondary schools, not CEIT university students in Myanmar. UCI lacks real attendance and homework-completion measurements; attendance is estimated from absences and unavailable features are excluded. G2 is temporally close to G3 and is highly predictive, so results can appear stronger than an earlier-warning deployment. MTU validation remains pending until an approved, anonymized CEIT dataset is supplied. No institutional decision should rely on this experiment alone.
"""
    (output / "discussion.md").write_text(discussion, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/experiments/uci")
    parser.add_argument("--dataset-name", default="UCI Student Performance — Mathematics")
    parser.add_argument("--pass-threshold", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    summary = run_experiment(args.data, args.output, args.dataset_name, args.pass_threshold, args.seed, args.epochs)
    print(json.dumps({"best_model_by_f1": summary["best_model_by_f1"], "models": summary["models"]}, indent=2))


if __name__ == "__main__":
    main()
