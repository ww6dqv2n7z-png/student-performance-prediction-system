"""Run validated predictions using locally trained artifacts."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import canonicalize_dataframe
from .preprocessing import SafePreprocessor
from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MAX_INPUT_BYTES = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _load_artifacts_cached(directory: str):
    import tensorflow as tf

    root = Path(directory)
    if not root.is_dir():
        raise ValueError("Artifact directory does not exist")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported artifact version")
    for name in ("model.keras", "preprocessor.json", "metadata.json"):
        path = root / name
        if not path.is_file() or _sha256(path) != manifest.get("files", {}).get(name):
            raise ValueError(f"Artifact integrity check failed: {name}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    preprocessor = SafePreprocessor.load(root / "preprocessor.json")
    # safe_mode blocks unsafe Lambda/object deserialization in modern Keras.
    model = tf.keras.models.load_model(root / "model.keras", safe_mode=True, compile=False)
    return model, preprocessor, metadata


def load_artifacts(directory: str | Path):
    return _load_artifacts_cached(str(Path(directory).expanduser().resolve()))


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".json":
        raise ValueError("Input must be an existing .json file")
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Prediction input exceeds the 1 MB safety limit")
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    if not records or len(records) > 10_000 or not all(isinstance(item, dict) for item in records):
        raise ValueError("Input must contain one object or a list of at most 10,000 objects")
    return records


def predict(artifact_dir: str | Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model, preprocessor, metadata = load_artifacts(artifact_dir)
    canonical, _ = canonicalize_dataframe(pd.DataFrame(records))
    for name in preprocessor.features:
        if name in NUMERIC_FEATURES:
            spec = NUMERIC_FEATURES[name]
            values = pd.to_numeric(canonical[name], errors="coerce")
            if values.isna().any() or not values.between(spec.minimum, spec.maximum).all():
                raise ValueError(f"{name} is required and must be between {spec.minimum} and {spec.maximum}")
        elif name in CATEGORICAL_FEATURES:
            values = canonical[name].astype("string").str.strip().str.lower()
            if values.isna().any() or (values == "unknown").any():
                raise ValueError(f"{name} is required")
    matrix = preprocessor.transform(canonical)
    raw = model.predict(matrix, verbose=0).reshape(-1)
    if metadata["task"] == "classification":
        threshold = float(metadata["decision_threshold"])
        return [
            {"prediction": "pass" if score >= threshold else "fail", "pass_probability": round(float(score), 6)}
            for score in raw
        ]
    return [{"predicted_final_grade": round(float(score), 4)} for score in raw]


def predict_with_explanations(artifact_dir: str | Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predict and estimate local feature influence by replacing one input at a time."""
    results = predict(artifact_dir, records)
    model, preprocessor, metadata = load_artifacts(artifact_dir)
    canonical, _ = canonicalize_dataframe(pd.DataFrame(records))
    baseline = model.predict(preprocessor.transform(canonical), verbose=0).reshape(-1)
    labels = {
        "gender": "Gender", "age": "Age", "attendance": "Attendance",
        "study_time": "Study time", "previous_grade": "Previous grade",
        "internet_access": "Internet access", "family_support": "Family support",
        "absences": "Absences", "participation": "Participation",
        "homework_completion": "Homework completion",
    }
    per_record: list[list[dict[str, Any]]] = [[] for _ in records]
    for feature in preprocessor.features:
        altered = canonical.copy()
        if feature in preprocessor.numeric:
            altered[feature] = preprocessor.numeric[feature].median
        else:
            known = [item for item in preprocessor.categories[feature] if item != "__unknown__"]
            altered[feature] = known[0] if known else "unknown"
        comparison = model.predict(preprocessor.transform(altered), verbose=0).reshape(-1)
        for index, (actual, neutral) in enumerate(zip(baseline, comparison, strict=True)):
            difference = float(actual - neutral)
            if metadata["task"] == "classification":
                description = "raises pass likelihood" if difference >= 0 else "lowers pass likelihood"
            else:
                description = "raises predicted grade" if difference >= 0 else "lowers predicted grade"
            per_record[index].append({
                "feature": feature,
                "label": labels.get(feature, feature.replace("_", " ").title()),
                "impact": round(difference, 5),
                "direction": description,
            })
    for result, influences in zip(results, per_record, strict=True):
        result["explanation"] = sorted(influences, key=lambda item: abs(item["impact"]), reverse=True)[:4]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts/model")
    parser.add_argument("--input", required=True, help="JSON object or list of student records")
    args = parser.parse_args()
    print(json.dumps(predict(args.artifacts, _read_records(args.input)), indent=2))


if __name__ == "__main__":
    main()
