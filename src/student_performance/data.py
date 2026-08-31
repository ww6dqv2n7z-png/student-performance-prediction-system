"""Dataset loading, column normalization, and value validation."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .schema import ALIASES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, normalize_column_name

MAX_DATASET_BYTES = 100 * 1024 * 1024


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file() or csv_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be an existing .csv file")
    if csv_path.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("Dataset exceeds the 100 MB safety limit")
    # UCI files use semicolons; sep=None safely detects comma/semicolon delimiters.
    return pd.read_csv(csv_path, sep=None, engine="python")


def _find_source(columns: dict[str, str], canonical: str) -> str | None:
    for alias in ALIASES[canonical]:
        if alias in columns:
            return columns[alias]
    return None


def _normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
        "no": "no", "n": "no", "false": "no", "0": "no",
    }
    values = series.astype("string").str.strip().str.lower()
    return values.map(mapping).fillna(values).fillna("unknown")


def canonicalize_dataframe(frame: pd.DataFrame, school_days: int = 200) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty:
        raise ValueError("Dataset is empty")
    if len(frame) > 1_000_000:
        raise ValueError("Dataset exceeds the one-million-row safety limit")

    normalized = {normalize_column_name(column): str(column) for column in frame.columns}
    output = pd.DataFrame(index=frame.index)
    missing: list[str] = []

    for canonical in (*FEATURES, "final_grade"):
        source = _find_source(normalized, canonical)
        if source is None:
            output[canonical] = np.nan
            missing.append(canonical)
        else:
            output[canonical] = frame[source]

    # UCI has absences but no attendance percentage; derive a transparent estimate.
    if output["attendance"].isna().all() and not output["absences"].isna().all():
        absences = pd.to_numeric(output["absences"], errors="coerce")
        output["attendance"] = (100 - (absences / school_days * 100)).clip(0, 100)
        missing.remove("attendance")

    if "activities" in normalized and output["participation"].notna().any():
        activity = _normalize_binary(output["participation"])
        output["participation"] = activity.map({"yes": 100.0, "no": 0.0})

    for name in NUMERIC_FEATURES:
        output[name] = pd.to_numeric(output[name], errors="coerce")
    output["final_grade"] = pd.to_numeric(output["final_grade"], errors="coerce")

    output["gender"] = output["gender"].astype("string").str.strip().str.lower().replace({"m": "male", "f": "female"}).fillna("unknown")
    output["internet_access"] = _normalize_binary(output["internet_access"])
    output["family_support"] = _normalize_binary(output["family_support"])
    missing_features = [name for name in FEATURES if output[name].isna().all()]
    if missing_features:
        warnings.warn(
            "Dropping unavailable features: " + ", ".join(missing_features),
            stacklevel=2,
        )
    return output, missing_features


def validate_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["final_grade"]).copy()
    if len(clean) < 30:
        raise ValueError("At least 30 rows with a valid final grade are required")
    if clean["final_grade"].nunique() < 2:
        raise ValueError("Final grade must contain at least two distinct values")
    return clean
