"""A small JSON-serializable preprocessor; no pickle deserialization required."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


@dataclass
class NumericState:
    median: float
    mean: float
    scale: float
    minimum: float
    maximum: float


class SafePreprocessor:
    """Median-impute/standardize numbers and one-hot encode categories."""

    VERSION = 1

    def __init__(self, features: list[str]):
        unknown = set(features) - (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES))
        if unknown or not features:
            raise ValueError(f"Invalid feature selection: {sorted(unknown)}")
        self.features = list(features)
        self.numeric: dict[str, NumericState] = {}
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "SafePreprocessor":
        for name in self.features:
            if name in NUMERIC_FEATURES:
                spec = NUMERIC_FEATURES[name]
                values = pd.to_numeric(frame[name], errors="coerce")
                valid = values[(values >= spec.minimum) & (values <= spec.maximum)]
                median = float(valid.median()) if not valid.empty else spec.default
                filled = values.where(values.between(spec.minimum, spec.maximum), np.nan).fillna(median)
                mean = float(filled.mean())
                scale = float(filled.std(ddof=0)) or 1.0
                self.numeric[name] = NumericState(median, mean, scale, spec.minimum, spec.maximum)
            else:
                values = frame[name].astype("string").str.strip().str.lower().fillna("unknown")
                categories = sorted(set(values.tolist()) | {"__unknown__"})
                self.categories[name] = categories
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.numeric and not self.categories:
            raise RuntimeError("Preprocessor is not fitted")
        columns: list[np.ndarray] = []
        for name in self.features:
            if name in self.numeric:
                state = self.numeric[name]
                values = pd.to_numeric(frame.get(name), errors="coerce")
                invalid = ~values.between(state.minimum, state.maximum)
                values = values.mask(invalid).fillna(state.median)
                columns.append(((values.to_numpy(dtype=np.float32) - state.mean) / state.scale)[:, None])
            else:
                values = frame.get(name, pd.Series("unknown", index=frame.index)).astype("string").str.strip().str.lower().fillna("unknown")
                categories = self.categories[name]
                lookup = {category: index for index, category in enumerate(categories)}
                unknown_index = lookup["__unknown__"]
                indices = np.array([lookup.get(value, unknown_index) for value in values], dtype=int)
                encoded = np.zeros((len(frame), len(categories)), dtype=np.float32)
                encoded[np.arange(len(frame)), indices] = 1.0
                columns.append(encoded)
        return np.concatenate(columns, axis=1).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    @property
    def output_size(self) -> int:
        return len(self.numeric) + sum(len(items) for items in self.categories.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "features": self.features,
            "numeric": {name: asdict(state) for name, state in self.numeric.items()},
            "categories": self.categories,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SafePreprocessor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != cls.VERSION:
            raise ValueError("Unsupported preprocessor version")
        instance = cls(payload["features"])
        instance.numeric = {name: NumericState(**state) for name, state in payload["numeric"].items()}
        instance.categories = {name: list(values) for name, values in payload["categories"].items()}
        return instance

