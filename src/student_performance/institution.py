"""Validated institutional configuration for the CEIT departmental pilot."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def load_institution() -> dict[str, Any]:
    path = Path("config/institution.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"institution", "department", "department_code", "academic_levels", "mtu_validation_status"}
    if not required <= payload.keys() or not isinstance(payload["academic_levels"], list):
        raise ValueError("Institution configuration is invalid")
    return payload


def valid_level(year: int, semester: int) -> bool:
    return any(level["year"] == year and level["semester"] == semester for level in load_institution()["academic_levels"])


def academic_label(year: int, semester: int) -> str:
    for level in load_institution()["academic_levels"]:
        if level["year"] == year and level["semester"] == semester:
            return str(level["label"])
    return f"Year {year} · Semester {semester}"

