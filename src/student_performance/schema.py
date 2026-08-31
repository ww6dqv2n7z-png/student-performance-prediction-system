"""Canonical feature definitions and strict validation boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericFeature:
    minimum: float
    maximum: float
    default: float


NUMERIC_FEATURES: dict[str, NumericFeature] = {
    "age": NumericFeature(10, 100, 18),
    "attendance": NumericFeature(0, 100, 100),
    "study_time": NumericFeature(0, 168, 2),
    "previous_grade": NumericFeature(0, 100, 0),
    "absences": NumericFeature(0, 365, 0),
    "participation": NumericFeature(0, 100, 0),
    "homework_completion": NumericFeature(0, 100, 0),
}

CATEGORICAL_FEATURES = ("gender", "internet_access", "family_support")
FEATURES = tuple(NUMERIC_FEATURES) + CATEGORICAL_FEATURES

ALIASES: dict[str, tuple[str, ...]] = {
    "gender": ("gender", "sex"),
    "age": ("age",),
    "attendance": ("attendance", "attendance_percent", "attendance_percentage"),
    "study_time": ("study_time", "studytime", "weekly_study_hours", "study_hours"),
    "previous_grade": ("previous_grade", "previous_grades", "g2", "previous_score"),
    "internet_access": ("internet_access", "internet"),
    "family_support": ("family_support", "family_educational_support", "famsup"),
    "absences": ("absences", "number_of_absences"),
    "participation": ("participation", "class_participation", "activities"),
    "homework_completion": ("homework_completion", "homework", "assignments_completed"),
    "final_grade": ("final_grade", "finalgrade", "g3", "performance_index"),
}


def normalize_column_name(name: object) -> str:
    return str(name).strip().lower().replace("%", "percent").replace("-", "_").replace(" ", "_")

