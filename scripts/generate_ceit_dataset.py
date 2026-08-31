"""Generate a reproducible, non-identifiable synthetic MTU CEIT project dataset.

The data is suitable for software demonstration and controlled experiments only. It is
not a substitute for approved institutional data and must never be described as real.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = (
    (1, 1, "First Year", 1.5),
    (2, 1, "Second Year", 1.0),
    (3, 1, "Third Year", 0.0),
    (4, 1, "Fourth Year", -1.0),
    (5, 1, "Fifth Year - First Semester", -2.0),
    (5, 2, "Fifth Year - Second Semester", -2.5),
    (6, 1, "Final Year", -3.0),
)
DEPARTMENT = "Computer Engineering and Information Technology"


def generate(records_per_level: int = 200, seed: int = 20260831) -> pd.DataFrame:
    """Return a balanced synthetic CEIT cohort with deterministic pseudo-random values."""
    if not 30 <= records_per_level <= 100_000:
        raise ValueError("records_per_level must be between 30 and 100,000")
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    sequence = 1

    for year, semester, label, difficulty in LEVELS:
        rows = records_per_level
        lower_performing = rng.random(rows) < 0.18
        ability = np.where(lower_performing, rng.normal(-1.4, 0.65, rows), rng.normal(0.25, 0.85, rows))
        family = rng.random(rows) < 0.68
        internet = rng.random(rows) < 0.88
        engagement = np.clip(rng.normal(0, 1, rows) + 0.25 * family + 0.15 * internet, -2.5, 2.5)

        age = np.clip(np.rint(rng.normal(17.5 + year, 0.75, rows)), 17, 28).astype(int)
        previous = np.clip(63 + 13 * ability + rng.normal(0, 5, rows), 20, 98)
        study = np.clip(7 + 1.1 * family + 0.7 * internet + 1.2 * engagement + rng.normal(0, 2.1, rows), 0.5, 24)
        attendance = np.clip(84 + 3 * family + 1.5 * internet + 4.5 * engagement + rng.normal(0, 4.5, rows), 50, 100)
        absences = np.clip(np.rint((100 - attendance) * 1.5 + rng.normal(0, 2.2, rows)), 0, 75).astype(int)
        participation = np.clip(62 + 13 * engagement + rng.normal(0, 8, rows), 10, 100)
        homework = np.clip(68 + 11 * engagement + 5 * family + 2 * internet + rng.normal(0, 8, rows), 10, 100)

        # Gender is intentionally excluded from the outcome equation. Family support and
        # internet access affect generated learning behaviours, not the grade directly.
        final_grade = np.clip(
            3
            + 0.42 * previous
            + 0.16 * attendance
            + 0.35 * study
            + 0.08 * participation
            + 0.10 * homework
            - 0.18 * absences
            + difficulty
            + rng.normal(0, 7, rows),
            0,
            100,
        )
        codes = [f"SYN-CEIT-{index:04d}" for index in range(sequence, sequence + rows)]
        sequence += rows
        frames.append(pd.DataFrame({
            "student_code": codes,
            "department": DEPARTMENT,
            "academic_year": year,
            "semester": semester,
            "academic_level": label,
            "gender": rng.choice(["female", "male"], rows, p=[0.5, 0.5]),
            "age": age,
            "attendance": attendance.round(1),
            "study_time": study.round(1),
            "previous_grade": previous.round(1),
            "internet_access": np.where(internet, "yes", "no"),
            "family_support": np.where(family, "yes", "no"),
            "absences": absences,
            "participation": participation.round(1),
            "homework_completion": homework.round(1),
            "final_grade": final_grade.round(1),
            "pass_fail": np.where(final_grade >= 50, "pass", "fail"),
            "dataset_type": "synthetic",
            "generation_seed": seed,
        }))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-per-level", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", default="data/ceit_synthetic_students.csv")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(args.records_per_level, args.seed)
    frame.to_csv(output, index=False)
    print(f"Created {output} with {len(frame):,} clearly labelled synthetic records")


if __name__ == "__main__":
    main()
