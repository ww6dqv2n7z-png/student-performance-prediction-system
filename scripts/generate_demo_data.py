"""Generate non-identifiable synthetic data for a quick end-to-end demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(55, 100, rows)
    study_time = rng.uniform(0.5, 18, rows)
    previous = np.clip(rng.normal(65, 15, rows), 0, 100)
    absences = np.clip(np.rint((100 - attendance) / 3 + rng.normal(1, 2, rows)), 0, 40)
    participation = rng.uniform(20, 100, rows)
    homework = rng.uniform(30, 100, rows)
    internet = rng.choice(["yes", "no"], rows, p=[0.85, 0.15])
    family = rng.choice(["yes", "no"], rows, p=[0.7, 0.3])
    grade = (
        0.37 * previous + 0.16 * attendance + 0.65 * study_time
        + 0.08 * participation + 0.10 * homework - 0.35 * absences
        + 2 * (internet == "yes") + 2 * (family == "yes") + rng.normal(0, 5, rows)
    )
    return pd.DataFrame({
        "gender": rng.choice(["female", "male"], rows),
        "age": rng.integers(15, 23, rows),
        "attendance": attendance.round(2),
        "study_time": study_time.round(2),
        "previous_grade": previous.round(2),
        "internet_access": internet,
        "family_support": family,
        "absences": absences.astype(int),
        "participation": participation.round(2),
        "homework_completion": homework.round(2),
        "final_grade": np.clip(grade, 0, 100).round(2),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/demo_students.csv")
    args = parser.parse_args()
    if not 30 <= args.rows <= 1_000_000:
        raise SystemExit("rows must be between 30 and 1,000,000")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows, args.seed).to_csv(output, index=False)
    print(f"Created {output} with {args.rows} synthetic rows")


if __name__ == "__main__":
    main()

