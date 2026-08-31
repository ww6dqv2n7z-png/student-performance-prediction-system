"""Add clearly labelled synthetic MTU demo records to an initialized local system."""

from __future__ import annotations

import argparse

from student_performance.api import StudentCreate, _insert_student, _store_prediction
from student_performance.database import connect
from student_performance.predict import predict_with_explanations
from generate_demo_data import generate


DEPARTMENT = "Computer Engineering and Information Technology"
LEVELS = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (6, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--artifacts", default="artifacts/classifier")
    args = parser.parse_args()
    if not 1 <= args.rows <= 100:
        raise SystemExit("rows must be between 1 and 100")
    demo = generate(args.rows, 2026).drop(columns=["final_grade"])
    created: list[tuple[int, dict[str, object]]] = []
    with connect() as connection:
        admin = connection.execute("SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()
        if admin is None:
            raise SystemExit("Create an administrator before seeding demo records")
        for index, record in enumerate(demo.to_dict(orient="records"), start=1):
            year, semester = LEVELS[(index - 1) % len(LEVELS)]
            payload = StudentCreate(
                student_code=f"DEMO-{index:03d}", department=DEPARTMENT,
                academic_year=year, semester=semester, **record,
            )
            existing = connection.execute("SELECT id FROM students WHERE student_code=?", (payload.student_code,)).fetchone()
            if existing is None:
                student_id = _insert_student(connection, payload, admin["id"])
                created.append((student_id, record))
    for student_id, record in created:
        result = predict_with_explanations(args.artifacts, [record])[0]
        with connect() as connection:
            admin_id = connection.execute("SELECT id FROM users WHERE role='admin' ORDER BY id LIMIT 1").fetchone()[0]
            _store_prediction(connection, student_id, result, admin_id)
    print(f"Created {len(created)} synthetic demo student records with predictions")


if __name__ == "__main__":
    main()
