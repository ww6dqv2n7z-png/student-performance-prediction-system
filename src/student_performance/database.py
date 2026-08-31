"""SQLite persistence for the local MTU academic support application."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
import sqlite3
from typing import Iterator


def database_path() -> Path:
    configured = os.environ.get("STUDENT_DATABASE", "data/student_support.db")
    path = Path(configured).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'teacher')),
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    student_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
    department TEXT NOT NULL,
    academic_year INTEGER NOT NULL CHECK (academic_year BETWEEN 1 AND 6),
    semester INTEGER NOT NULL CHECK (semester BETWEEN 1 AND 2),
    gender TEXT NOT NULL CHECK (gender IN ('female', 'male', 'other')),
    age INTEGER NOT NULL CHECK (age BETWEEN 10 AND 100),
    attendance REAL NOT NULL CHECK (attendance BETWEEN 0 AND 100),
    study_time REAL NOT NULL CHECK (study_time BETWEEN 0 AND 168),
    previous_grade REAL NOT NULL CHECK (previous_grade BETWEEN 0 AND 100),
    internet_access INTEGER NOT NULL CHECK (internet_access IN (0, 1)),
    family_support INTEGER NOT NULL CHECK (family_support IN (0, 1)),
    absences INTEGER NOT NULL CHECK (absences BETWEEN 0 AND 365),
    participation REAL NOT NULL CHECK (participation BETWEEN 0 AND 100),
    homework_completion REAL NOT NULL CHECK (homework_completion BETWEEN 0 AND 100),
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    student_id INTEGER REFERENCES students(id) ON DELETE SET NULL,
    result_label TEXT,
    pass_probability REAL,
    predicted_final_grade REAL,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    explanation_json TEXT NOT NULL DEFAULT '[]',
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed')),
    assigned_to TEXT NOT NULL,
    due_date TEXT,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_students_department ON students(department);
CREATE INDEX IF NOT EXISTS idx_predictions_student_created ON predictions(student_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interventions_status ON interventions(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);
"""


def initialize_database() -> None:
    with connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        connection.execute("PRAGMA optimize")
    try:
        database_path().chmod(0o600)
    except OSError:
        # The database may live on a filesystem that does not support POSIX modes.
        pass


def audit(connection: sqlite3.Connection, user_id: int | None, action: str, entity_type: str, entity_id: int | None = None) -> None:
    connection.execute(
        "INSERT INTO audit_logs (user_id, action, entity_type, entity_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, action[:80], entity_type[:40], entity_id, utc_now()),
    )
