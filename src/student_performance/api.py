"""Secure local API for the MTU Student Academic Support System."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
import csv
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Annotated, Literal

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

from .auth import create_session, delete_session, hash_password, session_user, verify_password
from .database import audit, connect, initialize_database, utc_now
from .institution import academic_label, load_institution, valid_level
from .predict import predict_with_explanations

COOKIE_NAME = "mtu_student_session"
ALLOWED_ORIGINS = {"http://localhost:3000", "http://127.0.0.1:3000"}
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    gender: Literal["female", "male", "other"]
    age: int = Field(ge=10, le=100)
    attendance: float = Field(ge=0, le=100)
    study_time: float = Field(ge=0, le=168)
    previous_grade: float = Field(ge=0, le=100)
    internet_access: bool
    family_support: bool
    absences: int = Field(ge=0, le=365)
    participation: float = Field(ge=0, le=100)
    homework_completion: float = Field(ge=0, le=100)


class StudentCreate(StudentFeatures):
    student_code: str = Field(min_length=2, max_length=30, pattern=r"^[A-Za-z0-9_-]+$")
    department: str = Field(min_length=2, max_length=80)
    academic_year: int = Field(ge=1, le=6)
    semester: int = Field(ge=1, le=2)

    @field_validator("department")
    @classmethod
    def department_is_ceit(cls, value: str) -> str:
        expected = load_institution()["department"]
        if value.casefold() not in {str(expected).casefold(), str(load_institution()["department_code"]).casefold()}:
            raise ValueError("This pilot accepts CEIT department records only")
        return str(expected)

    @field_validator("semester")
    @classmethod
    def academic_level_exists(cls, semester: int, info):
        year = info.data.get("academic_year")
        if year is not None and not valid_level(year, semester):
            raise ValueError("Academic year and semester are not valid for CEIT")
        return semester


class BatchStudents(BaseModel):
    model_config = ConfigDict(extra="forbid")
    students: list[StudentCreate] = Field(min_length=1, max_length=500)


class LoginInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    email: str = Field(min_length=5, max_length=254)
    full_name: str = Field(min_length=2, max_length=100)
    role: Literal["admin", "teacher"]
    password: str = Field(min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value: str) -> str:
        value = value.lower()
        if not EMAIL_PATTERN.fullmatch(value):
            raise ValueError("Enter a valid email address")
        return value


class InterventionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    student_id: int = Field(gt=0)
    prediction_id: int | None = Field(default=None, gt=0)
    action: str = Field(min_length=3, max_length=200)
    notes: str = Field(default="", max_length=2000)
    assigned_to: str = Field(min_length=2, max_length=100)
    due_date: date | None = None


class InterventionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    status: Literal["pending", "in_progress", "completed"]
    notes: str = Field(default="", max_length=2000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="MTU Student Academic Support API",
    description="Local decision-support API for Mandalay Technological University.",
    version="2.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)

_requests: dict[str, deque[float]] = defaultdict(deque)
_login_attempts: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT = 90
WINDOW_SECONDS = 60


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    content_length = request.headers.get("content-length", "0")
    if not content_length.isdigit() or int(content_length) > 2_000_000:
        return JSONResponse({"detail": "Request is too large"}, status_code=413)
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"detail": "Origin is not allowed"}, status_code=403)
    client = request.client.host if request.client else "local"
    now = time.monotonic()
    history = _requests[client]
    while history and history[0] < now - WINDOW_SECONDS:
        history.popleft()
    if len(history) >= RATE_LIMIT:
        return JSONResponse({"detail": "Too many requests; try again shortly"}, status_code=429)
    history.append(now)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


def _model_directory() -> Path:
    return Path(os.environ.get("STUDENT_MODEL_DIR", "artifacts/classifier")).expanduser().resolve()


def _research_directory() -> Path:
    return Path(os.environ.get("STUDENT_RESEARCH_DIR", "artifacts/experiments/uci")).expanduser().resolve()


def current_user(session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None) -> dict[str, object]:
    with connect() as connection:
        row = session_user(connection, session)
    if row is None:
        raise HTTPException(status_code=401, detail="Sign in required")
    return dict(row)


def admin_user(user: Annotated[dict[str, object], Depends(current_user)]) -> dict[str, object]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return user


User = Annotated[dict[str, object], Depends(current_user)]
Admin = Annotated[dict[str, object], Depends(admin_user)]


def _student_dict(row: sqlite3.Row) -> dict[str, object]:
    result = dict(row)
    result["internet_access"] = bool(result["internet_access"])
    result["family_support"] = bool(result["family_support"])
    result["academic_label"] = academic_label(result["academic_year"], result["semester"])
    return result


def _risk(result: dict[str, object]) -> str:
    if "pass_probability" in result:
        probability = float(result["pass_probability"])
        return "high" if probability < 0.4 else "medium" if probability < 0.7 else "low"
    grade = float(result.get("predicted_final_grade", 0))
    return "high" if grade < 50 else "medium" if grade < 65 else "low"


def _store_prediction(connection: sqlite3.Connection, student_id: int | None, result: dict[str, object], user_id: int) -> int:
    cursor = connection.execute(
        """INSERT INTO predictions
           (student_id, result_label, pass_probability, predicted_final_grade, risk_level, explanation_json, created_by, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            student_id, result.get("prediction"), result.get("pass_probability"),
            result.get("predicted_final_grade"), _risk(result),
            json.dumps(result.get("explanation", [])), user_id, utc_now(),
        ),
    )
    audit(connection, user_id, "create_prediction", "prediction", cursor.lastrowid)
    return int(cursor.lastrowid)


@app.get("/api/health")
def health() -> dict[str, object]:
    root = _model_directory()
    required = ("model.keras", "preprocessor.json", "metadata.json", "manifest.json")
    with connect() as connection:
        users = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    ready = root.is_dir() and all((root / name).is_file() for name in required)
    return {"status": "ready" if ready and users else "setup_required", "model_ready": ready, "admin_ready": users > 0}


@app.get("/api/config")
def institution_config() -> dict[str, object]:
    return load_institution()


@app.post("/api/auth/login")
def login(credentials: LoginInput, response: Response) -> dict[str, object]:
    attempt_key = credentials.email.lower()
    now = time.monotonic()
    attempts = _login_attempts[attempt_key]
    while attempts and attempts[0] < now - 300:
        attempts.popleft()
    if len(attempts) >= 5:
        raise HTTPException(status_code=429, detail="Too many sign-in attempts; wait five minutes")
    with connect() as connection:
        row = connection.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (credentials.email.lower(),)).fetchone()
        if row is None or not verify_password(credentials.password, row["password_salt"], row["password_hash"]):
            attempts.append(now)
            raise HTTPException(status_code=401, detail="Invalid email or password")
        token = create_session(connection, row["id"])
        audit(connection, row["id"], "login", "session")
        attempts.clear()
    response.set_cookie(COOKIE_NAME, token, max_age=8 * 3600, httponly=True, samesite="strict", secure=False, path="/")
    return {"user": {"id": row["id"], "email": row["email"], "full_name": row["full_name"], "role": row["role"]}}


@app.post("/api/auth/logout")
def logout(response: Response, user: User, session: Annotated[str | None, Cookie(alias=COOKIE_NAME)] = None) -> dict[str, bool]:
    if session:
        with connect() as connection:
            delete_session(connection, session)
            audit(connection, int(user["id"]), "logout", "session")
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: User) -> dict[str, object]:
    return {"user": user}


@app.post("/api/auth/change-password")
def change_password(payload: PasswordChange, user: User, response: Response) -> dict[str, bool]:
    with connect() as connection:
        row = connection.execute("SELECT password_salt, password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()
        if not verify_password(payload.current_password, row["password_salt"], row["password_hash"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        salt, digest = hash_password(payload.new_password)
        connection.execute("UPDATE users SET password_salt = ?, password_hash = ? WHERE id = ?", (salt, digest, user["id"]))
        connection.execute("DELETE FROM sessions WHERE user_id = ?", (user["id"],))
        audit(connection, int(user["id"]), "change_password", "user", int(user["id"]))
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/users")
def list_users(_: Admin) -> dict[str, object]:
    with connect() as connection:
        rows = connection.execute("SELECT id, email, full_name, role, is_active, created_at FROM users ORDER BY full_name").fetchall()
    return {"users": [dict(row) for row in rows]}


@app.post("/api/users", status_code=201)
def add_user(payload: UserCreate, admin: Admin) -> dict[str, object]:
    salt, digest = hash_password(payload.password)
    try:
        with connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email, full_name, role, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (payload.email, payload.full_name, payload.role, salt, digest, utc_now()),
            )
            audit(connection, int(admin["id"]), "create_user", "user", cursor.lastrowid)
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Email already exists") from exc
    return {"id": user_id}


@app.get("/api/dashboard")
def dashboard(user: User) -> dict[str, object]:
    with connect() as connection:
        total_students = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        total_predictions = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        active_interventions = connection.execute("SELECT COUNT(*) FROM interventions WHERE status != 'completed'").fetchone()[0]
        risk_rows = connection.execute(
            """WITH latest AS (
                 SELECT risk_level, ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY id DESC) rank
                 FROM predictions WHERE student_id IS NOT NULL
               )
               SELECT risk_level, COUNT(*) count FROM latest WHERE rank=1 GROUP BY risk_level"""
        ).fetchall()
        departments = connection.execute(
            """WITH latest AS (
                 SELECT p.*, ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY id DESC) rank
                 FROM predictions p WHERE student_id IS NOT NULL
               )
               SELECT s.department, COUNT(DISTINCT s.id) students,
                      SUM(CASE WHEN l.risk_level = 'high' AND l.rank = 1 THEN 1 ELSE 0 END) high_risk,
                      ROUND(AVG(CASE WHEN l.rank = 1 THEN l.pass_probability END) * 100, 1) avg_pass_probability
               FROM students s LEFT JOIN latest l ON l.student_id = s.id
               GROUP BY s.department ORDER BY students DESC"""
        ).fetchall()
        cohorts = connection.execute(
            """WITH latest AS (
                 SELECT p.*, ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY id DESC) rank
                 FROM predictions p WHERE student_id IS NOT NULL
               )
               SELECT s.academic_year, s.semester, COUNT(DISTINCT s.id) students,
                      SUM(CASE WHEN l.risk_level = 'high' AND l.rank = 1 THEN 1 ELSE 0 END) high_risk,
                      ROUND(AVG(CASE WHEN l.rank = 1 THEN l.pass_probability END) * 100, 1) avg_pass_probability
               FROM students s LEFT JOIN latest l ON l.student_id = s.id
               GROUP BY s.academic_year, s.semester ORDER BY s.academic_year, s.semester"""
        ).fetchall()
        recent = connection.execute(
            """SELECT p.id, p.result_label, p.pass_probability, p.predicted_final_grade, p.risk_level, p.created_at,
                      s.student_code, s.department
               FROM predictions p LEFT JOIN students s ON s.id = p.student_id
               ORDER BY p.id DESC LIMIT 8"""
        ).fetchall()
    root = _model_directory()
    metrics = {}
    model_metadata = {}
    try:
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        model_metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    research = {}
    try:
        research = json.loads((_research_directory() / "benchmark_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "summary": {"students": total_students, "predictions": total_predictions, "active_interventions": active_interventions,
                    "high_risk": next((row["count"] for row in risk_rows if row["risk_level"] == "high"), 0)},
        "risk_distribution": {row["risk_level"]: row["count"] for row in risk_rows},
        "departments": [dict(row) for row in departments],
        "cohorts": [{**dict(row), "label": academic_label(row["academic_year"], row["semester"])} for row in cohorts],
        "recent_predictions": [dict(row) for row in recent],
        "model_metrics": metrics, "model_metadata": model_metadata, "research": research,
        "institution": load_institution(), "viewer": user,
    }


@app.get("/api/students")
def list_students(user: User, search: str = Query(default="", max_length=80), limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
    pattern = f"%{search.strip()}%"
    with connect() as connection:
        rows = connection.execute(
            """SELECT s.*, p.risk_level, p.pass_probability, p.result_label, p.created_at prediction_date
               FROM students s LEFT JOIN predictions p ON p.id = (
                 SELECT id FROM predictions WHERE student_id = s.id ORDER BY id DESC LIMIT 1
               )
               WHERE s.student_code LIKE ? OR s.department LIKE ?
               ORDER BY s.updated_at DESC LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
    return {"students": [_student_dict(row) for row in rows], "viewer": user}


def _insert_student(connection: sqlite3.Connection, student: StudentCreate, user_id: int) -> int:
    values = student.model_dump()
    now = utc_now()
    cursor = connection.execute(
        """INSERT INTO students
           (student_code, department, academic_year, semester, gender, age, attendance, study_time,
            previous_grade, internet_access, family_support, absences, participation, homework_completion,
            created_by, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (values["student_code"].upper(), values["department"], values["academic_year"], values["semester"],
         values["gender"], values["age"], values["attendance"], values["study_time"], values["previous_grade"],
         int(values["internet_access"]), int(values["family_support"]), values["absences"], values["participation"],
         values["homework_completion"], user_id, now, now),
    )
    audit(connection, user_id, "create_student", "student", cursor.lastrowid)
    return int(cursor.lastrowid)


@app.post("/api/students", status_code=201)
def add_student(student: StudentCreate, user: User) -> dict[str, int]:
    try:
        with connect() as connection:
            student_id = _insert_student(connection, student, int(user["id"]))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Student code already exists") from exc
    return {"id": student_id}


@app.post("/api/students/batch")
def add_students_batch(payload: BatchStudents, user: User) -> dict[str, object]:
    created = 0
    duplicates: list[str] = []
    with connect() as connection:
        for student in payload.students:
            try:
                _insert_student(connection, student, int(user["id"]))
                created += 1
            except sqlite3.IntegrityError:
                duplicates.append(student.student_code)
        audit(connection, int(user["id"]), "batch_import", "student", created)
    return {"created": created, "duplicates": duplicates}


@app.put("/api/students/{student_id}")
def update_student(student_id: int, student: StudentCreate, user: User) -> dict[str, bool]:
    values = student.model_dump()
    try:
        with connect() as connection:
            cursor = connection.execute(
                """UPDATE students SET student_code=?, department=?, academic_year=?, semester=?, gender=?, age=?,
                   attendance=?, study_time=?, previous_grade=?, internet_access=?, family_support=?, absences=?,
                   participation=?, homework_completion=?, updated_at=? WHERE id=?""",
                (values["student_code"].upper(), values["department"], values["academic_year"], values["semester"],
                 values["gender"], values["age"], values["attendance"], values["study_time"], values["previous_grade"],
                 int(values["internet_access"]), int(values["family_support"]), values["absences"], values["participation"],
                 values["homework_completion"], utc_now(), student_id),
            )
            if not cursor.rowcount:
                raise HTTPException(status_code=404, detail="Student not found")
            audit(connection, int(user["id"]), "update_student", "student", student_id)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Student code already exists") from exc
    return {"ok": True}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, admin: Admin) -> dict[str, bool]:
    with connect() as connection:
        cursor = connection.execute("DELETE FROM students WHERE id = ?", (student_id,))
        if not cursor.rowcount:
            raise HTTPException(status_code=404, detail="Student not found")
        audit(connection, int(admin["id"]), "delete_student", "student", student_id)
    return {"ok": True}


def _student_features(row: sqlite3.Row) -> dict[str, object]:
    fields = StudentFeatures.model_fields
    return {name: bool(row[name]) if name in {"internet_access", "family_support"} else row[name] for name in fields}


@app.post("/api/students/{student_id}/predict")
def predict_student(student_id: int, user: User) -> dict[str, object]:
    with connect() as connection:
        student = connection.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    try:
        result = predict_with_explanations(_model_directory(), [_student_features(student)])[0]
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="The trained model is unavailable or failed its integrity check") from exc
    with connect() as connection:
        prediction_id = _store_prediction(connection, student_id, result, int(user["id"]))
    return {"id": prediction_id, "risk_level": _risk(result), "result": result,
            "notice": "Use this signal to offer support, never as the sole academic decision."}


@app.post("/api/predict")
def make_prediction(student: StudentFeatures, user: User) -> dict[str, object]:
    try:
        result = predict_with_explanations(_model_directory(), [student.model_dump()])[0]
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="The trained model is unavailable or failed its integrity check") from exc
    with connect() as connection:
        prediction_id = _store_prediction(connection, None, result, int(user["id"]))
    return {"id": prediction_id, "risk_level": _risk(result), "result": result,
            "notice": "Use this signal to offer support, never as the sole academic decision."}


@app.get("/api/predictions")
def list_predictions(_: User, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, object]:
    with connect() as connection:
        rows = connection.execute(
            """SELECT p.*, s.student_code, s.department, u.full_name created_by_name
               FROM predictions p LEFT JOIN students s ON s.id=p.student_id JOIN users u ON u.id=p.created_by
               ORDER BY p.id DESC LIMIT ?""", (limit,),
        ).fetchall()
    predictions = []
    for row in rows:
        item = dict(row)
        item["explanation"] = json.loads(item.pop("explanation_json"))
        predictions.append(item)
    return {"predictions": predictions}


@app.get("/api/interventions")
def list_interventions(_: User, status: Literal["all", "pending", "in_progress", "completed"] = "all") -> dict[str, object]:
    clause, parameters = ("", ()) if status == "all" else ("WHERE i.status = ?", (status,))
    with connect() as connection:
        rows = connection.execute(
            f"""SELECT i.*, s.student_code, s.department FROM interventions i
                 JOIN students s ON s.id=i.student_id {clause} ORDER BY i.updated_at DESC LIMIT 500""", parameters,
        ).fetchall()
    return {"interventions": [dict(row) for row in rows]}


@app.post("/api/interventions", status_code=201)
def add_intervention(payload: InterventionCreate, user: User) -> dict[str, int]:
    now = utc_now()
    try:
        with connect() as connection:
            cursor = connection.execute(
                """INSERT INTO interventions
                   (student_id, prediction_id, action, notes, status, assigned_to, due_date, created_by, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)""",
                (payload.student_id, payload.prediction_id, payload.action, payload.notes, payload.assigned_to,
                 payload.due_date.isoformat() if payload.due_date else None, user["id"], now, now),
            )
            audit(connection, int(user["id"]), "create_intervention", "intervention", cursor.lastrowid)
            intervention_id = cursor.lastrowid
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=400, detail="Student or prediction does not exist") from exc
    return {"id": intervention_id}


@app.patch("/api/interventions/{intervention_id}")
def update_intervention(intervention_id: int, payload: InterventionUpdate, user: User) -> dict[str, bool]:
    with connect() as connection:
        cursor = connection.execute("UPDATE interventions SET status=?, notes=?, updated_at=? WHERE id=?",
                                    (payload.status, payload.notes, utc_now(), intervention_id))
        if not cursor.rowcount:
            raise HTTPException(status_code=404, detail="Intervention not found")
        audit(connection, int(user["id"]), "update_intervention", "intervention", intervention_id)
    return {"ok": True}


@app.get("/api/reports/at-risk.csv")
def at_risk_report(_: User):
    with connect() as connection:
        rows = connection.execute(
            """SELECT s.student_code, s.department, s.academic_year, s.semester, p.risk_level,
                      ROUND(p.pass_probability * 100, 1) pass_probability_percent, p.created_at
               FROM students s JOIN predictions p ON p.id=(SELECT id FROM predictions WHERE student_id=s.id ORDER BY id DESC LIMIT 1)
               WHERE p.risk_level IN ('high', 'medium') ORDER BY p.risk_level DESC, s.department, s.student_code"""
        ).fetchall()
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Student Code", "Department", "Academic Year", "Semester", "Risk Level", "Pass Probability (%)", "Prediction Date"])
    writer.writerows([tuple(row) for row in rows])
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=mtu_at_risk_students.csv"})


@app.get("/api/research/assets/{filename}")
def research_asset(filename: str, _: User):
    allowed = {"model_comparison.png", "confusion_matrices.png", "fairness_audit.png"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Research asset not found")
    path = _research_directory() / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Research asset not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/research/files/{filename}")
def research_file(filename: str, _: User):
    allowed = {
        "model_comparison.csv": "text/csv",
        "fairness_audit.csv": "text/csv",
        "discussion.md": "text/markdown",
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Research file not found")
    path = _research_directory() / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Research file not found")
    return FileResponse(path, media_type=allowed[filename], filename=filename)


def main() -> None:
    import uvicorn
    uvicorn.run("student_performance.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
