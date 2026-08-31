"""Password hashing and server-side sessions using Python's standard library."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import secrets
import sqlite3

from .database import utc_now

PBKDF2_ITERATIONS = 600_000
SESSION_HOURS = 8


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if len(password) < 12 or len(password) > 128:
        raise ValueError("Password must contain between 12 and 128 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return base64.urlsafe_b64encode(salt).decode(), base64.urlsafe_b64encode(digest).decode()


def verify_password(password: str, encoded_salt: str, expected_hash: str) -> bool:
    try:
        salt = base64.urlsafe_b64decode(encoded_salt.encode())
        _, actual_hash = hash_password(password, salt)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual_hash, expected_hash)


def create_session(connection: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    expires = datetime.now(UTC) + timedelta(hours=SESSION_HOURS)
    connection.execute("DELETE FROM sessions WHERE expires_at < ?", (utc_now(),))
    connection.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token_hash, user_id, expires.isoformat(), utc_now()),
    )
    return token


def delete_session(connection: sqlite3.Connection, token: str) -> None:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def session_user(connection: sqlite3.Connection, token: str | None) -> sqlite3.Row | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return connection.execute(
        """SELECT u.id, u.email, u.full_name, u.role
           FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token_hash = ? AND s.expires_at > ? AND u.is_active = 1""",
        (token_hash, utc_now()),
    ).fetchone()

