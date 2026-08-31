"""Administrative commands for the local MTU application."""

from __future__ import annotations

import argparse
from getpass import getpass

from .auth import hash_password
from .database import audit, connect, initialize_database, utc_now


def create_user(email: str, full_name: str, role: str, password: str) -> None:
    initialize_database()
    salt, digest = hash_password(password)
    with connect() as connection:
        cursor = connection.execute(
            """INSERT INTO users (email, full_name, role, password_salt, password_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (email.strip().lower(), full_name.strip(), role, salt, digest, utc_now()),
        )
        audit(connection, cursor.lastrowid, "create_user", "user", cursor.lastrowid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-user")
    create.add_argument("--email", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--role", choices=("admin", "teacher"), default="admin")
    args = parser.parse_args()
    if args.command == "create-user":
        password = getpass("Password (12+ characters): ")
        confirmation = getpass("Confirm password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match")
        create_user(args.email, args.name, args.role, password)
        print(f"Created {args.role}: {args.email.lower()}")


if __name__ == "__main__":
    main()

