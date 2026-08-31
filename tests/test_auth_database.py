from student_performance.auth import hash_password, verify_password
from student_performance.database import connect, initialize_database


def test_password_hashing_is_salted_and_verifiable():
    salt_one, digest_one = hash_password("a-secure-test-password")
    salt_two, digest_two = hash_password("a-secure-test-password")
    assert salt_one != salt_two
    assert digest_one != digest_two
    assert verify_password("a-secure-test-password", salt_one, digest_one)
    assert not verify_password("incorrect-password", salt_one, digest_one)


def test_database_schema_and_constraints(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDENT_DATABASE", str(tmp_path / "test.db"))
    initialize_database()
    with connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
        indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='index'")}
    assert {"users", "students", "predictions", "interventions", "audit_logs"} <= tables
    assert "idx_predictions_student_created" in indexes

