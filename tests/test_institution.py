import pytest
from pydantic import ValidationError

from student_performance.api import StudentCreate
from student_performance.institution import academic_label, load_institution, valid_level


def _student(**overrides):
    values = {
        "student_code": "CEIT-TEST-01",
        "department": "CEIT",
        "academic_year": 5,
        "semester": 2,
        "gender": "female",
        "age": 22,
        "attendance": 90,
        "study_time": 8,
        "previous_grade": 70,
        "internet_access": True,
        "family_support": True,
        "absences": 3,
        "participation": 80,
        "homework_completion": 85,
    }
    values.update(overrides)
    return StudentCreate(**values)


def test_ceit_academic_structure_is_loaded():
    config = load_institution()
    assert config["department_code"] == "CEIT"
    assert len(config["academic_levels"]) == 7
    assert valid_level(5, 2)
    assert academic_label(6, 1) == "Final Year"


def test_student_input_is_locked_to_ceit_and_supported_levels():
    student = _student()
    assert student.department == "Computer Engineering and Information Technology"
    with pytest.raises(ValidationError):
        _student(department="Civil Engineering")
    with pytest.raises(ValidationError):
        _student(academic_year=3, semester=2)
