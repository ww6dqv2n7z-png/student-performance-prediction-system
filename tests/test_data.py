import pandas as pd
import pytest

from student_performance.data import canonicalize_dataframe, validate_training_data


def test_uci_columns_are_mapped_and_attendance_is_derived():
    frame = pd.DataFrame({
        "sex": ["F", "M"],
        "age": [16, 17],
        "studytime": [2, 3],
        "G2": [12, 14],
        "internet": ["yes", "no"],
        "famsup": ["yes", "no"],
        "absences": [0, 20],
        "activities": ["yes", "no"],
        "G3": [13, 15],
    })
    canonical, missing = canonicalize_dataframe(frame, school_days=200)
    assert canonical["gender"].tolist() == ["female", "male"]
    assert canonical["attendance"].tolist() == [100.0, 90.0]
    assert canonical["participation"].tolist() == [100.0, 0.0]
    assert "homework_completion" in missing
    assert "attendance" not in missing


def test_too_small_dataset_is_rejected():
    frame = pd.DataFrame({"final_grade": range(10)})
    with pytest.raises(ValueError, match="At least 30"):
        validate_training_data(frame)

