import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_ceit_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_ceit_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ceit_synthetic_dataset_is_balanced_valid_and_reproducible():
    first = MODULE.generate(records_per_level=30, seed=7)
    second = MODULE.generate(records_per_level=30, seed=7)
    assert first.equals(second)
    assert len(first) == 210
    assert first["student_code"].is_unique
    assert set(first.groupby(["academic_year", "semester"]).size()) == {30}
    assert set(first["dataset_type"]) == {"synthetic"}
    assert first["final_grade"].between(0, 100).all()
    assert first["attendance"].between(0, 100).all()
    assert first["pass_fail"].isin(["pass", "fail"]).all()
