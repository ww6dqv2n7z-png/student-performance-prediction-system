import json

import numpy as np
import pandas as pd

from student_performance.preprocessing import SafePreprocessor


def test_preprocessor_round_trip_uses_json(tmp_path):
    frame = pd.DataFrame({
        "age": [16, 17, 18, None],
        "attendance": [80, 90, 100, 95],
        "gender": ["female", "male", "female", "male"],
    })
    processor = SafePreprocessor(["age", "attendance", "gender"])
    expected = processor.fit_transform(frame)
    path = tmp_path / "preprocessor.json"
    processor.save(path)
    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    actual = SafePreprocessor.load(path).transform(frame)
    np.testing.assert_allclose(actual, expected)


def test_unknown_category_has_a_stable_column():
    train = pd.DataFrame({"gender": ["female", "male"]})
    processor = SafePreprocessor(["gender"]).fit(train)
    transformed = processor.transform(pd.DataFrame({"gender": ["nonbinary"]}))
    assert transformed.shape == (1, 3)
    assert transformed.sum() == 1

