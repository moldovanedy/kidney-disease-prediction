from pathlib import Path

from kidney_predictor.schema import MODEL_FEATURE_COLUMNS, read_training_data


def test_bundled_csv_matches_training_schema() -> None:
    csv_path = Path(__file__).parents[1] / "data" / "sample_kidney_data.csv"

    features, labels = read_training_data(csv_path)

    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert len(features) == len(labels) > 0
    assert set(labels) == {0.0, 1.0}
    assert "ckd" not in features.columns
    assert "id" not in features.columns
