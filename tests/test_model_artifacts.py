from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from kidney_predictor.schema import read_training_data
from kidney_predictor.train import train


def test_training_exports_model_and_learned_factor_importance(tmp_path: Path) -> None:
    csv_path = Path(__file__).parents[1] / "data" / "sample_kidney_data.csv"
    model_path = tmp_path / "saved_model"
    importance_path = tmp_path / "feature_importance.csv"

    train(csv_path, model_path, epochs=1)

    importance = pd.read_csv(importance_path)
    assert len(importance) == 24
    assert importance["factor"].is_unique
    assert (importance["mean_absolute_probability_change"] >= 0).all()
    assert importance["mean_absolute_probability_change"].nunique() > 1

    features, _ = read_training_data(csv_path)
    checkpoint = tf.train.list_variables(str(model_path / "variables" / "variables"))
    checkpoint_shapes = {tuple(shape) for _, shape in checkpoint}
    assert (features.shape[1], 1) in checkpoint_shapes
    assert (features.shape[1], 64) not in checkpoint_shapes

    predictor = tf.saved_model.load(str(model_path)).signatures["serving_default"]
    predictions = predictor(
        features=tf.convert_to_tensor(features.to_numpy(dtype=np.float32))
    )["probability"].numpy()
    assert predictions.shape == (len(features),)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()
