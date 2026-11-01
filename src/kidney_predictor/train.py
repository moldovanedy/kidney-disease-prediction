import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from kidney_predictor.schema import (
    CATEGORICAL_VALUES,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    read_training_data,
)


class SavedModelPredictor(tf.Module):
    def __init__(self, classifier: tf.keras.Model, medians: tf.Tensor) -> None:
        super().__init__()
        self.classifier = classifier
        self.medians = medians

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[None, len(MODEL_FEATURE_COLUMNS)],
                dtype=tf.float32,
                name="features",
            )
        ]
    )
    def serve(self, features: tf.Tensor) -> dict[str, tf.Tensor]:
        features = tf.where(tf.math.is_finite(features), features, self.medians)
        return {"probability": tf.squeeze(self.classifier(features), axis=-1)}


def write_feature_importance(
    classifier: tf.keras.Model,
    features: pd.DataFrame,
    values: np.ndarray,
    medians: np.ndarray,
    output_path: Path,
) -> None:
    baseline_probabilities = classifier(values, training=False).numpy().reshape(-1)
    feature_groups = {column: [column] for column in NUMERIC_FEATURE_COLUMNS}
    feature_groups.update(
        {
            column: [
                feature_name
                for feature_name in MODEL_FEATURE_COLUMNS
                if feature_name.startswith(f"{column}=")
            ]
            for column in CATEGORICAL_VALUES
        }
    )

    importance_rows = []
    for factor, encoded_features in feature_groups.items():
        feature_indices = [MODEL_FEATURE_COLUMNS.index(name) for name in encoded_features]
        reference_values = values.copy()
        if factor in NUMERIC_FEATURE_COLUMNS:
            reference = float(medians[feature_indices[0]])
            reference_values[:, feature_indices[0]] = reference
            reference_label = str(reference)
        else:
            category_frequencies = features[encoded_features].mean(axis=0).to_numpy()
            reference_index = int(np.argmax(category_frequencies))
            reference_values[:, feature_indices] = 0.0
            reference_values[:, feature_indices[reference_index]] = 1.0
            reference_label = encoded_features[reference_index].split("=", 1)[1]

        reference_probabilities = classifier(reference_values, training=False).numpy().reshape(-1)
        probability_changes = reference_probabilities - baseline_probabilities
        importance_rows.append(
            {
                "factor": factor,
                "reference_value": reference_label,
                "mean_absolute_probability_change": float(np.mean(np.abs(probability_changes))),
                "mean_probability_change_to_reference": float(np.mean(probability_changes)),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(importance_rows).to_csv(output_path, index=False)


def train(csv_path: Path, output_dir: Path, epochs: int) -> None:
    features, labels = read_training_data(csv_path)
    values = features.to_numpy(dtype="float32")
    medians = tf.convert_to_tensor(features.median().to_numpy(dtype="float32"))
    values = tf.where(tf.math.is_finite(values), values, medians)
    tf.keras.utils.set_random_seed(42)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(values)
    classifier = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(values.shape[1],)),
            normalizer,
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    classifier.compile(optimizer="adam", loss="binary_crossentropy")
    classifier.fit(values, labels, epochs=epochs, batch_size=16, verbose=0)

    predictor = SavedModelPredictor(classifier, medians)
    tf.saved_model.save(
        predictor,
        str(output_dir),
        signatures={"serving_default": predictor.serve},
    )
    write_feature_importance(
        classifier,
        features,
        values.numpy(),
        medians.numpy(),
        output_dir.parent / "feature_importance.csv",
    )
    print(f"Saved TensorFlow model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TensorFlow logistic model")
    parser.add_argument("--csv", type=Path, default=Path("data/sample_kidney_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/saved_model"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    train(args.csv, args.output, args.epochs)


if __name__ == "__main__":
    main()
