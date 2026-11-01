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
    def __init__(
        self,
        classifier: tf.keras.Model,
        medians: np.ndarray,
    ) -> None:
        super().__init__()
        normalizer = next(
            layer
            for layer in classifier.layers
            if isinstance(layer, tf.keras.layers.Normalization)
        )
        dense_layers = [
            layer for layer in classifier.layers if isinstance(layer, tf.keras.layers.Dense)
        ]
        self.medians = tf.Variable(
            tf.convert_to_tensor(medians, dtype=tf.float32),
            trainable=False,
            name="feature_medians",
        )
        self.normalization_mean = tf.Variable(
            normalizer.mean.numpy().reshape(-1),
            trainable=False,
            name="normalization_mean",
        )
        self.normalization_variance = tf.Variable(
            normalizer.variance.numpy().reshape(-1),
            trainable=False,
            name="normalization_variance",
        )
        self.dense_kernels = [
            tf.Variable(layer.kernel, trainable=False, name=f"dense_kernel_{index}")
            for index, layer in enumerate(dense_layers)
        ]
        self.dense_biases = [
            tf.Variable(layer.bias, trainable=False, name=f"dense_bias_{index}")
            for index, layer in enumerate(dense_layers)
        ]

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
        features = (features - self.normalization_mean) * tf.math.rsqrt(
            tf.maximum(self.normalization_variance, 1e-7)
        )
        for index, (kernel, bias) in enumerate(zip(self.dense_kernels, self.dense_biases)):
            features = tf.linalg.matmul(features, kernel) + bias
            if index < len(self.dense_kernels) - 1:
                features = tf.nn.relu(features)
        return {"probability": tf.squeeze(tf.math.sigmoid(features), axis=-1)}


def stratified_split(labels: np.ndarray, validation_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    training_indices = []
    validation_indices = []
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        rng.shuffle(label_indices)
        validation_count = max(1, int(round(len(label_indices) * validation_fraction)))
        validation_indices.extend(label_indices[:validation_count])
        training_indices.extend(label_indices[validation_count:])
    return np.asarray(training_indices), np.asarray(validation_indices)


def write_feature_importance(
    classifier: tf.keras.Model,
    features: pd.DataFrame,
    values: np.ndarray,
    medians: np.ndarray,
    output_path: Path,
) -> None:
    baseline_probabilities = tf.math.sigmoid(
        classifier(values, training=False)
    ).numpy().reshape(-1)
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

        reference_probabilities = tf.math.sigmoid(
            classifier(reference_values, training=False)
        ).numpy().reshape(-1)
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
    raw_values = features.to_numpy(dtype="float32")
    training_indices, validation_indices = stratified_split(labels)
    medians = features.iloc[training_indices].median().to_numpy(dtype="float32")
    values = tf.where(tf.math.is_finite(raw_values), raw_values, medians).numpy()
    training_values = values[training_indices]
    validation_values = values[validation_indices]
    tf.keras.utils.set_random_seed(42)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(training_values)
    classifier = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(training_values.shape[1],)),
            normalizer,
            tf.keras.layers.Dense(
                1,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
        ]
    )
    classifier.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)],
    )
    classifier.fit(
        training_values,
        labels[training_indices],
        validation_data=(validation_values, labels[validation_indices]),
        epochs=epochs,
        batch_size=16,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
            )
        ],
    )
    validation_logits = classifier(validation_values, training=False).numpy().reshape(-1)
    validation_accuracy = float(
        np.mean((validation_logits >= 0.0) == labels[validation_indices].astype(bool))
    )

    predictor = SavedModelPredictor(classifier, medians)
    tf.saved_model.save(
        predictor,
        str(output_dir),
        signatures={"serving_default": predictor.serve},
    )
    write_feature_importance(
        classifier,
        features.iloc[training_indices],
        training_values,
        medians,
        output_dir.parent / "feature_importance.csv",
    )
    print(f"Saved TensorFlow model to {output_dir}")
    print(f"Stratified holdout accuracy: {validation_accuracy:.3f}")


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
