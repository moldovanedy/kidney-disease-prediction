import argparse
from pathlib import Path

import tensorflow as tf

from kidney_predictor.schema import MODEL_FEATURE_COLUMNS, read_training_data


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
        return {"probability": self.classifier(features)}


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
