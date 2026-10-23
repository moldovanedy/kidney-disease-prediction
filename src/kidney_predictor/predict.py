import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from kidney_predictor.schema import read_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CKD probabilities locally")
    parser.add_argument("--input", type=Path, required=True, help="CSV with feature columns")
    parser.add_argument("--model", type=Path, default=Path("artifacts/saved_model"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    features = read_features(args.input)
    model = tf.saved_model.load(str(args.model))
    predict = model.signatures["serving_default"]
    result = predict(features=tf.convert_to_tensor(features.to_numpy(dtype=np.float32)))
    probabilities = result["probability"].numpy().reshape(-1)

    output = pd.read_csv(args.input)
    output["ckd_probability"] = probabilities
    output.to_csv(args.output, index=False)
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
