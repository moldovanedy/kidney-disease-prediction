from pathlib import Path

import pandas as pd
import numpy as np

NUMERIC_FEATURE_COLUMNS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc",
]
CATEGORICAL_VALUES = {
    "rbc": ("normal", "abnormal"),
    "pc": ("normal", "abnormal"),
    "pcc": ("present", "notpresent"),
    "ba": ("present", "notpresent"),
    "htn": ("yes", "no"),
    "dm": ("yes", "no"),
    "cad": ("yes", "no"),
    "appet": ("good", "poor"),
    "pe": ("yes", "no"),
    "ane": ("yes", "no"),
}
CATEGORICAL_FEATURE_COLUMNS = list(CATEGORICAL_VALUES)
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + [
    f"{column}={value}"
    for column, values in CATEGORICAL_VALUES.items()
    for value in (*values, "__missing__", "__unknown__")
]
TARGET_COLUMN = "ckd"


def read_features(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, na_values=["?"])
    return prepare_features(frame)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing feature columns: {', '.join(sorted(missing))}")

    numeric = pd.DataFrame(index=frame.index)
    for column in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric[column] = values.where(values.abs() != float("inf"))

    encoded = [numeric]
    for column, values in CATEGORICAL_VALUES.items():
        categories = frame[column].astype("string").str.strip().str.lower()
        categories = categories.fillna("__missing__")
        categories = categories.where(categories.isin(values), "__unknown__")
        encoded.append(
            pd.DataFrame(
                {
                    f"{column}={value}": categories.eq(value).astype("float32")
                    for value in (*values, "__missing__", "__unknown__")
                },
                index=frame.index,
            )
        )
    return pd.concat(encoded, axis=1)[MODEL_FEATURE_COLUMNS].astype("float32")


def read_training_data(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, na_values=["?"])
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"CSV must include the '{TARGET_COLUMN}' target column")

    features = prepare_features(frame)
    labels = frame[TARGET_COLUMN].astype("string").str.strip().str.lower()
    label_values = labels.map({"ckd": 1.0, "notckd": 0.0})
    if label_values.isna().any():
        raise ValueError(f"'{TARGET_COLUMN}' values must be 'ckd' or 'notckd'")
    return features, label_values.to_numpy(dtype="float32")
