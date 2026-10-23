import os
import re
from pathlib import Path

from google.cloud import bigquery, storage


def required_env(name: str, pattern: str) -> str:
    value = os.environ.get(name, "")
    if not re.fullmatch(pattern, value):
        raise ValueError(f"{name} is missing or has an invalid format")
    return value


def main() -> None:
    project = required_env("GCP_PROJECT_ID", r"[a-z][a-z0-9-]{4,28}[a-z0-9]")
    bucket_name = required_env("GCP_BUCKET", r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
    dataset_id = required_env("BQ_DATASET", r"[A-Za-z_][A-Za-z0-9_]{0,1023}")
    model_id = required_env("BQ_MODEL", r"[A-Za-z_][A-Za-z0-9_]{0,1023}")
    location = os.getenv("BQ_LOCATION") or "US"
    deployment_id = os.getenv("GITHUB_SHA", "manual")
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts/saved_model"))
    if not (model_dir / "saved_model.pb").is_file():
        raise FileNotFoundError(f"No SavedModel found at {model_dir}")

    object_prefix = f"kidney-predictor/{deployment_id}/saved_model"
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            object_name = f"{object_prefix}/{file_path.relative_to(model_dir).as_posix()}"
            bucket.blob(object_name).upload_from_filename(file_path)

    bigquery_client = bigquery.Client(project=project, location=location)
    model_path = f"gs://{bucket_name}/{object_prefix}/*"
    statement = (
        f"CREATE OR REPLACE MODEL `{project}.{dataset_id}.{model_id}` "
        "OPTIONS (MODEL_TYPE = 'TENSORFLOW', "
        f"MODEL_PATH = '{model_path}')"
    )
    bigquery_client.query(statement).result()
    print(f"Registered BigQuery ML model `{project}.{dataset_id}.{model_id}`")


if __name__ == "__main__":
    main()
