# Kidney disease prediction template

A small Python app that trains a TensorFlow binary logistic-regression model, emits a probability for each input row, and can register its SavedModel in BigQuery ML.

## Local use

Requires Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
python -m kidney_predictor.train
python -m kidney_predictor.predict --input data/sample_kidney_data.csv
```

## GitHub Actions deployment

`.github/workflows/deploy-bigquery-ml.yml` trains and deploys on pushes to `main`, and can also be started with **Actions > Train and deploy BigQuery ML > Run workflow**. It uses GitHub OIDC with Google Workload Identity Federation; no service-account key is stored in GitHub.

### 1. Create/select a GCP project

Create a project in the Google Cloud Console, attach an active billing account, and select it in Cloud Shell or a local `gcloud` terminal. The identity running the setup commands needs permission to enable APIs, create resources, and grant IAM roles.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

export GCP_PROJECT_ID="YOUR_PROJECT_ID"
export PROJECT_NUMBER="$(gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)')"
export BQ_LOCATION="US"
export GCP_BUCKET="YOUR_GLOBALLY_UNIQUE_LOWERCASE_BUCKET_NAME"
export BQ_DATASET="kidney_prediction"
export BQ_MODEL="kidney_logistic_regression"
export SA_NAME="github-kidney-deployer"
export SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
export POOL_ID="github-actions"
export PROVIDER_ID="github-provider"
export GITHUB_REPOSITORY="OWNER/REPO"
```

Replace every `YOUR_...` and `OWNER/REPO` value. `GCP_BUCKET` must be globally unique. Use the same supported location for the bucket and dataset; `US` is the BigQuery multi-region, not the `us-central1` region.

### 2. Enable APIs

```bash
gcloud services enable \
  bigquery.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  --project="$GCP_PROJECT_ID"
```

### 3. Create the bucket and dataset

```bash
gcloud storage buckets create "gs://${GCP_BUCKET}" \
  --project="$GCP_PROJECT_ID" \
  --location="$BQ_LOCATION" \
  --uniform-bucket-level-access

bq --location="$BQ_LOCATION" mk --dataset "${GCP_PROJECT_ID}:${BQ_DATASET}"
```

The bucket stores each commit's SavedModel files. The dataset stores the imported BigQuery ML model.

### 4. Create a deployment service account and grant its access

```bash
gcloud iam service-accounts create "$SA_NAME" \
  --project="$GCP_PROJECT_ID" \
  --display-name="GitHub Actions kidney model deployer"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.jobUser"

bq add-iam-policy-binding \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataEditor" \
  "${GCP_PROJECT_ID}:${BQ_DATASET}"

gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

The service account can now run BigQuery jobs, replace models in this dataset, and upload model files to this bucket. Do not create or download a service-account key.

### 5. Allow BigQuery to read the uploaded model

BigQuery reads the SavedModel from Cloud Storage using its project service agent, not the GitHub deployment identity. Enable the BigQuery service identity and grant it read-only access to the bucket:

```bash
gcloud beta services identity create \
  --service=bigquery.googleapis.com \
  --project="$GCP_PROJECT_ID"

export BIGQUERY_SERVICE_AGENT="service-${PROJECT_NUMBER}@gcp-sa-bigquery.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET}" \
  --member="serviceAccount:${BIGQUERY_SERVICE_AGENT}" \
  --role="roles/storage.objectViewer"
```

If your installed `gcloud` has no `beta` component, enable the BigQuery API first and use the service agent shown under **IAM & Admin > IAM > Google-managed service accounts**. The address normally has the form used above.

### 6. Configure GitHub Workload Identity Federation

Create a pool and OIDC provider. Restrict the provider to the exact repository and `main` branch so another repository or branch cannot impersonate this service account:

```bash
gcloud iam workload-identity-pools create "$POOL_ID" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="$POOL_ID" \
  --display-name="GitHub repository provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository=='${GITHUB_REPOSITORY}' && assertion.ref=='refs/heads/main'"

export WIF_PRINCIPAL="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${GITHUB_REPOSITORY}"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$GCP_PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="$WIF_PRINCIPAL"
```

Get the provider resource name for the GitHub variable:

```bash
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"
echo "$SA_EMAIL"
```

### 7. Add GitHub repository variables

In GitHub, open **Settings > Secrets and variables > Actions > Variables**, then add these repository variables (they are identifiers, not secrets):

| Variable              | Value                                              |
| --------------------- | -------------------------------------------------- |
| `GCP_WIF_PROVIDER`    | Full provider resource name printed in step 6      |
| `GCP_SERVICE_ACCOUNT` | Service-account email printed in step 6            |
| `GCP_PROJECT_ID`      | Project ID                                         |
| `GCP_BUCKET`          | Bucket name only, without `gs://`                  |
| `BQ_DATASET`          | Dataset ID, for example `kidney_prediction`        |
| `BQ_MODEL`            | Model ID, for example `kidney_logistic_regression` |
| `BQ_LOCATION`         | Dataset and bucket location, for example `US`      |

### 8. Deploy and verify

Commit and push the reviewed changes to the repository's `main` branch. In GitHub, open **Actions > Train and deploy BigQuery ML** and confirm the latest run succeeds. You can also select **Run workflow**, choosing `main` as the branch. Each successful run uploads to `gs://<bucket>/kidney-predictor/<commit>/saved_model/` and creates or replaces `${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_MODEL}`.

To verify the model exists, run this in BigQuery Studio (substitute your project, dataset, and model IDs):

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `your-project.your_dataset.kidney_logistic_regression`,
  (SELECT ARRAY<FLOAT64>[/* 54 preprocessed values, in MODEL_FEATURE_COLUMNS order */]
   AS features)
);
```

The `features` array contains the 14 numeric values followed by one-hot values for each categorical field, in `MODEL_FEATURE_COLUMNS` order. Missing numeric entries can be passed as `CAST('NaN' AS FLOAT64)` and are imputed by the model. This is a software template, not a validated medical model or a clinical decision tool.
