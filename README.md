# Kidney disease prediction template

A small Python app that trains a regularized TensorFlow logistic-regression model with a single output node, emits a probability for each input row, and can register its SavedModel in BigQuery ML.

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

(bq add-iam-policy-binding \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/bigquery.dataEditor" \
  "${GCP_PROJECT_ID}:${BQ_DATASET}")

OR

(bq show --format=prettyjson "${GCP_PROJECT_ID}:${BQ_DATASET}" \
  | jq --arg sa "${SA_EMAIL}" \
      '.access += [{"role":"WRITER","userByEmail":$sa}]' \
  > /tmp/dataset_updated.json
bq update --source /tmp/dataset_updated.json "${GCP_PROJECT_ID}:${BQ_DATASET}")

gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

The service account can now run BigQuery jobs, replace models in this dataset, and upload model files to this bucket. Do not create or download a service-account key.

### 5. Allow BigQuery to read the uploaded model

BigQuery reads the SavedModel from Cloud Storage using its project service agent, not the GitHub deployment identity. This is a Google-managed account, so you do not create it in the service-account page. First confirm `PROJECT_NUMBER` is set to the numeric project number (not the project ID), then run the identity command. Its output includes the service-agent email:

```bash
echo "$PROJECT_NUMBER"

gcloud beta services identity create \
  --service=bigquery.googleapis.com \
  --project="$GCP_PROJECT_ID"

export BIGQUERY_SERVICE_AGENT="service-${PROJECT_NUMBER}@gcp-sa-bigquery.iam.gserviceaccount.com"
echo "$BIGQUERY_SERVICE_AGENT"

gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET}" \
  --member="serviceAccount:${BIGQUERY_SERVICE_AGENT}" \
  --role="roles/storage.objectViewer"
```

If the identity command fails, make sure step 2 completed successfully and that `GCP_PROJECT_ID` names the intended project. Re-read the project number with `gcloud projects describe "$GCP_PROJECT_ID" --format='value(projectNumber)'`. If your installed `gcloud` has no `beta` component, install/enable the Cloud SDK beta component or use the Google Cloud Console's **IAM & Admin > IAM** page with Google-managed service accounts visible; the expected account is `service-PROJECT_NUMBER@gcp-sa-bigquery.iam.gserviceaccount.com`. If not, replace with one that exists.

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

### Score rows and rank factors in BigQuery

Load the bundled CSV into a BigQuery input table. Substitute your project and dataset IDs:

```bash
bq load --autodetect --replace --source_format=CSV --skip_leading_rows=1 \
  your-project:kidney_prediction.kidney_input \
  data/sample_kidney_data.csv
```

Run the following in BigQuery Studio after deployment. It encodes the categorical columns in the SavedModel's 54-value input order and reads numeric medians from the table published by the training job. Replace the three-part table names with your project, dataset, and configured model IDs:

```sql
WITH medians AS (
  SELECT
    MAX(IF(factor = 'age', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS age,
    MAX(IF(factor = 'bp', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS bp,
    MAX(IF(factor = 'sg', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS sg,
    MAX(IF(factor = 'al', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS al,
    MAX(IF(factor = 'su', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS su,
    MAX(IF(factor = 'bgr', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS bgr,
    MAX(IF(factor = 'bu', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS bu,
    MAX(IF(factor = 'sc', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS sc,
    MAX(IF(factor = 'sod', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS sod,
    MAX(IF(factor = 'pot', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS pot,
    MAX(IF(factor = 'hemo', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS hemo,
    MAX(IF(factor = 'pcv', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS pcv,
    MAX(IF(factor = 'wc', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS wc,
    MAX(IF(factor = 'rc', SAFE_CAST(reference_value AS FLOAT64), NULL)) AS rc
  FROM `your-project.kidney_prediction.kidney_logistic_regression_feature_importance`
), cleaned AS (
  SELECT
    *,
    NULLIF(NULLIF(LOWER(TRIM(CAST(rbc AS STRING))), ''), '?') AS rbc_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(pc AS STRING))), ''), '?') AS pc_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(pcc AS STRING))), ''), '?') AS pcc_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(ba AS STRING))), ''), '?') AS ba_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(htn AS STRING))), ''), '?') AS htn_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(dm AS STRING))), ''), '?') AS dm_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(cad AS STRING))), ''), '?') AS cad_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(appet AS STRING))), ''), '?') AS appet_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(pe AS STRING))), ''), '?') AS pe_value,
    NULLIF(NULLIF(LOWER(TRIM(CAST(ane AS STRING))), ''), '?') AS ane_value
  FROM `your-project.kidney_prediction.kidney_input`
), encoded AS (
  SELECT
    c.id,
    ARRAY<FLOAT64>[
      COALESCE(SAFE_CAST(c.age AS FLOAT64), m.age),
      COALESCE(SAFE_CAST(c.bp AS FLOAT64), m.bp),
      COALESCE(SAFE_CAST(c.sg AS FLOAT64), m.sg),
      COALESCE(SAFE_CAST(c.al AS FLOAT64), m.al),
      COALESCE(SAFE_CAST(c.su AS FLOAT64), m.su),
      COALESCE(SAFE_CAST(c.bgr AS FLOAT64), m.bgr),
      COALESCE(SAFE_CAST(c.bu AS FLOAT64), m.bu),
      COALESCE(SAFE_CAST(c.sc AS FLOAT64), m.sc),
      COALESCE(SAFE_CAST(c.sod AS FLOAT64), m.sod),
      COALESCE(SAFE_CAST(c.pot AS FLOAT64), m.pot),
      COALESCE(SAFE_CAST(c.hemo AS FLOAT64), m.hemo),
      COALESCE(SAFE_CAST(c.pcv AS FLOAT64), m.pcv),
      COALESCE(SAFE_CAST(c.wc AS FLOAT64), m.wc),
      COALESCE(SAFE_CAST(c.rc AS FLOAT64), m.rc),
      IF(c.rbc_value = 'normal', 1.0, 0.0),
      IF(c.rbc_value = 'abnormal', 1.0, 0.0),
      IF(c.rbc_value IS NULL, 1.0, 0.0),
      IF(c.rbc_value IS NOT NULL AND c.rbc_value NOT IN ('normal', 'abnormal'), 1.0, 0.0),
      IF(c.pc_value = 'normal', 1.0, 0.0),
      IF(c.pc_value = 'abnormal', 1.0, 0.0),
      IF(c.pc_value IS NULL, 1.0, 0.0),
      IF(c.pc_value IS NOT NULL AND c.pc_value NOT IN ('normal', 'abnormal'), 1.0, 0.0),
      IF(c.pcc_value = 'present', 1.0, 0.0),
      IF(c.pcc_value = 'notpresent', 1.0, 0.0),
      IF(c.pcc_value IS NULL, 1.0, 0.0),
      IF(c.pcc_value IS NOT NULL AND c.pcc_value NOT IN ('present', 'notpresent'), 1.0, 0.0),
      IF(c.ba_value = 'present', 1.0, 0.0),
      IF(c.ba_value = 'notpresent', 1.0, 0.0),
      IF(c.ba_value IS NULL, 1.0, 0.0),
      IF(c.ba_value IS NOT NULL AND c.ba_value NOT IN ('present', 'notpresent'), 1.0, 0.0),
      IF(c.htn_value = 'yes', 1.0, 0.0),
      IF(c.htn_value = 'no', 1.0, 0.0),
      IF(c.htn_value IS NULL, 1.0, 0.0),
      IF(c.htn_value IS NOT NULL AND c.htn_value NOT IN ('yes', 'no'), 1.0, 0.0),
      IF(c.dm_value = 'yes', 1.0, 0.0),
      IF(c.dm_value = 'no', 1.0, 0.0),
      IF(c.dm_value IS NULL, 1.0, 0.0),
      IF(c.dm_value IS NOT NULL AND c.dm_value NOT IN ('yes', 'no'), 1.0, 0.0),
      IF(c.cad_value = 'yes', 1.0, 0.0),
      IF(c.cad_value = 'no', 1.0, 0.0),
      IF(c.cad_value IS NULL, 1.0, 0.0),
      IF(c.cad_value IS NOT NULL AND c.cad_value NOT IN ('yes', 'no'), 1.0, 0.0),
      IF(c.appet_value = 'good', 1.0, 0.0),
      IF(c.appet_value = 'poor', 1.0, 0.0),
      IF(c.appet_value IS NULL, 1.0, 0.0),
      IF(c.appet_value IS NOT NULL AND c.appet_value NOT IN ('good', 'poor'), 1.0, 0.0),
      IF(c.pe_value = 'yes', 1.0, 0.0),
      IF(c.pe_value = 'no', 1.0, 0.0),
      IF(c.pe_value IS NULL, 1.0, 0.0),
      IF(c.pe_value IS NOT NULL AND c.pe_value NOT IN ('yes', 'no'), 1.0, 0.0),
      IF(c.ane_value = 'yes', 1.0, 0.0),
      IF(c.ane_value = 'no', 1.0, 0.0),
      IF(c.ane_value IS NULL, 1.0, 0.0),
      IF(c.ane_value IS NOT NULL AND c.ane_value NOT IN ('yes', 'no'), 1.0, 0.0)
    ] AS features
  FROM cleaned AS c
  CROSS JOIN medians AS m
)
SELECT id, probability AS ckd_risk_probability
FROM ML.PREDICT(
  MODEL `your-project.kidney_prediction.kidney_logistic_regression`,
  (SELECT id, features FROM encoded)
);
```

`ckd_risk_probability` is the model's estimated probability for the positive `ckd` class. The training job also writes model-derived, global factor impact values to `<BQ_MODEL>_feature_importance`. The impact compares each training row's probability with the model's probability after that factor is replaced by its training median (numeric) or most frequent category (categorical). Query the largest impacts with:

```sql
SELECT
  factor,
  reference_value,
  ROUND(mean_absolute_probability_change * 100, 2) AS mean_absolute_change_percentage_points,
  ROUND(mean_probability_change_to_reference * 100, 2) AS mean_signed_change_percentage_points
FROM `your-project.kidney_prediction.kidney_logistic_regression_feature_importance`
ORDER BY mean_absolute_probability_change DESC
LIMIT 10;
```

These are model/data-dependent associations, not causal effects or clinical advice. The bundled data is only a software example; this model has not been validated for clinical use.
