# MedPredict — Disease Prediction System with Monitoring (MLOps)

A machine-learning disease prediction service that **tracks experiments, versions models and data, serves predictions,
watches incoming patient data for drift, and retrains under strict guard-rails.**

Upgraded from the original *Flask + Decision Tree* app. The symptom dataset, disease metadata and dashboard UI are kept;
everything around the model is new.

| Requirement | Tool | Where |
|---|---|---|
| Disease prediction | Scikit-learn Decision Tree **vs** XGBoost (best chosen by cross-validation) | `src/modeling.py`, `src/train.py` |
| Experiment tracking & model versioning | **MLflow** (runs, metrics, Model Registry, `champion` alias) | `src/registry.py` |
| Patient dataset versioning | **DVC** (`dvc add`, `dvc.yaml`) — dataset hash is also logged to MLflow | `dvc.yaml`, `data/raw/` |
| Data drift & model monitoring | **Evidently AI** drift report + live-F1 check | `src/monitor.py` |
| Model serving | **FastAPI** (replaces Flask) | `src/api.py` |
| Containerization | **Docker** + docker-compose (API + MLflow server) | `Dockerfile`, `docker-compose.yml` |
| CI/CD | **GitHub Actions** (tests, pipeline smoke run, image build/push) | `.github/workflows/ci.yml` |
| Controlled retraining | Trigger → data gate → fair test → promotion gate → audit | `src/retrain.py` |

## How the pieces fit

```
        data/raw/disease_data.csv  ──(DVC)──►  src/train.py ──► MLflow runs + Model Registry
                                                                      │  alias: champion
                                                                      ▼
 patient ──► POST /predict  ◄───────────── FastAPI (src/api.py) loads champion
                 │  every prediction is logged          ▲
                 ▼                                      │ POST /admin/reload
      data/production/predictions.csv                   │
 clinician ─► POST /feedback (confirmed diagnosis)      │
                 ▼                                      │
          src/monitor.py (Evidently drift + live F1) ─► retrain_required?
                                                        │ yes
                                                        ▼
                                                  src/retrain.py
                       champion vs challenger on the SAME recent holdout → promote only if better
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train                  # trains DT + XGBoost, logs to MLflow, promotes champion v1
uvicorn src.api:app --reload         # dashboard at http://127.0.0.1:8000  (API docs at /docs)
mlflow ui --backend-store-uri sqlite:///mlflow.db        # experiments & registry at http://127.0.0.1:5000
```

### See the whole monitoring loop in one command
```bash
./demo.sh        # (Windows: run the commands inside it one by one)
```
It trains → sends **healthy** traffic (monitor says OK) → sends **drifted** traffic (outbreak + a new COVID variant
that rarely causes loss of smell) → monitor flags drift and a live-F1 drop → `retrain` promotes a new champion →
the same shifted traffic is now handled well. Open `reports/data_drift_report.html` for the Evidently report.

Typical output from the demo (synthetic data, so numbers are illustrative):

| Step | Result |
|---|---|
| Initial champion (XGBoost v1) | test macro-F1 ≈ 0.78 |
| Drifted traffic, champion v1 | 64 % of features drifted, live F1 ≈ 0.72 → **retrain required** |
| Retraining | champion F1 0.65 vs challenger 0.75 on recent holdout → **v2 promoted** |
| Drifted traffic, champion v2 | live F1 ≈ 0.80, no drift alarm |

### Using the real API instead of the simulator
```bash
python -m src.simulate_traffic --n 300 --drift 1.0 --api-url http://127.0.0.1:8000 --reset
python -m src.monitor
python -m src.retrain --reload-url http://127.0.0.1:8000      # hot-swaps the running API to the new champion
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | `{"fever":1,"cough":1,...}` → disease, confidence, severity, precautions, `prediction_id`, `model_version` |
| POST | `/feedback` | `{"prediction_id": "...", "true_label": "Dengue"}` — confirmed diagnosis, used for live accuracy |
| GET | `/health`, `/model-info` | liveness; which model version is serving |
| POST | `/admin/reload` | load the current champion without restarting |
| GET | `/monitoring/summary`, `/monitoring/report` | latest verdict (JSON) and the Evidently HTML report |

## Monitoring & retraining rules (all in `params.yaml`)

* **Drift**: Evidently tests every symptom column; retrain if ≥ 30 % of features drifted.
* **Performance**: once ≥ 50 diagnoses are confirmed, live macro-F1 is compared with the champion's test F1; retrain if it dropped ≥ 0.05.
  This catches *concept* drift (symptom → disease relationship changed) that feature drift alone can miss.
* **Retraining is controlled**, not automatic-and-hope:
  1. runs only when monitoring says so (or `--force`);
  2. needs enough confirmed diagnoses;
  3. champion and challenger are scored on the **same held-out slice of recent patients**;
  4. challenger is promoted only if it beats the champion by `min_improvement` — otherwise nothing changes;
  5. the decision is logged to MLflow and `reports/retrain_summary.json`;
  6. on promotion the drift baseline resets to the recent patient regime, production logs are archived, and the
     versioned dataset grows (then run `dvc add data/raw/disease_data.csv` and commit).

## Docker

```bash
docker compose up -d mlflow
docker compose run --rm api python -m src.train      # first champion, trained inside the stack
docker compose up -d api                              # http://localhost:8000   (MLflow UI: http://localhost:5000)
docker compose run --rm api python -m src.monitor     # run on a schedule (cron / Airflow) in a real deployment
```

## DVC

```bash
dvc init
dvc add data/raw/disease_data.csv         # versions the dataset; commit the generated .dvc file
git add data/raw/disease_data.csv.dvc .gitignore && git commit -m "track dataset v1"
dvc repro                                 # re-runs training only if data / code / params changed
# optional remote:  dvc remote add -d storage s3://bucket/path   (or a local folder / Google Drive)
```
The MD5 DVC stores for the dataset is the same `data_md5` tag logged on every MLflow run and registered model
version, so any model can be traced back to the exact data it was trained on.

## Tests & CI

```bash
pip install -r requirements-dev.txt && pytest -q
```
Tests train a model in a temp folder, exercise the API (predict, feedback, bad input), then run the full
simulate → monitor → retrain loop. GitHub Actions runs the same tests on every push, uploads the Evidently report as a
build artifact, and builds (and on `main` pushes) the Docker image.

## Project layout

```
src/
  config.py          paths, MLflow names, feature list
  data_gen.py        synthetic patients, with a --drift knob
  modeling.py        model wrappers + metrics
  registry.py        MLflow registry: register / promote / load champion
  train.py           train candidates, track, register, promote
  api.py             FastAPI service
  prod_log.py        logs predictions and confirmed diagnoses
  monitor.py         Evidently drift + live performance check
  retrain.py         guarded retraining
  simulate_traffic.py  demo traffic generator
  metadata.py        disease descriptions / precautions (from the old app)
params.yaml  dvc.yaml  Dockerfile  docker-compose.yml  .github/workflows/ci.yml  tests/  demo.sh
```

## Known limits

* Data is **synthetic** (generated from probability tables), so accuracy (~0.78 macro-F1) reflects overlapping symptoms
  between diseases, not real clinical performance. Swap in a real dataset by replacing `data/raw/disease_data.csv`
  and the feature list in `src/config.py`.
* This is a learning/portfolio project, **not a medical device**.
