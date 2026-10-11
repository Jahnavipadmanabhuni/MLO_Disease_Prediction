"""Central configuration: paths, MLflow settings, feature list."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent          # where the code/templates live
ROOT = Path(os.getenv("MLOPS_HOME", CODE_ROOT))             # where data/models/reports are written

RAW_DATA = ROOT / "data" / "raw" / "disease_data.csv"       # versioned with DVC
REFERENCE_DATA = ROOT / "data" / "reference" / "reference.csv"
PROD_DIR = ROOT / "data" / "production"
PREDICTIONS_LOG = PROD_DIR / "predictions.csv"
LABELS_LOG = PROD_DIR / "labels.csv"
MODELS_DIR = ROOT / "models"
LOCAL_CHAMPION = MODELS_DIR / "champion.joblib"             # fallback copy of the champion model
LOCAL_CHAMPION_META = MODELS_DIR / "champion_meta.json"
REPORTS_DIR = ROOT / "reports"
MONITOR_SUMMARY = REPORTS_DIR / "monitoring_summary.json"
DRIFT_REPORT_HTML = REPORTS_DIR / "data_drift_report.html"
RETRAIN_SUMMARY = REPORTS_DIR / "retrain_summary.json"
PARAMS_FILE = CODE_ROOT / "params.yaml"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
TRAIN_EXPERIMENT = "disease-prediction"
MONITOR_EXPERIMENT = "disease-monitoring"
MODEL_NAME = "disease-classifier"
CHAMPION_ALIAS = "champion"

TARGET = "disease"
FEATURES = [
    "fever", "cough", "headache", "fatigue", "sore_throat", "runny_nose", "body_ache",
    "shortness_of_breath", "loss_of_taste_smell", "itchy_eyes", "sneezing", "nausea",
    "joint_pain", "skin_rash",
]
