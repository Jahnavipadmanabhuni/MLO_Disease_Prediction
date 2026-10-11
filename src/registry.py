"""MLflow experiment tracking + model registry helpers (champion/challenger pattern).

'champion' is an MLflow *alias* on a registered model version. The API always serves whatever
version currently holds that alias. A copy is also exported to models/champion.joblib so the API
(and CI tests) still work if the MLflow server is unreachable.
"""
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from .config import (
    CHAMPION_ALIAS, FEATURES, LOCAL_CHAMPION, LOCAL_CHAMPION_META, MODEL_NAME, MODELS_DIR,
    REFERENCE_DATA, TARGET, TRACKING_URI,
)


def init_mlflow(experiment: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


def champion_meta() -> dict | None:
    """Metadata of the current champion (version + the test metrics it was promoted with)."""
    try:
        mv = _client().get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None
    t = mv.tags or {}
    return {
        "version": str(mv.version),
        "run_id": mv.run_id,
        "model_type": t.get("model_type"),
        "f1_macro": float(t["f1_macro"]) if "f1_macro" in t else None,
        "accuracy": float(t["accuracy"]) if "accuracy" in t else None,
        "promoted_at": t.get("promoted_at"),
        "source": "mlflow",
    }


def load_champion():
    """Return (model, meta). Tries the MLflow registry first, then the local fallback file."""
    meta = champion_meta()
    if meta:
        try:
            return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"), meta
        except Exception as exc:  # artifact store unreachable, etc.
            print(f"[registry] MLflow load failed ({exc}); falling back to local copy")
    if LOCAL_CHAMPION.exists():
        meta = json.loads(LOCAL_CHAMPION_META.read_text()) if LOCAL_CHAMPION_META.exists() else {}
        meta["source"] = "local-file"
        return joblib.load(LOCAL_CHAMPION), meta
    raise FileNotFoundError("No champion model found. Run `python -m src.train` first.")


def register_model(run_id: str, model, example: pd.DataFrame, tags: dict) -> str:
    """Log the model into an existing run and register it as a new version. Returns the version."""
    with mlflow.start_run(run_id=run_id):
        info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME, input_example=example.head(3),
            serialization_format="cloudpickle",  # MLflow 3 defaults to skops, which rejects custom classes
        )
    version = str(info.registered_model_version)
    for k, v in tags.items():
        _client().set_model_version_tag(MODEL_NAME, version, k, str(v))
    return version


def promote(version: str, model, tags: dict, reference_df: pd.DataFrame):
    """Make `version` the champion, export the local fallback, and refresh the drift reference data."""
    tags = {**tags, "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    c = _client()
    for k, v in tags.items():
        c.set_model_version_tag(MODEL_NAME, version, k, str(v))
    c.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LOCAL_CHAMPION)
    LOCAL_CHAMPION_META.write_text(json.dumps({"version": version, **tags}, indent=2))

    REFERENCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    reference_df[FEATURES + [TARGET]].to_csv(REFERENCE_DATA, index=False)
