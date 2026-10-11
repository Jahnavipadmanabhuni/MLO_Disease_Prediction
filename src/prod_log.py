"""Production traffic log: every prediction the API serves, plus confirmed diagnoses that arrive later.

This is the 'incoming patient data' that the monitoring step compares against the training reference.
"""
import shutil
import threading
import uuid
from datetime import datetime, timezone

import pandas as pd

from .config import FEATURES, LABELS_LOG, PREDICTIONS_LOG, PROD_DIR, TARGET

PRED_COLUMNS = ["prediction_id", "timestamp", "model_version"] + FEATURES + ["prediction", "confidence"]
LABEL_COLUMNS = ["prediction_id", "timestamp", "true_label"]
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _append(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        df.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_predictions(rows: pd.DataFrame):
    """rows must contain PRED_COLUMNS (timestamp is added when missing)."""
    rows = rows.copy()
    if "timestamp" not in rows:
        rows["timestamp"] = _now()
    _append(rows[PRED_COLUMNS], PREDICTIONS_LOG)


def log_labels(prediction_ids, true_labels):
    df = pd.DataFrame({"prediction_id": list(prediction_ids), "timestamp": _now(), "true_label": list(true_labels)})
    _append(df[LABEL_COLUMNS], LABELS_LOG)


def read_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_LOG) if PREDICTIONS_LOG.exists() else pd.DataFrame(columns=PRED_COLUMNS)


def read_labeled() -> pd.DataFrame:
    """Predictions joined with their confirmed diagnosis (TARGET column = ground truth)."""
    preds = read_predictions()
    if preds.empty or not LABELS_LOG.exists():
        return pd.DataFrame(columns=PRED_COLUMNS + [TARGET])
    labels = pd.read_csv(LABELS_LOG).drop_duplicates("prediction_id", keep="last")
    merged = preds.merge(labels[["prediction_id", "true_label"]], on="prediction_id", how="inner")
    return merged.rename(columns={"true_label": TARGET})


def archive_window() -> str | None:
    """Move the current logs into data/production/archive/<ts>/ so the next monitoring window starts clean."""
    if not PREDICTIONS_LOG.exists():
        return None
    dest = PROD_DIR / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest.mkdir(parents=True, exist_ok=True)
    for p in (PREDICTIONS_LOG, LABELS_LOG):
        if p.exists():
            shutil.move(str(p), dest / p.name)
    return str(dest)
