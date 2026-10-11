"""FastAPI model server (replaces the Flask app).

    uvicorn src.api:app --reload

Endpoints
  GET  /                      dashboard UI (same page as before)
  POST /predict               symptoms -> disease; every call is logged for monitoring
  POST /feedback              confirmed diagnosis for an earlier prediction (ground truth for live accuracy)
  GET  /health, /model-info   liveness + which model version is serving
  POST /admin/reload          hot-swap to the current champion (call after a retraining promotion)
  GET  /monitoring/summary    latest drift/performance verdict;  /monitoring/report = Evidently HTML
"""
import json
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import prod_log, registry
from .config import CODE_ROOT, DRIFT_REPORT_HTML, FEATURES, MONITOR_SUMMARY
from .metadata import DEFAULT_METADATA, DISEASE_METADATA

_state = {"model": None, "meta": {}}
_state_lock = threading.Lock()


def load_model():
    model, meta = registry.load_champion()
    with _state_lock:
        _state["model"], _state["meta"] = model, meta
    print(f"[api] serving model v{meta.get('version')} ({meta.get('model_type')}) from {meta.get('source')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="MedPredict MLOps API", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=CODE_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=str(CODE_ROOT / "templates"))


def _error(msg: str, code: int = 400):
    # keep the {"error": ...} shape the existing frontend already understands
    return JSONResponse({"error": msg}, status_code=code)


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/model-info")
def model_info():
    return _state["meta"]


@app.post("/predict")
def predict(payload: dict = Body(...)):
    if not payload:
        return _error("No input data provided")
    values = {}
    for f in FEATURES:
        try:
            v = int(payload.get(f, 0))
            if v not in (0, 1):
                raise ValueError
        except (ValueError, TypeError):
            return _error(f"Invalid value for symptom '{f}'. Must be 0 or 1.")
        values[f] = v

    with _state_lock:
        model, meta = _state["model"], _state["meta"]
    X = pd.DataFrame([values])[FEATURES]
    pred = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(proba[list(model.classes_).index(pred)])

    pid = prod_log.new_id()
    prod_log.log_predictions(pd.DataFrame([{
        "prediction_id": pid, "model_version": meta.get("version"), **values,
        "prediction": pred, "confidence": round(confidence, 4),
    }]))

    info = DISEASE_METADATA.get(pred, DEFAULT_METADATA)
    return {
        "success": True,
        "prediction_id": pid,
        "model_version": meta.get("version"),
        "prediction": pred,
        "confidence": round(confidence * 100, 1),
        "description": info["description"],
        "severity": info["severity"],
        "precautions": info["precautions"],
        "symptoms_analyzed": values,
    }


@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    pid, label = payload.get("prediction_id"), payload.get("true_label")
    if not pid or not label:
        return _error("Both 'prediction_id' and 'true_label' are required")
    with _state_lock:
        known = [str(c) for c in _state["model"].classes_]
    if label not in known:
        return _error(f"Unknown disease '{label}'. Expected one of {known}")
    prod_log.log_labels([pid], [label])
    return {"success": True}


@app.post("/admin/reload")
def reload_model():
    load_model()
    return {"success": True, "serving": _state["meta"]}


@app.get("/monitoring/summary")
def monitoring_summary():
    if not MONITOR_SUMMARY.exists():
        return _error("No monitoring run yet. Run `python -m src.monitor`.", 404)
    return json.loads(MONITOR_SUMMARY.read_text())


@app.get("/monitoring/report")
def monitoring_report():
    if not DRIFT_REPORT_HTML.exists():
        return _error("No drift report yet. Run `python -m src.monitor`.", 404)
    return FileResponse(DRIFT_REPORT_HTML, media_type="text/html")
