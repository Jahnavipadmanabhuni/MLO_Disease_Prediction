"""Monitoring: compare live traffic against the training reference and decide if retraining is needed.

    python -m src.monitor            # writes reports/data_drift_report.html + reports/monitoring_summary.json

Two independent signals:
  1. DATA DRIFT   (Evidently) - did incoming symptom patterns change vs. what the model was trained on?
  2. PERFORMANCE  - once confirmed diagnoses arrive via /feedback, is live macro-F1 falling below
                    the champion's test F1?  (This catches concept drift that feature drift can miss.)
"""
import argparse
import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, f1_score

from . import prod_log, registry
from .config import (
    DRIFT_REPORT_HTML, FEATURES, MONITOR_EXPERIMENT, MONITOR_SUMMARY, REFERENCE_DATA, REPORTS_DIR, TARGET,
)
from .modeling import load_params


def run_drift(reference: pd.DataFrame, current: pd.DataFrame):
    """Run Evidently's DataDriftPreset on the symptom columns. Returns (share, drifted_features, p_values, snapshot)."""
    definition = DataDefinition(categorical_columns=FEATURES)
    ref_ds = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    cur_ds = Dataset.from_pandas(current[FEATURES], data_definition=definition)
    snapshot = Report([DataDriftPreset()]).run(cur_ds, ref_ds)

    drifted, scores = [], {}
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg.get("type", "").endswith("ValueDrift"):
            col, thr, val = cfg["column"], cfg["threshold"], m["value"]
            scores[col] = round(float(val), 5)
            is_drift = val < thr if "p_value" in cfg["method"] else val >= thr
            if is_drift:
                drifted.append(col)
    share = len(drifted) / len(FEATURES)
    return share, drifted, scores, snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit-code", action="store_true", help="exit with status 2 when retraining is required (for CI/cron)")
    args = ap.parse_args()

    params = load_params()["monitoring"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "ok", "retrain_required": False, "reasons": [],
    }

    if not REFERENCE_DATA.exists():
        raise SystemExit("No reference data. Run `python -m src.train` first.")
    reference = pd.read_csv(REFERENCE_DATA)
    current = prod_log.read_predictions()
    champ = registry.champion_meta() or {}
    summary.update(champion_version=champ.get("version"), n_reference=len(reference), n_current=len(current))

    if len(current) < params["min_current_rows"]:
        summary["status"] = "insufficient_data"
        summary["reasons"].append(f"only {len(current)} logged predictions; need {params['min_current_rows']}")
    else:
        share, drifted, scores, snapshot = run_drift(reference, current)
        snapshot.save_html(str(DRIFT_REPORT_HTML))
        summary["drift"] = {"share_of_features_drifted": round(share, 3), "n_drifted": len(drifted),
                            "drifted_features": drifted, "p_values": scores}
        if share >= params["drift_share_threshold"]:
            summary["retrain_required"] = True
            summary["status"] = "drift_detected"
            summary["reasons"].append(f"{share:.0%} of features drifted (threshold {params['drift_share_threshold']:.0%})")

        labeled = prod_log.read_labeled()
        if len(labeled) >= params["min_labeled_rows"]:
            live_f1 = float(f1_score(labeled[TARGET], labeled["prediction"], average="macro", zero_division=0))
            live_acc = float(accuracy_score(labeled[TARGET], labeled["prediction"]))
            base = champ.get("f1_macro")
            perf = {"n_labeled": len(labeled), "live_f1_macro": round(live_f1, 4), "live_accuracy": round(live_acc, 4),
                    "baseline_f1_macro": base}
            if base is not None:
                perf["f1_drop"] = round(base - live_f1, 4)
                if base - live_f1 >= params["f1_drop_threshold"]:
                    summary["retrain_required"] = True
                    summary["status"] = "performance_degraded" if summary["status"] == "ok" else summary["status"]
                    summary["reasons"].append(
                        f"live F1 {live_f1:.3f} is {base - live_f1:.3f} below champion test F1 {base:.3f}")
            summary["performance"] = perf
        else:
            summary["performance"] = {"n_labeled": len(labeled),
                                      "note": f"need {params['min_labeled_rows']} confirmed diagnoses to judge live accuracy"}

    MONITOR_SUMMARY.write_text(json.dumps(summary, indent=2))

    # log the monitoring run in MLflow too, so drift history sits next to the training history
    registry.init_mlflow(MONITOR_EXPERIMENT)
    with mlflow.start_run(run_name="monitor"):
        mlflow.set_tags({"status": summary["status"], "retrain_required": str(summary["retrain_required"]),
                         "champion_version": str(summary["champion_version"])})
        mlflow.log_metric("n_current", summary["n_current"])
        if "drift" in summary:
            mlflow.log_metric("drift_share", summary["drift"]["share_of_features_drifted"])
            mlflow.log_artifact(str(DRIFT_REPORT_HTML))
        perf = summary.get("performance", {})
        for k in ("live_f1_macro", "live_accuracy", "f1_drop"):
            if k in perf:
                mlflow.log_metric(k, perf[k])
        mlflow.log_artifact(str(MONITOR_SUMMARY))

    print(json.dumps(summary, indent=2))
    if args.exit_code and summary["retrain_required"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
