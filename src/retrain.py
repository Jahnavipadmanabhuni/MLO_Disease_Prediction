"""Controlled retraining: retrain ONLY when monitoring says so, and promote ONLY if the challenger wins.

    python -m src.retrain               # obeys reports/monitoring_summary.json
    python -m src.retrain --force       # retrain even if monitoring did not ask for it
    python -m src.retrain --reload-url http://localhost:8000   # hot-swap the running API afterwards

Guard rails (this is what makes it 'controlled'):
  1. Trigger   - drift / performance verdict from monitor.py (or --force)
  2. Data gate - needs enough confirmed diagnoses from production
  3. Fair test - champion and challenger are scored on the SAME held-out slice of recent patients
  4. Promotion - challenger must beat the champion's F1 by `min_improvement`; otherwise nothing changes
  5. Audit     - every decision is logged to MLflow + reports/retrain_summary.json
"""
import argparse
import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from . import prod_log, registry
from .config import FEATURES, MONITOR_SUMMARY, RAW_DATA, RETRAIN_SUMMARY, TARGET, TRAIN_EXPERIMENT
from .modeling import evaluate, file_md5, load_params
from .train import fit_candidates


def finish(summary: dict, code: int = 0):
    summary["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RETRAIN_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    RETRAIN_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reload-url", default=None)
    args = ap.parse_args()

    params = load_params()
    seed = params["data"]["seed"]
    summary = {"retrained": False, "promoted": False}

    # 1. trigger ---------------------------------------------------------------------------
    if not MONITOR_SUMMARY.exists() and not args.force:
        summary["reason"] = "no monitoring summary; run `python -m src.monitor` first"
        finish(summary, 1)
    verdict = json.loads(MONITOR_SUMMARY.read_text()) if MONITOR_SUMMARY.exists() else {}
    if not verdict.get("retrain_required") and not args.force:
        summary["reason"] = f"monitoring status is '{verdict.get('status')}'; no retraining needed"
        finish(summary)
    champ_now = (registry.champion_meta() or {}).get("version")
    if verdict and not args.force and str(verdict.get("champion_version")) != str(champ_now):
        summary["reason"] = "monitoring summary refers to an older champion; run `python -m src.monitor` again"
        finish(summary, 1)
    trigger = "; ".join(verdict.get("reasons", [])) or "forced"

    # 2. data gate -------------------------------------------------------------------------
    labeled = prod_log.read_labeled()
    need = params["monitoring"]["min_labeled_rows"]
    if len(labeled) < need:
        summary["reason"] = f"only {len(labeled)} confirmed diagnoses; need {need} before retraining is safe"
        finish(summary, 1)

    recent = labeled[FEATURES + [TARGET]].reset_index(drop=True)
    strat = recent[TARGET] if recent[TARGET].value_counts().min() >= 2 else None
    recent_train, holdout = train_test_split(
        recent, test_size=params["retraining"]["holdout_fraction"], random_state=seed, stratify=strat)
    base = pd.read_csv(RAW_DATA)
    train_df = pd.concat([base[FEATURES + [TARGET]], recent_train], ignore_index=True)

    # 3. fair test -------------------------------------------------------------------------
    champ_model, champ_meta = registry.load_champion()
    champ_metrics = evaluate(champ_model, holdout[FEATURES], holdout[TARGET])
    print(f"Trigger: {trigger}")
    print(f"Champion v{champ_meta.get('version')} on recent holdout: F1={champ_metrics['f1_macro']:.3f}")

    registry.init_mlflow(TRAIN_EXPERIMENT)
    tags = {"stage": "retraining", "trigger": trigger[:250], "base_data_md5": file_md5(RAW_DATA),
            "recent_labeled_rows": len(recent_train)}
    results, best = fit_candidates(train_df, holdout, params, seed, tags, "retrain")
    chal = results[best]
    gain = chal["metrics"]["f1_macro"] - champ_metrics["f1_macro"]
    promote = gain >= params["retraining"]["min_improvement"]

    summary.update(
        retrained=True, trigger=trigger, challenger=best,
        champion_version=champ_meta.get("version"),
        champion_holdout_f1=round(champ_metrics["f1_macro"], 4),
        challenger_holdout_f1=round(chal["metrics"]["f1_macro"], 4),
        f1_gain=round(gain, 4), required_gain=params["retraining"]["min_improvement"],
        train_rows=len(train_df), holdout_rows=len(holdout), promoted=promote,
    )

    # 4. promotion -------------------------------------------------------------------------
    if promote:
        mtags = {"model_type": best, "f1_macro": round(chal["metrics"]["f1_macro"], 4),
                 "accuracy": round(chal["metrics"]["accuracy"], 4), "trigger": "retraining"}
        version = registry.register_model(chal["run_id"], chal["model"], train_df[FEATURES], mtags)
        # new drift baseline = the recent patient regime this model was promoted for (not the old mix),
        # otherwise the same shifted traffic would keep re-triggering retraining
        registry.promote(version, chal["model"], mtags, recent)
        summary["new_champion_version"] = version

        # grow the versioned dataset with ALL newly confirmed patients -> `dvc add` it to snapshot the new data version
        pd.concat([base[FEATURES + [TARGET]], recent], ignore_index=True).to_csv(RAW_DATA, index=False)
        summary["dataset_updated"] = str(RAW_DATA)
        summary["archived_logs"] = prod_log.archive_window()
        # the old verdict is now obsolete -> reset it so nobody retrains twice on the same alarm
        MONITOR_SUMMARY.write_text(json.dumps({"status": "retrained_awaiting_new_data", "retrain_required": False,
                                               "champion_version": version}, indent=2))
        summary["next_steps"] = ["dvc add data/raw/disease_data.csv", "git add data/raw/disease_data.csv.dvc && git commit"]

        if args.reload_url:
            import httpx
            httpx.post(f"{args.reload_url}/admin/reload", timeout=10).raise_for_status()
            summary["api_reloaded"] = True
    else:
        summary["reason"] = "challenger did not beat the champion by the required margin; champion unchanged"

    # 5. audit -----------------------------------------------------------------------------
    with mlflow.start_run(run_name="retrain-decision"):
        mlflow.set_tags({"promoted": str(promote), "trigger": trigger[:250]})
        mlflow.log_metrics({"champion_holdout_f1": champ_metrics["f1_macro"],
                            "challenger_holdout_f1": chal["metrics"]["f1_macro"], "f1_gain": gain})
    finish(summary)


if __name__ == "__main__":
    main()
