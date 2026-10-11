"""Simulate incoming patients so you can demo monitoring without a real hospital feed.

    python -m src.simulate_traffic --n 300 --drift 0.0        # healthy traffic  -> monitor says OK
    python -m src.simulate_traffic --n 300 --drift 1.0        # changed world    -> monitor flags drift
    python -m src.simulate_traffic --n 300 --drift 1.0 --api-url http://localhost:8000   # go through the live API
"""
import argparse

import pandas as pd

from . import prod_log, registry
from .config import FEATURES, TARGET
from .data_gen import generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--label-rate", type=float, default=1.0, help="share of patients whose true diagnosis is confirmed")
    ap.add_argument("--reset", action="store_true", help="archive existing production logs first (clean demo window)")
    ap.add_argument("--api-url", default=None, help="if set, send traffic through the running FastAPI service")
    args = ap.parse_args()

    if args.reset:
        prod_log.archive_window()
    patients = generate(args.n, seed=args.seed, drift=args.drift)
    n_labeled = int(len(patients) * args.label_rate)

    if args.api_url:
        import httpx
        with httpx.Client(base_url=args.api_url, timeout=10) as c:
            for i, row in patients.iterrows():
                r = c.post("/predict", json={f: int(row[f]) for f in FEATURES}).json()
                if i < n_labeled:
                    c.post("/feedback", json={"prediction_id": r["prediction_id"], "true_label": row[TARGET]})
    else:
        model, meta = registry.load_champion()
        X = patients[FEATURES]
        preds = model.predict(X)
        conf = model.predict_proba(X).max(axis=1)
        ids = [prod_log.new_id() for _ in range(len(patients))]
        rows = X.copy()
        rows.insert(0, "prediction_id", ids)
        rows.insert(1, "model_version", meta.get("version"))
        rows["prediction"], rows["confidence"] = preds, conf.round(4)
        prod_log.log_predictions(rows)
        prod_log.log_labels(ids[:n_labeled], patients[TARGET].iloc[:n_labeled])
    print(f"Simulated {len(patients)} patients (drift={args.drift}), {n_labeled} with confirmed diagnosis.")


if __name__ == "__main__":
    main()
