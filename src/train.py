"""Train candidate models, track everything in MLflow, register the best, promote it if it earns it.

    python -m src.train
"""
import argparse

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import registry
from .config import FEATURES, RAW_DATA, TARGET, TRAIN_EXPERIMENT
from .modeling import build_candidates, evaluate, file_md5, load_params


def cv_f1(name: str, X: pd.DataFrame, y: pd.Series, params: dict, seed: int) -> float:
    """5-fold cross-validated macro-F1 on the TRAIN set only (used to pick the best candidate)."""
    scores = []
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
        model = build_candidates(params, seed)[name]
        model.fit(X.iloc[tr], y.iloc[tr])
        scores.append(evaluate(model, X.iloc[va], y.iloc[va])["f1_macro"])
    return sum(scores) / len(scores)


def fit_candidates(train_df, eval_df, params, seed, tags: dict, run_prefix: str):
    """Train every candidate, log one MLflow run each, return (results, best_name).

    Selection uses cross-validation on the training data; `eval_df` is only used for reporting,
    so the held-out numbers stay honest.
    """
    X_tr, y_tr = train_df[FEATURES], train_df[TARGET]
    X_ev, y_ev = eval_df[FEATURES], eval_df[TARGET]
    results = {}
    for name, model in build_candidates(params, seed).items():
        with mlflow.start_run(run_name=f"{run_prefix}-{name}") as run:
            mlflow.set_tags({"model_type": name, **tags})
            mlflow.log_params({f"{name}.{k}": v for k, v in params["models"][name].items()})
            mlflow.log_params({"n_train_rows": len(train_df), "n_eval_rows": len(eval_df), "seed": seed})
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_ev, y_ev)
            metrics["cv_f1_macro"] = cv_f1(name, X_tr, y_tr, params, seed)
            mlflow.log_metrics(metrics)
            results[name] = {"model": model, "metrics": metrics, "run_id": run.info.run_id}
            print(f"  {name:14s} test_f1={metrics['f1_macro']:.3f}  acc={metrics['accuracy']:.3f}  cv_f1={metrics['cv_f1_macro']:.3f}")
    best = max(results, key=lambda n: results[n]["metrics"]["cv_f1_macro"])
    return results, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW_DATA))
    args = ap.parse_args()

    params = load_params()
    seed = params["data"]["seed"]
    df = pd.read_csv(args.data)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing columns: {sorted(missing)}")

    train_df, test_df = train_test_split(df, test_size=params["data"]["test_size"], random_state=seed, stratify=df[TARGET])
    data_md5 = file_md5(args.data)

    registry.init_mlflow(TRAIN_EXPERIMENT)
    print(f"Training on {len(train_df)} rows, evaluating on {len(test_df)} rows (data md5 {data_md5[:8]})")
    results, best = fit_candidates(train_df, test_df, params, seed, {"stage": "initial-training", "data_md5": data_md5}, "train")
    b = results[best]
    print(f"Best candidate (by CV F1): {best}")

    version = registry.register_model(
        b["run_id"], b["model"], train_df[FEATURES],
        {"model_type": best, "f1_macro": round(b["metrics"]["f1_macro"], 4), "accuracy": round(b["metrics"]["accuracy"], 4), "data_md5": data_md5},
    )

    champ = registry.champion_meta()
    min_gain = params["retraining"]["min_improvement"]
    if champ is None:
        decision = "no champion yet -> promoted"
        promote = True
    else:
        champ_model, _ = registry.load_champion()
        champ_f1 = evaluate(champ_model, test_df[FEATURES], test_df[TARGET])["f1_macro"]
        promote = b["metrics"]["f1_macro"] >= champ_f1 + min_gain
        decision = f"challenger F1 {b['metrics']['f1_macro']:.3f} vs champion {champ_f1:.3f} (needs +{min_gain}) -> {'promoted' if promote else 'kept champion'}"
    if promote:
        registry.promote(
            version, b["model"],
            {"model_type": best, "f1_macro": round(b["metrics"]["f1_macro"], 4), "accuracy": round(b["metrics"]["accuracy"], 4), "data_md5": data_md5},
            train_df,
        )
    print(f"Registered version {version}: {decision}")


if __name__ == "__main__":
    main()
