"""Model definitions, evaluation helpers and params loading."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import PARAMS_FILE


class EncodedClassifier:
    """Lets any estimator predict disease NAMES (XGBoost only accepts integer labels).

    Exposes the sklearn interface (fit / predict / predict_proba / classes_) so the
    API and MLflow can treat Decision Tree and XGBoost identically.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        self.estimator.fit(X, self.encoder.fit_transform(y))
        self.classes_ = self.encoder.classes_
        return self

    def predict(self, X):
        return self.encoder.inverse_transform(np.asarray(self.estimator.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def load_params(path: Path = PARAMS_FILE) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_candidates(params: dict, seed: int) -> dict:
    """The models that compete in every training / retraining round."""
    return {
        "decision_tree": EncodedClassifier(
            DecisionTreeClassifier(random_state=seed, **params["models"]["decision_tree"])
        ),
        "xgboost": EncodedClassifier(
            XGBClassifier(random_state=seed, eval_metric="mlogloss", n_jobs=1, **params["models"]["xgboost"])
        ),
    }


def evaluate(model, X: pd.DataFrame, y) -> dict:
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
    }


def file_md5(path: Path) -> str:
    """Fingerprint of the dataset file -- logged to MLflow so a model can be traced to its data version."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
