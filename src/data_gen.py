"""Synthetic patient data generator with a controllable DRIFT knob.

drift=0.0  -> same distribution the model was trained on (the original project's data)
drift=1.0  -> a changed world, used to demo data/concept drift:
              * outbreak season: Dengue / COVID-19 / Pneumonia become more common (prior shift)
              * a new COVID variant: loss of taste/smell becomes rare, sore throat common (concept drift)
              * more fever & fatigue reported overall (covariate shift)
"""
import argparse
import copy

import numpy as np
import pandas as pd

from .config import FEATURES, RAW_DATA, TARGET

DISEASE_PROBABILITIES = {
    "COVID-19": {
        "fever": 0.82, "cough": 0.80, "headache": 0.60, "fatigue": 0.78, "sore_throat": 0.55,
        "runny_nose": 0.35, "body_ache": 0.65, "shortness_of_breath": 0.70, "loss_of_taste_smell": 0.75,
        "itchy_eyes": 0.05, "sneezing": 0.10, "nausea": 0.30, "joint_pain": 0.35, "skin_rash": 0.15
    },
    "Flu": {
        "fever": 0.90, "cough": 0.75, "headache": 0.70, "fatigue": 0.88, "sore_throat": 0.50,
        "runny_nose": 0.45, "body_ache": 0.85, "shortness_of_breath": 0.15, "loss_of_taste_smell": 0.02,
        "itchy_eyes": 0.08, "sneezing": 0.15, "nausea": 0.25, "joint_pain": 0.45, "skin_rash": 0.02
    },
    "Cold": {
        "fever": 0.10, "cough": 0.65, "headache": 0.15, "fatigue": 0.40, "sore_throat": 0.80,
        "runny_nose": 0.90, "body_ache": 0.20, "shortness_of_breath": 0.05, "loss_of_taste_smell": 0.05,
        "itchy_eyes": 0.15, "sneezing": 0.85, "nausea": 0.02, "joint_pain": 0.08, "skin_rash": 0.01
    },
    "Allergy": {
        "fever": 0.01, "cough": 0.25, "headache": 0.15, "fatigue": 0.25, "sore_throat": 0.30,
        "runny_nose": 0.85, "body_ache": 0.02, "shortness_of_breath": 0.10, "loss_of_taste_smell": 0.02,
        "itchy_eyes": 0.92, "sneezing": 0.90, "nausea": 0.01, "joint_pain": 0.02, "skin_rash": 0.18
    },
    "Migraine": {
        "fever": 0.01, "cough": 0.01, "headache": 0.98, "fatigue": 0.55, "sore_throat": 0.01,
        "runny_nose": 0.02, "body_ache": 0.15, "shortness_of_breath": 0.01, "loss_of_taste_smell": 0.01,
        "itchy_eyes": 0.05, "sneezing": 0.02, "nausea": 0.78, "joint_pain": 0.05, "skin_rash": 0.01
    },
    "Pneumonia": {
        "fever": 0.85, "cough": 0.92, "headache": 0.35, "fatigue": 0.82, "sore_throat": 0.25,
        "runny_nose": 0.12, "body_ache": 0.55, "shortness_of_breath": 0.90, "loss_of_taste_smell": 0.02,
        "itchy_eyes": 0.02, "sneezing": 0.02, "nausea": 0.20, "joint_pain": 0.25, "skin_rash": 0.01
    },
    "Dengue": {
        "fever": 0.95, "cough": 0.08, "headache": 0.82, "fatigue": 0.90, "sore_throat": 0.10,
        "runny_nose": 0.08, "body_ache": 0.92, "shortness_of_breath": 0.05, "loss_of_taste_smell": 0.02,
        "itchy_eyes": 0.18, "sneezing": 0.05, "nausea": 0.58, "joint_pain": 0.94, "skin_rash": 0.72
    },
    "Healthy": {
        "fever": 0.02, "cough": 0.03, "headache": 0.05, "fatigue": 0.05, "sore_throat": 0.02,
        "runny_nose": 0.03, "body_ache": 0.02, "shortness_of_breath": 0.01, "loss_of_taste_smell": 0.01,
        "itchy_eyes": 0.02, "sneezing": 0.03, "nausea": 0.01, "joint_pain": 0.02, "skin_rash": 0.01
    }
}


def _drifted_probs(drift: float) -> dict:
    probs = copy.deepcopy(DISEASE_PROBABILITIES)
    if drift <= 0:
        return probs
    d = min(drift, 1.0)
    # new COVID variant: taste/smell loss fades, sore throat & runny nose rise
    probs["COVID-19"]["loss_of_taste_smell"] *= (1 - 0.9 * d)
    probs["COVID-19"]["sore_throat"] = min(0.95, probs["COVID-19"]["sore_throat"] + 0.35 * d)
    probs["COVID-19"]["runny_nose"] = min(0.95, probs["COVID-19"]["runny_nose"] + 0.40 * d)
    probs["COVID-19"]["shortness_of_breath"] *= (1 - 0.5 * d)
    # population-wide shift: more fever/fatigue reports
    for disease in probs:
        for f in ("fever", "fatigue"):
            probs[disease][f] = min(0.98, probs[disease][f] + 0.15 * d)
    return probs


def _class_weights(drift: float) -> np.ndarray:
    w = {name: 1.0 for name in DISEASE_PROBABILITIES}
    for name in ("Dengue", "COVID-19", "Pneumonia"):
        w[name] = 1.0 + 2.0 * min(drift, 1.0)
    arr = np.array([w[n] for n in DISEASE_PROBABILITIES], dtype=float)
    return arr / arr.sum()


def generate(n: int = 400, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    """Return n synthetic patients. drift=0 reproduces the original balanced dataset style."""
    rng = np.random.default_rng(seed)
    probs = _drifted_probs(drift)
    diseases = list(DISEASE_PROBABILITIES)
    if drift <= 0:
        labels = np.repeat(diseases, int(np.ceil(n / len(diseases))))[:n]
        rng.shuffle(labels)
    else:
        labels = rng.choice(diseases, size=n, p=_class_weights(drift))
    rows = []
    for d in labels:
        rows.append([int(rng.random() < probs[d][f]) for f in FEATURES] + [d])
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic patient data")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift", type=float, default=0.0, help="0 = no drift, 1 = strong drift")
    ap.add_argument("--out", default=str(RAW_DATA))
    args = ap.parse_args()
    df = generate(args.n, args.seed, args.drift)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows (drift={args.drift}) -> {args.out}")


if __name__ == "__main__":
    main()
