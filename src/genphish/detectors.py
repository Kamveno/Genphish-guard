"""S3 - baseline detectors.

A detector = (scaler + classifier) trained to separate phishing from legitimate
mail on the 17 stylometric features. We train one detector per generator (each
sees that generator's phishing + the shared legitimate mail) and one pooled
detector (all generators). XGBoost is the primary model; Logistic Regression is
the transparent comparison baseline.

A detector exposes score(X) -> P(phishing) in [0,1]. Thresholds are NOT baked in
here; threshold selection and transfer live in transfer.py and calibration.py,
because the whole research question is about where the threshold goes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from .config import FEATURES, LABEL, SOURCE, HUMAN, SEED


@dataclass
class Detector:
    scaler: StandardScaler
    clf: object
    kind: str          # "xgboost" | "logreg" | "rf" | "mlp"
    trained_on: str    # generator name or "pooled"
    features: list     # the feature columns this detector was fitted on (S6)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X[self.features].values)
        return self.clf.predict_proba(Xs)[:, 1]


def _make(kind: str, seed: int = SEED):
    """Detector families. xgboost is primary; the rest exist so S6 can show the
    phenomenon is a property of the SHIFT, not of one model family."""
    if kind == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
            eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=4,
        )
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=seed, n_jobs=4,
        )
    if kind == "mlp":
        # The "learned representation" arm: a non-linear net over the same 17
        # features. Not a text encoder (see neural.py for why).
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
            learning_rate_init=3e-3, max_iter=400, early_stopping=True,
            n_iter_no_change=15, random_state=seed,
        )
    raise ValueError(kind)


DETECTOR_KINDS = ["xgboost", "logreg", "rf", "mlp"]


def train(df_phish_source: pd.DataFrame, df_ham: pd.DataFrame, kind: str,
          trained_on: str, features: list | None = None,
          seed: int = SEED, shuffle_y: bool = False) -> Detector:
    """Train one detector on (this generator's phishing) + (legitimate mail).

    shuffle_y permutes the TRAINING labels only, across the assembled training
    set, leaving row selection and all test labels untouched. That is the S6
    negative control: it severs the feature-label association without changing
    anything else, so a detector trained this way must be unable to rank
    (test AUC ~ 0.5). Permuting labels earlier — before the split — would not
    work, because rows are selected by provenance, so the two groups would
    still be genuinely different mail and the control would leak.
    """
    features = list(features) if features else list(FEATURES)
    train_df = pd.concat([df_phish_source, df_ham], ignore_index=True)
    X = train_df[features].values
    y = train_df[LABEL].values
    if shuffle_y:
        y = np.random.default_rng(seed).permutation(y)
    scaler = StandardScaler().fit(X)
    clf = _make(kind, seed)
    clf.fit(scaler.transform(X), y)
    return Detector(scaler, clf, kind, trained_on, features)


def split_corpus(df: pd.DataFrame, generators: list[str], seed: int = SEED):
    """Return (ham_train, ham_test, phish_by_gen_train, phish_by_gen_test).

    Legitimate mail is split once (shared). Each generator's phishing is split
    70/30; the test halves feed the transfer matrix and the unlabelled-target
    calibration, so no sample is both tuned on and scored on.

    `seed` reshuffles the split, which is what stability.py varies to show the
    headline numbers are not one lucky partition.
    """
    ham = df[df[SOURCE] == HUMAN].sample(frac=1, random_state=seed)
    cut = int(0.7 * len(ham))
    ham_tr, ham_te = ham.iloc[:cut], ham.iloc[cut:]

    ph_tr, ph_te = {}, {}
    for g in generators:
        gp = df[(df[SOURCE] == g) & (df[LABEL] == 1)].sample(frac=1, random_state=seed)
        c = int(0.7 * len(gp))
        ph_tr[g], ph_te[g] = gp.iloc[:c], gp.iloc[c:]
    return ham_tr, ham_te, ph_tr, ph_te


def train_all(df: pd.DataFrame, generators: list[str], kind: str = "xgboost",
              features: list | None = None, seed: int = SEED,
              shuffle_y: bool = False):
    """Train per-generator detectors and a pooled detector."""
    ham_tr, ham_te, ph_tr, ph_te = split_corpus(df, generators, seed)
    dets = {g: train(ph_tr[g], ham_tr, kind, g, features, seed, shuffle_y)
            for g in generators}
    pooled_phish = pd.concat([ph_tr[g] for g in generators], ignore_index=True)
    dets["pooled"] = train(pooled_phish, ham_tr, kind, "pooled", features, seed,
                           shuffle_y)
    return dets, (ham_tr, ham_te, ph_tr, ph_te)
