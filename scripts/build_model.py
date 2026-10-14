"""Train the deployable detectors once and persist them for the API.

Run:
    python scripts/build_model.py

Writes
------
models/detectors.pkl      all four detector families, per generator + pooled
models/metadata.json      thresholds, calibration table, corpus stats
web/model.json            scaler + logreg weights for the zero-install browser demo

The API loads models/detectors.pkl at startup. If it is missing the API trains
on first boot instead (~20 s), so the app never hard-fails on a fresh clone.
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genphish import corpus, detectors as D, calibration as C      # noqa: E402
from genphish.config import (FEATURES, GENERATORS, LABEL, SOURCE,  # noqa: E402
                             SEED, TARGET_RECALL)

MODELS = ROOT / "models"
WEB = ROOT / "web"
MODELS.mkdir(exist_ok=True)
WEB.mkdir(exist_ok=True)


def _f1_at(y, scores, thr) -> float:
    return float(f1_score(y, (scores >= thr).astype(int), zero_division=0))


def main() -> None:
    print("[1/5] loading corpus ...")
    df, is_real = corpus.load_corpus()
    data_mode = "REAL" if is_real else "SYNTHETIC"
    print(f"      mode={data_mode}  rows={len(df)}  "
          f"phish={int(df[LABEL].sum())}  ham={int((df[LABEL] == 0).sum())}")

    print("[2/5] training detector families ...")
    bundles, splits = {}, None
    for kind in D.DETECTOR_KINDS:
        dets, splits = D.train_all(df, GENERATORS, kind=kind, seed=SEED)
        bundles[kind] = dets
        print(f"      {kind:<8} -> {len(dets)} detectors")

    ham_tr, ham_te, ph_tr, ph_te = splits

    print("[3/5] computing operating thresholds + calibration table ...")
    meta = {
        "data_mode": data_mode,
        "seed": SEED,
        "features": list(FEATURES),
        "generators": list(GENERATORS),
        "detector_kinds": list(D.DETECTOR_KINDS),
        "target_recall": TARGET_RECALL,
        "corpus": {
            "rows": int(len(df)),
            "phishing": int(df[LABEL].sum()),
            "legitimate": int((df[LABEL] == 0).sum()),
            "by_source": {str(k): int(v)
                          for k, v in df[SOURCE].value_counts().items()},
        },
        "thresholds": {},
        "calibration": [],
    }

    pooled_phish_te = pd.concat([ph_te[g] for g in GENERATORS],
                                ignore_index=True)

    # Source threshold per (kind, generator), chosen at TARGET_RECALL on the
    # detector's OWN generator. This is the threshold that goes stale on shift.
    for kind, dets in bundles.items():
        meta["thresholds"][kind] = {}
        for g in list(GENERATORS) + ["pooled"]:
            det = dets[g]
            src_phish = pooled_phish_te if g == "pooled" else ph_te[g]
            eval_df = pd.concat([src_phish, ham_te], ignore_index=True)
            s = det.score(eval_df)
            y = eval_df[LABEL].values
            meta["thresholds"][kind][g] = {
                "stale": float(C.threshold_at_recall(s, y, TARGET_RECALL)),
                "oracle_f1": float(C.best_f1_threshold(s, y)),
                "source_prior": float(y.mean()),
            }

    # Cross-generator calibration table for the primary (xgboost) detector.
    xgb = bundles["xgboost"]
    for src in GENERATORS:
        det = xgb[src]
        src_eval = pd.concat([ph_te[src], ham_te], ignore_index=True)
        s_src = det.score(src_eval)
        y_src = src_eval[LABEL].values
        thr_stale = C.threshold_at_recall(s_src, y_src, TARGET_RECALL)
        prior = float(y_src.mean())

        for tgt in GENERATORS:
            if tgt == src:
                continue
            tgt_eval = pd.concat([ph_te[tgt], ham_te], ignore_index=True)
            s_tgt = det.score(tgt_eval)
            y_tgt = tgt_eval[LABEL].values

            thr_dm = C.dm_threshold(s_src, thr_stale, s_tgt)
            thr_em, pi_hat = C.em_threshold(s_tgt, prior)
            thr_st, _ = C.selftrain_threshold(s_tgt, thr_stale)
            thr_or = C.best_f1_threshold(s_tgt, y_tgt)

            meta["calibration"].append({
                "source": src, "target": tgt,
                "thr_stale": float(thr_stale), "thr_dm": float(thr_dm),
                "thr_em": float(thr_em), "thr_selftrain": float(thr_st),
                "thr_oracle": float(thr_or), "pi_hat": float(pi_hat),
                "f1_stale": _f1_at(y_tgt, s_tgt, thr_stale),
                "f1_dm": _f1_at(y_tgt, s_tgt, thr_dm),
                "f1_em": _f1_at(y_tgt, s_tgt, thr_em),
                "f1_selftrain": _f1_at(y_tgt, s_tgt, thr_st),
                "f1_oracle": _f1_at(y_tgt, s_tgt, thr_or),
            })

    print(f"      {len(meta['calibration'])} cross-generator pairs")

    print("[4/5] writing models/detectors.pkl + models/metadata.json ...")
    with open(MODELS / "detectors.pkl", "wb") as fh:
        pickle.dump({"bundles": bundles, "splits": splits}, fh)
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("[5/5] exporting web/model.json for the zero-install browser demo ...")
    lr = bundles["logreg"]["pooled"]
    pooled_thr = meta["thresholds"]["logreg"]["pooled"]
    web_model = {
        "kind": "logreg",
        "trained_on": "pooled",
        "data_mode": data_mode,
        "features": list(lr.features),
        "scaler_mean": [float(x) for x in lr.scaler.mean_],
        "scaler_scale": [float(x) for x in lr.scaler.scale_],
        "coef": [float(x) for x in lr.clf.coef_[0]],
        "intercept": float(lr.clf.intercept_[0]),
        "threshold": pooled_thr["stale"],
        "t_lo": 0.35,
        "t_hi": 0.65,
        "corpus": meta["corpus"],
    }
    (WEB / "model.json").write_text(json.dumps(web_model, indent=2))

    print("\nDone.")
    print(f"  data mode     : {data_mode}")
    print(f"  detectors     : {sum(len(b) for b in bundles.values())}")
    print(f"  browser model : logreg/pooled, thr={web_model['threshold']:.4f}")


if __name__ == "__main__":
    main()
