"""Inference service — the single object the API routes talk to.

Loads models/detectors.pkl if present, otherwise trains on first boot so a
fresh clone still works. Everything the routes need (scoring, tiering,
calibration lookup, transfer matrix, drift stream, SOAR log) hangs off one
`Service` instance created at import time.
"""
from __future__ import annotations

import json
import pickle
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genphish import corpus, detectors as D, features as F        # noqa: E402
from genphish import calibration as C, zerotrust as ZT            # noqa: E402
from genphish import soar as SOAR, drift_monitor as DM            # noqa: E402
from genphish.config import (FEATURES, GENERATORS, LABEL, SOURCE,  # noqa: E402
                             SEED, TARGET_RECALL, DELTA)

MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# Feature groups, used for the explainability panel in the UI.
FEATURE_GROUPS = {
    "length": ["char_count", "word_count", "sentence_count", "avg_sentence_len"],
    "lexical": ["avg_word_len", "type_token_ratio", "stopword_ratio",
                "function_word_ratio", "personal_pronoun_ratio",
                "flesch_reading_ease"],
    "orthographic": ["uppercase_ratio", "digit_ratio", "punct_ratio"],
    "persuasion": ["exclamation_count", "question_count", "url_count",
                   "urgency_term_count"],
}
_GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}


@dataclass
class DriftSession:
    """One browser's online drift monitor."""
    monitor: object
    history: list = field(default_factory=list)
    created: float = field(default_factory=time.time)


class Service:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ready = False
        self.boot_error: str | None = None
        self.data_mode = "UNKNOWN"
        self.bundles: dict = {}
        self.splits = None
        self.meta: dict = {}
        self.incidents: list[SOAR.Incident] = []
        self.drift_sessions: dict[str, DriftSession] = {}
        self._load()

    # ---------------------------------------------------------------- boot
    def _load(self) -> None:
        try:
            pkl = MODELS / "detectors.pkl"
            meta_path = MODELS / "metadata.json"
            if pkl.exists() and meta_path.exists():
                with open(pkl, "rb") as fh:
                    blob = pickle.load(fh)
                self.bundles = blob["bundles"]
                self.splits = blob["splits"]
                self.meta = json.loads(meta_path.read_text())
                self.data_mode = self.meta.get("data_mode", "SYNTHETIC")
                # A metadata file written by an older build may predate the
                # calibration table; recompute rather than serve a blank tab.
                if not self.meta.get("calibration"):
                    self.meta["calibration"] = self._build_calibration_table()
            else:
                self._train_from_scratch()
            self.ready = True
        except Exception as exc:                       # pragma: no cover
            self.boot_error = f"{type(exc).__name__}: {exc}"
            self.ready = False

    def _train_from_scratch(self) -> None:
        df, is_real = corpus.load_corpus()
        self.data_mode = "REAL" if is_real else "SYNTHETIC"
        splits = None
        for kind in D.DETECTOR_KINDS:
            dets, splits = D.train_all(df, GENERATORS, kind=kind, seed=SEED)
            self.bundles[kind] = dets
        self.splits = splits
        ham_tr, ham_te, ph_tr, ph_te = splits
        self.meta = {
            "data_mode": self.data_mode,
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
        pooled_te = pd.concat([ph_te[g] for g in GENERATORS], ignore_index=True)
        for kind, dets in self.bundles.items():
            self.meta["thresholds"][kind] = {}
            for g in list(GENERATORS) + ["pooled"]:
                src = pooled_te if g == "pooled" else ph_te[g]
                ev = pd.concat([src, ham_te], ignore_index=True)
                s, y = dets[g].score(ev), ev[LABEL].values
                self.meta["thresholds"][kind][g] = {
                    "stale": float(C.threshold_at_recall(s, y, TARGET_RECALL)),
                    "oracle_f1": float(C.best_f1_threshold(s, y)),
                    "source_prior": float(y.mean()),
                }
        self.meta["calibration"] = self._build_calibration_table()

    def _build_calibration_table(self, kind: str = "xgboost") -> list[dict]:
        """Cross-generator threshold transport for every ordered source→target
        pair. Identical to what scripts/build_model.py persists, recomputed
        here so a fresh clone with no models/ directory still serves the
        Calibration tab and the label-free threshold options.
        """
        from sklearn.metrics import f1_score
        if not self.splits or kind not in self.bundles:
            return []
        ham_tr, ham_te, ph_tr, ph_te = self.splits
        bundle = self.bundles[kind]
        rows: list[dict] = []

        def f1_at(y, s, t):
            return float(f1_score(y, (s >= t).astype(int), zero_division=0))

        for src in GENERATORS:
            det = bundle[src]
            src_eval = pd.concat([ph_te[src], ham_te], ignore_index=True)
            s_src, y_src = det.score(src_eval), src_eval[LABEL].values
            thr_stale = C.threshold_at_recall(s_src, y_src, TARGET_RECALL)
            prior = float(y_src.mean())

            for tgt in GENERATORS:
                if tgt == src:
                    continue
                tgt_eval = pd.concat([ph_te[tgt], ham_te], ignore_index=True)
                s_tgt, y_tgt = det.score(tgt_eval), tgt_eval[LABEL].values

                thr_dm = C.dm_threshold(s_src, thr_stale, s_tgt)
                thr_em, pi_hat = C.em_threshold(s_tgt, prior)
                thr_st, _ = C.selftrain_threshold(s_tgt, thr_stale)
                thr_or = C.best_f1_threshold(s_tgt, y_tgt)

                rows.append({
                    "source": src, "target": tgt,
                    "thr_stale": float(thr_stale), "thr_dm": float(thr_dm),
                    "thr_em": float(thr_em), "thr_selftrain": float(thr_st),
                    "thr_oracle": float(thr_or), "pi_hat": float(pi_hat),
                    "f1_stale": f1_at(y_tgt, s_tgt, thr_stale),
                    "f1_dm": f1_at(y_tgt, s_tgt, thr_dm),
                    "f1_em": f1_at(y_tgt, s_tgt, thr_em),
                    "f1_selftrain": f1_at(y_tgt, s_tgt, thr_st),
                    "f1_oracle": f1_at(y_tgt, s_tgt, thr_or),
                })
        return rows

    # ------------------------------------------------------------ detectors
    def detector_names(self) -> list[str]:
        return list(GENERATORS) + ["pooled"]

    def get_detector(self, kind: str, trained_on: str):
        if kind not in self.bundles:
            raise KeyError(f"unknown detector family '{kind}'")
        bundle = self.bundles[kind]
        if trained_on not in bundle:
            raise KeyError(f"unknown detector '{trained_on}'")
        return bundle[trained_on]

    def threshold_for(self, kind: str, trained_on: str,
                      calibration: str = "stale") -> float:
        """Operating threshold for a (family, detector, calibration) triple.

        'stale' is the source threshold as shipped — the one that goes wrong
        under generator shift. 'oracle' is the labelled upper bound. The three
        label-free methods come from the calibration table: averaged over the
        pairs whose source is this detector, or over every pair when the
        detector is the pooled one (which has no single source generator).
        """
        node = self.meta.get("thresholds", {}).get(kind, {}).get(trained_on)
        if node is None:
            return 0.5
        if calibration in ("stale", "", None):
            return float(node["stale"])
        if calibration == "oracle":
            return float(node["oracle_f1"])

        key = f"thr_{calibration}"
        all_rows = self.meta.get("calibration", [])
        rows = [r for r in all_rows if r["source"] == trained_on]
        if not rows:
            # pooled (or a detector absent from the table): average every pair
            rows = all_rows
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else float(node["stale"])

    # --------------------------------------------------------------- detect
    def extract(self, text: str) -> dict:
        return F.extract(text)

    def score_texts(self, texts: list[str], kind: str, trained_on: str
                    ) -> tuple[np.ndarray, pd.DataFrame]:
        frame = F.extract_frame(texts)
        det = self.get_detector(kind, trained_on)
        return det.score(frame), frame

    def contributions(self, feats: dict, kind: str, trained_on: str
                      ) -> list[dict]:
        """Per-feature contribution to the decision.

        Linear models give signed z-score * coefficient directly. Tree
        ensembles expose a global gain importance, which we sign by the
        feature's own z-score so the UI can show direction as well as weight.
        """
        det = self.get_detector(kind, trained_on)
        cols = list(det.features)
        x = np.array([feats[c] for c in cols], dtype=float)
        z = (x - det.scaler.mean_) / np.where(det.scaler.scale_ == 0, 1,
                                              det.scaler.scale_)
        clf = det.clf
        if hasattr(clf, "coef_"):
            contrib = z * clf.coef_[0]
        elif hasattr(clf, "feature_importances_"):
            contrib = z * np.asarray(clf.feature_importances_, dtype=float)
        else:
            contrib = z * 0.0
        out = [
            {
                "feature": c,
                "group": _GROUP_OF.get(c, "other"),
                "value": float(x[i]),
                "z": float(z[i]),
                "contribution": float(contrib[i]),
            }
            for i, c in enumerate(cols)
        ]
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out

    @staticmethod
    def tier_bounds(threshold: float, band: float = 0.6,
                    t_lo: float | None = None, t_hi: float | None = None
                    ) -> tuple[float, float]:
        """Zero-trust tier boundaries anchored to the operating threshold.

        The calibrated threshold IS the BLOCK gate: at or above it the
        detector is confident enough to refuse delivery. DEFUSE is the band
        immediately below it, where the detector is uncertain — the tier that
        exists precisely because generator shift makes it uncertain rather
        than wrong. Anchoring the tiers to the threshold rather than fixing
        them at 0.35/0.65 is what makes recalibration visibly move the policy:
        change the calibration method and the whole tier structure moves with
        the threshold it produced.

        `band` is the fraction of the threshold spanned by DEFUSE. Explicit
        t_lo / t_hi override the derivation entirely.
        """
        hi = float(threshold) if t_hi is None else float(t_hi)
        lo = hi * (1.0 - float(band)) if t_lo is None else float(t_lo)
        lo = max(0.0, min(lo, hi))
        return lo, hi

    def detect(self, text: str, kind: str = "xgboost",
               trained_on: str = "pooled", calibration: str = "stale",
               t_lo: float | None = None, t_hi: float | None = None,
               raise_incident: bool = True, band: float = 0.6) -> dict:
        scores, frame = self.score_texts([text], kind, trained_on)
        score = float(scores[0])
        thr = self.threshold_for(kind, trained_on, calibration)
        lo, hi = self.tier_bounds(thr, band, t_lo, t_hi)
        tier = ZT.classify(score, thr, lo, hi)
        feats = frame.iloc[0].to_dict()
        contribs = self.contributions(feats, kind, trained_on)

        incident = None
        if raise_incident and tier in ("BLOCK", "DEFUSE"):
            inc = SOAR.Incident(
                id=str(uuid.uuid4())[:8].upper(),
                timestamp=time.time(),
                tier=tier,
                confidence=score,
                threshold=thr,
                cal_method=calibration,
                source_det=f"{kind}/{trained_on}",
                severity={"BLOCK": "P1-Critical",
                          "DEFUSE": "P2-High"}[tier],
                raw_features={k: round(float(v), 4) for k, v in feats.items()},
            )
            with self._lock:
                self.incidents.append(inc)
                self.incidents = self.incidents[-500:]
            incident = inc.to_dict()
            incident["playbook"] = SOAR.playbook_response(inc)

        return {
            "score": score,
            "threshold": thr,
            "tier": tier,
            "t_lo": lo,
            "t_hi": hi,
            "detector": {"family": kind, "trained_on": trained_on},
            "calibration": calibration,
            "features": {k: float(v) for k, v in feats.items()},
            "contributions": contribs,
            "group_contributions": self._group_totals(contribs),
            "incident": incident,
            "data_mode": self.data_mode,
        }

    @staticmethod
    def _group_totals(contribs: list[dict]) -> list[dict]:
        tot: dict[str, float] = {}
        for c in contribs:
            tot[c["group"]] = tot.get(c["group"], 0.0) + c["contribution"]
        return [{"group": g, "contribution": v}
                for g, v in sorted(tot.items(), key=lambda kv: -abs(kv[1]))]

    def detect_batch(self, texts: list[str], kind: str = "xgboost",
                     trained_on: str = "pooled", calibration: str = "stale",
                     raise_incidents: bool = False) -> dict:
        scores, frame = self.score_texts(texts, kind, trained_on)
        thr = self.threshold_for(kind, trained_on, calibration)
        lo, hi = self.tier_bounds(thr)
        tiers = ZT.classify_batch(scores, lo, hi)
        rows = [
            {"index": i, "score": float(s), "tier": t,
             "preview": (texts[i][:120] + "…") if len(texts[i]) > 120 else texts[i]}
            for i, (s, t) in enumerate(zip(scores, tiers))
        ]
        created = []
        if raise_incidents:
            feats = [frame.iloc[i].to_dict() for i in range(len(frame))]
            incs = SOAR.incidents_from_scores(
                scores, tiers, calibration, f"{kind}/{trained_on}",
                threshold=thr,
                raw_features=[{k: round(float(v), 4) for k, v in f.items()}
                              for f in feats],
            )
            with self._lock:
                self.incidents.extend(incs)
                self.incidents = self.incidents[-500:]
            created = [i.to_dict() for i in incs]
        counts = {t: int(sum(1 for x in tiers if x == t))
                  for t in ("BLOCK", "DEFUSE", "DELIVER")}
        return {
            "n": len(texts), "threshold": thr, "counts": counts,
            "t_lo": lo, "t_hi": hi,
            "results": rows, "incidents_created": created,
            "detector": {"family": kind, "trained_on": trained_on},
            "calibration": calibration, "data_mode": self.data_mode,
        }

    # ------------------------------------------------------------- transfer
    def transfer(self) -> dict:
        """Transfer matrices. Prefers the pre-computed CSVs from run_all;
        falls back to computing them live from the loaded splits."""
        auc_p = RESULTS / "transfer_auc.csv"
        f1_p = RESULTS / "transfer_f1_default.csv"
        if auc_p.exists() and f1_p.exists():
            auc = pd.read_csv(auc_p, index_col=0)
            f1 = pd.read_csv(f1_p, index_col=0)
            return {
                "sources": [str(i) for i in auc.index],
                "targets": [str(c) for c in auc.columns],
                "auc": auc.values.tolist(),
                "f1": f1.values.tolist(),
                "source": "results/transfer_*.csv",
                "data_mode": self.data_mode,
            }
        return self._transfer_live()

    def _transfer_live(self) -> dict:
        from sklearn.metrics import roc_auc_score, f1_score
        ham_tr, ham_te, ph_tr, ph_te = self.splits
        xgb = self.bundles["xgboost"]
        auc, f1 = [], []
        for src in GENERATORS:
            det = xgb[src]
            thr = self.threshold_for("xgboost", src, "stale")
            arow, frow = [], []
            for tgt in GENERATORS:
                ev = pd.concat([ph_te[tgt], ham_te], ignore_index=True)
                s, y = det.score(ev), ev[LABEL].values
                arow.append(float(roc_auc_score(y, s)))
                frow.append(float(f1_score(y, (s >= thr).astype(int),
                                           zero_division=0)))
            auc.append(arow)
            f1.append(frow)
        return {"sources": list(GENERATORS), "targets": list(GENERATORS),
                "auc": auc, "f1": f1, "source": "computed live",
                "data_mode": self.data_mode}

    # ---------------------------------------------------------- calibration
    def calibration_table(self) -> dict:
        rows = self.meta.get("calibration", [])
        if not rows:
            return {"rows": [], "summary": {}, "data_mode": self.data_mode}
        def mean(k):
            return float(np.mean([r[k] for r in rows]))
        stale, dm = mean("f1_stale"), mean("f1_dm")
        oracle = mean("f1_oracle")
        denom = oracle - stale
        return {
            "rows": rows,
            "summary": {
                "f1_stale": stale,
                "f1_dm": dm,
                "f1_em": mean("f1_em"),
                "f1_selftrain": mean("f1_selftrain"),
                "f1_oracle": oracle,
                "gap_recovered_dm": float((dm - stale) / denom) if denom > 1e-9 else 0.0,
                "pi_hat_mean": mean("pi_hat"),
                "true_prior": self.true_target_prior(),
                "n_pairs": len(rows),
            },
            "data_mode": self.data_mode,
        }

    def true_target_prior(self) -> float:
        """Actual phishing prevalence in the held-out target evaluation sets.

        This is what EM's π̂ should be compared against — not the whole-corpus
        prevalence, which is a different (and much higher) number because the
        corpus is balanced by construction while a target evaluation set is
        one generator's phishing against the shared legitimate mail.
        """
        if not self.splits:
            return float("nan")
        ham_tr, ham_te, ph_tr, ph_te = self.splits
        priors = [len(ph_te[g]) / (len(ph_te[g]) + len(ham_te))
                  for g in GENERATORS if g in ph_te]
        return float(np.mean(priors)) if priors else float("nan")

    # --------------------------------------------------------------- drift
    def drift_precomputed(self) -> dict:
        p = RESULTS / "drift_monitor.csv"
        if not p.exists():
            return {"available": False}
        df = pd.read_csv(p)
        return {
            "available": True,
            "columns": list(df.columns),
            "rows": df.to_dict("records"),
            "ceiling": 1.0 / DELTA,
            "delta": DELTA,
        }

    def drift_new_session(self, band_lo: float = 0.35, band_hi: float = 0.65,
                          warmup: int = 30) -> str:
        sid = str(uuid.uuid4())[:12]
        with self._lock:
            # evict old sessions so a long-running server does not grow forever
            if len(self.drift_sessions) > 64:
                oldest = sorted(self.drift_sessions.items(),
                                key=lambda kv: kv[1].created)[:32]
                for k, _ in oldest:
                    self.drift_sessions.pop(k, None)
            self.drift_sessions[sid] = DriftSession(
                monitor=DM.ConformalMartingale(
                    delta=DELTA, band_lo=band_lo, band_hi=band_hi,
                    warmup=warmup))
        return sid

    def drift_step(self, sid: str, score: float) -> dict:
        sess = self.drift_sessions.get(sid)
        if sess is None:
            raise KeyError("unknown drift session")
        wealth, alarm = sess.monitor.update(float(score))
        sess.history.append({"score": float(score), "wealth": float(wealth),
                             "alarm": bool(alarm)})
        return {"n": len(sess.history), "score": float(score),
                "wealth": float(wealth), "alarm": bool(alarm),
                "ceiling": 1.0 / DELTA}

    def drift_simulate(self, n: int = 400, shift_at: int = 200,
                       kind: str = "xgboost", source: str | None = None,
                       target: str | None = None) -> dict:
        """Score real held-out mail from `source`, then switch to `target`,
        feeding both through the online martingale. This is the live demo of
        the trigger firing on a generator change.

        The band and null rate come from the source detector's own calibration
        scores, exactly as drift_monitor.monitor_stream derives them. A fixed
        mid-range band would be blind here: tree-ensemble scores saturate, so
        almost nothing ever lands in [0.35, 0.65].
        """
        source = source or GENERATORS[0]
        target = target or GENERATORS[1]
        if source == target:
            target = next(g for g in GENERATORS if g != source)
        ham_tr, ham_te, ph_tr, ph_te = self.splits
        det = self.get_detector(kind, source)
        rng = np.random.default_rng(SEED)

        # Calibration reference: the source detector on its OWN mail.
        cal_ham = det.score(ham_tr)
        cal_phish = det.score(ph_tr[source])
        mon = DM.ConformalMartingale.from_calibration(
            cal_ham, cal_phish, delta=DELTA)

        def draw(gen, k):
            pool = pd.concat([ph_te[gen], ham_te], ignore_index=True)
            idx = rng.choice(len(pool), size=k, replace=True)
            sub = pool.iloc[idx]
            return det.score(sub), sub[LABEL].values

        pre_n = max(1, min(int(shift_at), int(n) - 1))
        post_n = int(n) - pre_n
        s_pre, y_pre = draw(source, pre_n)
        s_post, y_post = draw(target, post_n)
        scores = np.concatenate([s_pre, s_post])
        labels = np.concatenate([y_pre, y_post])

        wealth, alarm_at = [], None
        for i, s in enumerate(scores):
            w, a = mon.update(float(s))
            wealth.append(float(w))
            if a and alarm_at is None:
                alarm_at = i

        band_lo, band_hi = mon.band
        in_band = ((scores >= band_lo) & (scores < band_hi))
        return {
            "n": int(n), "shift_at": int(pre_n),
            "source": source, "target": target, "detector_family": kind,
            "scores": [float(s) for s in scores],
            "labels": [int(y) for y in labels],
            "wealth": wealth,
            "alarm_index": alarm_at,
            "ttd": (alarm_at - pre_n) if alarm_at is not None and alarm_at >= pre_n else None,
            "ceiling": 1.0 / DELTA, "delta": DELTA,
            "band": [float(band_lo), float(band_hi)],
            "r0": float(mon.r0) if mon.r0 is not None else None,
            "band_rate_pre": float(in_band[:pre_n].mean()) if pre_n else 0.0,
            "band_rate_post": float(in_band[pre_n:].mean()) if post_n else 0.0,
            "data_mode": self.data_mode,
        }

    # ---------------------------------------------------------------- soar
    def soar_log(self, limit: int = 100) -> dict:
        with self._lock:
            items = list(self.incidents)[-limit:][::-1]
        return {
            "count": len(self.incidents),
            "incidents": [
                {**i.to_dict(), "playbook": SOAR.playbook_response(i)}
                for i in items
            ],
        }

    def soar_clear(self) -> dict:
        with self._lock:
            n = len(self.incidents)
            self.incidents = []
        return {"cleared": n}

    def soar_dispatch(self, incident_id: str, endpoint: str | None) -> dict:
        with self._lock:
            match = next((i for i in self.incidents if i.id == incident_id), None)
        if match is None:
            raise KeyError(f"no incident {incident_id}")
        if not endpoint:
            conn = SOAR.SimulationConnector()
            res = conn.send(match)
            return {"mode": "simulation", "result": res,
                    "playbook": SOAR.playbook_response(match)}
        conn = SOAR.GenericWebhookConnector(endpoint)
        return {"mode": "webhook", "endpoint": endpoint,
                "result": conn.send(match),
                "playbook": SOAR.playbook_response(match)}

    # --------------------------------------------------------------- stats
    def stats(self) -> dict:
        res: dict = {
            "data_mode": self.data_mode,
            "corpus": self.meta.get("corpus", {}),
            "generators": list(GENERATORS),
            "detector_kinds": list(self.bundles.keys()),
            "features": list(FEATURES),
            "feature_groups": FEATURE_GROUPS,
            "target_recall": TARGET_RECALL,
            "delta": DELTA,
            "incidents": len(self.incidents),
        }
        summary = RESULTS / "summary.json"
        if summary.exists():
            try:
                res["run_all_summary"] = json.loads(summary.read_text())
            except Exception:
                pass
        ext = RESULTS / "extended_summary.json"
        if ext.exists():
            try:
                res["extended_summary"] = json.loads(ext.read_text())
            except Exception:
                pass
        cal = self.calibration_table()
        res["calibration_summary"] = cal["summary"]
        return res

    # -------------------------------------------------------------- samples
    def samples(self) -> list[dict]:
        """Hand-written demo emails covering every tier, so a reviewer can
        click rather than type."""
        return _SAMPLES


_SAMPLES = [
    {
        "name": "LLM phishing — credential harvest (GPT-style)",
        "expect": "BLOCK",
        "text": (
            "Subject: URGENT: Your account access will be suspended\n\n"
            "Dear Valued Customer,\n\n"
            "We have detected unusual sign-in activity on your account from an "
            "unrecognized device. For your security, your account access has been "
            "temporarily limited and will be SUSPENDED within 24 hours unless you "
            "verify your identity immediately.\n\n"
            "Please confirm your account details now by clicking the secure link below:\n"
            "https://account-verify-secure.com/login?ref=8821\n\n"
            "Failure to act will result in permanent closure of your account. "
            "This is your final warning!\n\n"
            "Thank you for your immediate attention,\n"
            "Account Security Team"
        ),
    },
    {
        "name": "LLM phishing — invoice lure, low urgency",
        "expect": "BLOCK",
        "text": (
            "Subject: Invoice #40219 is ready for your review\n\n"
            "Hello,\n\n"
            "Please find attached the outstanding invoice for the services rendered "
            "last quarter. Our records indicate this payment is now past due.\n\n"
            "You can review and settle the balance through our billing portal here: "
            "www.billing-portal-secure.net/inv/40219\n\n"
            "If you believe this message was sent in error, kindly confirm your "
            "account information so we can update our records.\n\n"
            "Best regards,\nAccounts Receivable"
        ),
    },
    {
        "name": "False positive — legitimate IT security notice",
        "expect": "BLOCK (wrongly)",
        "note": (
            "A genuine internal notice that the detector flags as phishing. "
            "It carries the same persuasion markers real phishing does — "
            "'password', 'security', 'verify', a reset URL — and S6 showed "
            "those four features are the whole detector. This is the documented "
            "false-positive mode, and the reason the DEFUSE tier exists: an "
            "uncertain verdict should strip the links, not bin the mail."
        ),
        "text": (
            "Subject: Scheduled password rotation this Friday\n\n"
            "Hi team,\n\n"
            "As part of our quarterly security policy, all staff passwords will "
            "need to be rotated before Friday 5pm. You can do this from the usual "
            "internal portal, or by visiting https://it.internal.example.org/reset.\n\n"
            "If you have any trouble, raise a ticket with the service desk and "
            "we will sort it out.\n\n"
            "Thanks,\nIT Operations"
        ),
    },
    {
        "name": "False negative — mild phishing, markers suppressed",
        "expect": "DELIVER (wrongly)",
        "note": (
            "Phishing written without urgency language, exclamation marks or a "
            "visible link — the evasion S7 measures. The detector scores it like "
            "ordinary mail because the persuasion features it depends on have "
            "been removed. No threshold fixes this; only the drift monitor "
            "notices, and only in aggregate."
        ),
        "text": (
            "Subject: Following up on the document you requested\n\n"
            "Hello,\n\n"
            "Further to our conversation last week, I have prepared the summary "
            "you asked about. The file is in the shared folder under the usual "
            "project directory, and I have granted your address read access.\n\n"
            "When you have a moment, could you sign in with your work credentials "
            "and let me know whether the figures in section two match what you "
            "have on record.\n\n"
            "Kind regards,\nDaniel"
        ),
    },
    {
        "name": "Legitimate — colleague follow-up",
        "expect": "DELIVER",
        "text": (
            "Subject: Re: notes from Tuesday's planning session\n\n"
            "Hi Ravi,\n\n"
            "Thanks for sending those through. I went over the deployment timeline "
            "and I think the second week is tight but workable, assuming the staging "
            "environment is ready by then.\n\n"
            "I have added a couple of comments in the shared document about the "
            "migration order. Happy to walk through it on Thursday if that is easier.\n\n"
            "Cheers,\nMeera"
        ),
    },
    {
        "name": "Legitimate — newsletter",
        "expect": "DELIVER",
        "text": (
            "Subject: This month in distributed systems\n\n"
            "Welcome to the October edition.\n\n"
            "This month we look at consensus under partial synchrony, a readable "
            "introduction to CRDTs, and a retrospective on three years of running "
            "a multi-region database. There is also a short interview with the "
            "maintainers of an open source scheduler about backpressure.\n\n"
            "As always, you can read the full issue on the website, and reply to "
            "this message if there is a topic you would like covered."
        ),
    },
]


service = Service()
