"""End-to-end pipeline: one command regenerates every table and figure.

    python -m genphish.run_all            # xgboost (primary)
    python -m genphish.run_all --logreg   # logistic-regression baseline

Writes results/*.csv, results/summary.json and results/figures/*.png, and prints
a readable summary. Stamps DATA_MODE (REAL / SYNTHETIC) on every output.
"""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
import pandas as pd

from .config import (GENERATORS, LABEL, SOURCE, RESULTS, FIGURES, SEED,
                     TARGET_RECALL)
from .corpus import load_corpus
from . import detectors as D, transfer as T, calibration as CAL
from . import drift_monitor as DM, zerotrust as ZT, error_analysis as EA
from . import evaluate as EV, plots as P


def _stream(det, phish_df, ham_df, n, prev, rng):
    npos = int(prev * n); nneg = n - npos
    pos = phish_df.sample(npos, replace=len(phish_df) < npos, random_state=int(rng.integers(1e9)))
    neg = ham_df.sample(nneg, replace=len(ham_df) < nneg, random_state=int(rng.integers(1e9)))
    return rng.permutation(det.score(pd.concat([pos, neg])))


def main(kind="xgboost"):
    t0 = time.time()
    df, is_real = load_corpus()
    mode = "REAL" if is_real else "SYNTHETIC"
    print(f"\n{'='*64}\n  genphish-guard pipeline   |   DATA MODE: {mode}   |   model: {kind}\n{'='*64}")
    if not is_real:
        print("  (synthetic corpus: numbers are REAL computations on SYNTHETIC\n"
              "   data; drop the Zenodo corpus_features.csv in data/ for real numbers)")

    # ---- S3/S4: detectors + transfer matrix -----------------------------
    dets, (ham_tr, ham_te, ph_tr, ph_te) = D.train_all(df, GENERATORS, kind)
    cache = T.score_cache(dets, ham_te, ph_te, GENERATORS)
    F = T.f1_matrix(cache, GENERATORS)
    A = T.auc_matrix(cache, GENERATORS)
    sF, sA = T.summarise(F), T.summarise(A)
    F.to_csv(RESULTS / "transfer_f1_default.csv")
    A.to_csv(RESULTS / "transfer_auc.csv")
    P.heatmap(F, f"Transfer F1 at default threshold ({mode})", "fig1_transfer_f1.png", is_real)
    P.heatmap(A, f"Transfer AUC — signal survives ({mode})", "fig2_transfer_auc.png", is_real, vmin=0.5)
    print(f"\n[S4] Transfer matrix  (F1 @0.5)")
    print(F.round(3).to_string())
    print(f"  diagonal F1 {sF['diagonal_mean']:.3f} | off-diagonal F1 {sF['off_diagonal_mean']:.3f} "
          f"| gap {sF['transfer_gap_pp']:.1f} pp | off-diagonal AUC {sA['off_diagonal_mean']:.3f}")

    # ---- S5: label-free calibration (off-diagonal pairs) ----------------
    cres = []
    for s in GENERATORS:
        ss, sy = cache[(s, s)]
        for tgt in GENERATORS:
            if s == tgt:
                continue
            ts, ty = cache[(s, tgt)]
            cres.append(CAL.calibrate_pair(s, tgt, ss, sy, ts, ty))
    cal_df = pd.DataFrame([vars(r) | {
        "rec_dm": r.recovery("dm"), "rec_em": r.recovery("em"), "rec_self": r.recovery("self")
    } for r in cres])
    cal_df.to_csv(RESULTS / "calibration_pairs.csv", index=False)
    mean_f1 = {k: float(cal_df[f"f1_{k}"].mean()) for k in ["stale", "dm", "em", "self", "oracle"]}
    rec = {k: float(cal_df[f"rec_{k}"].mean()) for k in ["dm", "em", "self"]}
    P.calibration_bars(mean_f1, f"Label-free calibration recovery ({mode})",
                       "fig3_calibration_recovery.png", is_real)
    print(f"\n[S5] Label-free calibration  (mean over {len(cres)} cross-generator pairs)")
    print(f"  stale F1 {mean_f1['stale']:.3f}  ->  DM {mean_f1['dm']:.3f} ({rec['dm']*100:.0f}% gap) | "
          f"EM {mean_f1['em']:.3f} ({rec['em']*100:.0f}%) | self {mean_f1['self']:.3f} ({rec['self']*100:.0f}%) "
          f"| oracle {mean_f1['oracle']:.3f}")

    # ---- Drift monitor: severe / mild / no-shift ------------------------
    rng = np.random.default_rng(SEED)
    src = "deepseek3.2"; det = dets[src]
    h = ham_te.sample(frac=1, random_state=1)
    ref_h, live_h = h.iloc[:len(h)//2], h.iloc[len(h)//2:]
    band = (float(np.quantile(det.score(ref_h), 0.99)),    # top of legitimate mail
            float(np.quantile(det.score(ph_te[src]), 0.01)))   # bottom of known phishing
    cal = _stream(det, ph_te[src], live_h, 3000, 0.25, rng)    # source reference (r0), deployment pool
    pre = _stream(det, ph_te[src], live_h, 1500, 0.25, rng)    # source-era delivered mail
    scen = {"severe (→llama)": "llama-3.3-70b", "mild (→gpt-4.1)": "gpt-4.1"}
    trajs, drift_rows = {}, []
    for lab, tgt in scen.items():
        post = _stream(det, ph_te[tgt], live_h, 1500, 0.25, rng)
        r = DM.monitor_stream(cal, pre, post, band)
        trajs[lab] = (r.wealth, r.switch_index)
        drift_rows.append({"scenario": lab, "target": tgt, "switch_idx": r.switch_index,
                           "alarm_idx": r.alarm_index, "ttd_emails": r.ttd,
                           "max_wealth": float(r.wealth.max()), "fired": r.fired})
    post_ctrl = _stream(det, ph_te[src], live_h, 1500, 0.25, rng)
    rc = DM.monitor_stream(cal, pre, post_ctrl, band)
    trajs["no shift (control)"] = (rc.wealth, rc.switch_index)
    drift_rows.append({"scenario": "no shift (control)", "target": src,
                       "switch_idx": rc.switch_index, "alarm_idx": rc.alarm_index,
                       "ttd_emails": rc.ttd, "max_wealth": float(rc.wealth.max()), "fired": rc.fired})
    pd.DataFrame(drift_rows).to_csv(RESULTS / "drift_monitor.csv", index=False)
    P.drift_trajectories(trajs, 1.0 / 0.01, f"Conformal drift monitor ({mode})",
                         "fig5_drift_monitor.png", is_real)
    print(f"\n[Drift] band=[{band[0]:.3f},{band[1]:.3f}]  null rate r0={rc.r0:.4f}  δ=0.01 ceiling=100")
    for row in drift_rows:
        print(f"  {row['scenario']:<22} fired={str(row['fired']):>5}  maxW={row['max_wealth']:>10.1f}  "
              f"TTD={row['ttd_emails']} emails")

    # ---- Zero-trust layer (worst shifted pair) --------------------------
    ts, ty = cache[("deepseek3.2", "llama-3.3-70b")]
    pts, win = ZT.sweep_cost_ratio(ts, ty)
    pd.DataFrame([vars(p) for p in pts]).to_csv(RESULTS / "zerotrust_sweep.csv", index=False)
    legit = ZT.legit_preserved_at_equal_containment(ts, ty)
    P.zerotrust_curves(pts, win, f"Zero-trust expected cost ({mode})",
                       "fig6_zerotrust.png", is_real)
    print(f"\n[Zero-trust] graded policy wins for cost ratio in "
          f"{('[%.1f, %.0f]' % win) if win else 'none'}")
    print(f"  legit mail rescued from blocking at equal containment: "
          f"{legit['legit_rescued_from_block']*100:.1f}% of legitimate mail")

    # ---- Error analysis (worst pair) ------------------------------------
    worst_src, worst_tgt = "deepseek3.2", "llama-3.3-70b"
    eval_df = pd.concat([ph_te[worst_tgt], ham_te], ignore_index=True)
    wscores = dets[worst_src].score(eval_df)
    stale_thr = CAL.threshold_at_recall(*cache[(worst_src, worst_src)])
    dm_thr = CAL.dm_threshold(cache[(worst_src, worst_src)][0], stale_thr, wscores)
    err, prof = EA.summarise_missed(eval_df, wscores, stale_thr, dm_thr)
    prof.to_csv(RESULTS / "error_profile.csv")
    P.error_profile(prof, f"Missed vs caught phishing, {worst_src}→{worst_tgt} ({mode})",
                    "fig7_error_profile.png", is_real)
    print(f"\n[Error analysis] {worst_src}→{worst_tgt}: "
          f"{err['n_missed_stale']}/{err['n_phish']} phishing missed at stale threshold "
          f"({err['miss_rate_stale']*100:.0f}%); DM recovers {err['recovery_rate']*100:.0f}% of them")
    print(f"  top discriminating features: {', '.join(err['top_discriminating_features'][:4])}")

    # ---- score-shift figure --------------------------------------------
    P.score_shift({
        "legitimate": dets[worst_src].score(ham_te),
        "source phishing": dets[worst_src].score(ph_te[worst_src]),
        "target phishing (shifted)": dets[worst_src].score(ph_te[worst_tgt]),
    }, f"Score shift under generator change ({mode})", "fig8_score_shift.png", is_real)

    # ---- bootstrap CI on the headline number ----------------------------
    # CI is built around the statistic actually reported: the MEAN of the
    # off-diagonal per-pair AUCs. (Bootstrapping the pooled-scores AUC instead
    # estimates a different quantity and can produce an interval that does not
    # contain the reported mean.)
    auc_ci = EV.mean_offdiag_auc_ci(cache, GENERATORS, n=1000)

    # ---- summary.json ---------------------------------------------------
    summary = {
        "data_mode": mode, "model": kind, "seed": SEED, "n_emails": int(len(df)),
        "generators": GENERATORS,
        "transfer": {"f1": sF, "auc": sA},
        "transfer_auc_offdiag_ci95": {"point": auc_ci[0], "lo": auc_ci[1], "hi": auc_ci[2]},
        "calibration_mean_f1": mean_f1,
        "calibration_gap_recovery": rec,
        "drift_monitor": drift_rows,
        "zerotrust_win_ratio_range": win,
        "zerotrust_legit_preserved": legit,
        "error_analysis": {k: v for k, v in err.items() if k != "top_discriminating_features"} | {
            "top_features": err["top_discriminating_features"]},
        "runtime_sec": round(time.time() - t0, 1),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] {summary['runtime_sec']}s | results/ written | figures in results/figures/")
    print(f"       mean off-diagonal AUC {auc_ci[0]:.3f}, 95% CI [{auc_ci[1]:.3f}, {auc_ci[2]:.3f}]")
    print( "       extended studies (S6 ablations, S7 robustness, neural arm, stability):")
    print( "         python -m genphish.run_extended [--stability]\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logreg", action="store_true", help="use logistic regression instead of xgboost")
    args = ap.parse_args()
    main(kind="logreg" if args.logreg else "xgboost")
