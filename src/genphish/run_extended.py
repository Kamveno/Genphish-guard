"""Extended studies: S6 ablations, S7 robustness, the neural arm, stability.

`run_all` stays a ~15 s live demo of the core claim. These studies retrain the
detector set dozens of times, so they live behind their own command:

    python -m genphish.run_extended                 # S6 + S7 + neural arm
    python -m genphish.run_extended --stability     # + the multi-seed study
    python -m genphish.run_extended --seeds 4       # shorter stability sweep
    python -m genphish.run_extended --only robustness

Writes results/*.csv, results/extended_summary.json and the fig9-fig13 figures,
stamped REAL/SYNTHETIC exactly like the core pipeline.
"""
from __future__ import annotations
import argparse, json, time

import numpy as np
import pandas as pd

from .config import GENERATORS, RESULTS, FIGURES, SEED
from .corpus import load_corpus
from . import ablation as AB, robustness as RB, neural as NN, stability as ST
from . import plots as P

STUDIES = ("ablations", "robustness", "neural", "stability")


def _write(tables: dict) -> None:
    for name, frame in tables.items():
        if name.startswith("_"):
            continue
        frame.to_csv(RESULTS / f"{name}.csv", index=False)


def _h(title: str) -> None:
    print(f"\n{'-'*68}\n  {title}\n{'-'*68}")


def main(only=None, n_seeds: int = 10, kind: str = "xgboost",
         with_stability: bool = False, control_perms: int = 8) -> dict:
    t0 = time.time()
    only = set(only or STUDIES)
    if with_stability:
        only.add("stability")

    df, is_real = load_corpus()
    mode = "REAL" if is_real else "SYNTHETIC"
    print(f"\n{'='*68}\n  genphish-guard EXTENDED   |   DATA MODE: {mode}   |   model: {kind}"
          f"\n  studies: {', '.join(sorted(only))}\n{'='*68}")
    if not is_real:
        print("  (synthetic corpus: real computations on synthetic data)")

    summary: dict = {"data_mode": mode, "model": kind, "seed": SEED,
                     "studies": sorted(only)}

    # ------------------------------------------------------------ S6
    if "ablations" in only:
        _h("S6  ablations and controls")
        tables = AB.run_all_ablations(df, GENERATORS, kind, SEED,
                                      control_perms=control_perms)
        _write(tables)

        feats = tables["ablation_features"]
        print("\n[S6.1] feature-group ablation")
        print(feats[["config", "transfer_gap_pp", "offdiag_auc",
                     "f1_stale", "f1_dm", "rec_dm"]].round(3).to_string(index=False))
        P.ablation_bars(feats, f"S6 feature ablation ({mode})",
                        "fig9_ablation_features.png", is_real)

        ctrl = tables["ablation_shuffled_control"]
        print("\n[S6.2] shuffled-label negative control (permutation test)")
        print(ctrl.round(4).to_string(index=False))
        null_row = ctrl[ctrl["config"].str.startswith("shuffled")].iloc[0]
        print(f"  -> null AUC {null_row['offdiag_auc_mean']:.3f} "
              f"(range {null_row['offdiag_auc_min']:.3f}-{null_row['offdiag_auc_max']:.3f})"
              f"  near chance: {bool(null_row['near_chance'])}")
        print(f"     real beats every permutation: "
              f"{bool(null_row['real_exceeds_all_permutations'])}  "
              f"permutation p <= {null_row['permutation_p']:.3f}")

        fam = tables["ablation_detector_family"]
        print("\n[S6.3] detector-family ablation")
        print(fam[["config", "transfer_gap_pp", "offdiag_auc",
                   "f1_stale", "f1_dm", "rec_dm"]].round(3).to_string(index=False))
        P.family_bars(fam, f"S6 detector family ({mode})",
                      "fig13_detector_family.png", is_real)

        prov = tables["ablation_provenance"]
        print("\n[S6.4] provenance ablation — 'why not just pool the generators?'")
        print(prov.round(3).to_string(index=False))

        sens = tables["ablation_calibration_sensitivity"]
        print("\n[S6.5] calibration hyperparameter sensitivity")
        print(sens.round(3).to_string(index=False))

        summary["ablation"] = {
            "feature_groups": feats.set_index("config")[
                ["transfer_gap_pp", "offdiag_auc", "rec_dm"]].to_dict("index"),
            "shuffled_control_null_auc": float(null_row["offdiag_auc_mean"]),
            "shuffled_control_null_range": [float(null_row["offdiag_auc_min"]),
                                            float(null_row["offdiag_auc_max"])],
            "shuffled_control_permutation_p": float(null_row["permutation_p"]),
            "shuffled_control_passes": bool(
                null_row["near_chance"]
                and null_row["real_exceeds_all_permutations"]),
            "detector_family": fam.set_index("config")[
                ["transfer_gap_pp", "offdiag_auc", "f1_stale", "f1_dm", "rec_dm"]
            ].to_dict("index"),
            "pooled_upper_bound_f1": float(prov["f1_pooled_seen_target"].mean()),
            "dm_share_of_pooled_gap": float(prov["dm_closes_pct_of_pooled_gap"].mean()),
            "dm_recovery_range_over_operating_recall": [
                float(sens[sens.method == "dm"]["mean_recovery"].min()),
                float(sens[sens.method == "dm"]["mean_recovery"].max())],
        }

    # ------------------------------------------------------------ S7
    if "robustness" in only:
        _h("S7  robustness — paraphrase, register shift, targeted evasion")
        rob = RB.robustness_study(df, GENERATORS, kind, SEED)
        cur = RB.evasion_curve(df, GENERATORS, seed=SEED)
        rob.to_csv(RESULTS / "robustness_scenarios.csv", index=False)
        cur.to_csv(RESULTS / "robustness_evasion_curve.csv", index=False)
        print(rob[["scenario", "offdiag_auc", "diag_f1", "f1_stale", "f1_dm",
                   "f1_oracle", "rec_dm", "auc_drop_pp", "verdict"]]
              .round(3).to_string(index=False))
        P.robustness_panel(rob, f"S7 robustness to rewriting and evasion ({mode})",
                           "fig10_robustness.png", is_real)
        P.evasion_curve_plot(cur, f"S7 targeted evasion sweep ({mode})",
                             "fig11_evasion_curve.png", is_real)

        worst = rob.iloc[rob["f1_stale"].values.argmin()]
        deq = rob[rob["scenario"].str.startswith("de-quantization")].iloc[0]
        brittle = {}
        for m in cur["model"].unique():
            d = cur[cur["model"] == m].sort_values("eps")
            steps = d["offdiag_auc"].diff().abs().dropna()
            brittle[m] = {
                "auc_at_eps_0": float(d.iloc[0]["offdiag_auc"]),
                "auc_at_eps_1": float(d.iloc[-1]["offdiag_auc"]),
                "largest_single_step_auc_drop": float(steps.max()),
                "failure_shape": "step" if steps.max() > 0.08 else "gradual",
            }
        summary["robustness"] = {
            "scenarios": rob.set_index("scenario")[
                ["offdiag_auc", "f1_stale", "f1_dm", "rec_dm", "verdict"]
            ].to_dict("index"),
            "worst_scenario": str(worst["scenario"]),
            "worst_f1_stale": float(worst["f1_stale"]),
            "worst_f1_after_recalibration": float(worst["f1_dm"]),
            "recalibration_helps_in_all_scenarios": bool(
                (rob["f1_dm"] >= rob["f1_stale"] - 1e-9).all()),
            "dequantization_auc_cost_pp": float(deq["auc_drop_pp"]),
            "evasion_failure_shape": brittle,
        }
        print(f"\n  worst case: {worst['scenario']} — stale F1 {worst['f1_stale']:.3f}"
              f" -> {worst['f1_dm']:.3f} after label-free recalibration")
        print(f"  de-quantization probe costs {deq['auc_drop_pp']:.2f} pp AUC for a "
              f"1e-6 jitter — a representation brittleness, not an attack")
        for m, b in brittle.items():
            print(f"  evasion failure shape [{m}]: {b['failure_shape']}"
                  f"  (largest single-step AUC drop {b['largest_single_step_auc_drop']:.3f})")

    # ------------------------------------------------------------ neural arm
    if "neural" in only:
        _h("Neural arm — is the finding representation-specific?")
        narm = NN.run_neural_arm(df, GENERATORS, SEED)
        _write(narm)
        print(narm["neural_arm_summary"].round(3).to_string(index=False))
        v, rdy = narm["_verdict"], narm["_readiness"]
        print(f"\n  gap recovery by family: "
              f"{ {k: round(x, 3) for k, x in v['rec_dm_by_family'].items()} }")
        print(f"  label-free calibration helps in every family: "
              f"{v['label_free_helps_in_every_family']}   "
              f"spread {v['rec_dm_spread']:.3f} (tol {v['tolerance']})")
        print(f"  text-encoder arm: {rdy['status']}")
        if rdy["blocked_on"]:
            print(f"    blocked on : {rdy['blocked_on']}")
            print(f"    unblocked by: {rdy['unblocking_action']}")
        summary["neural_arm"] = {**v, "text_encoder_readiness": rdy}

    # ------------------------------------------------------------ stability
    if "stability" in only:
        _h(f"Stability — {n_seeds} partitions, paired significance tests")
        seeds = ST.DEFAULT_SEEDS[:n_seeds]
        tabs = ST.run_stability(df, GENERATORS, kind, seeds)
        _write(tabs)
        print("\n[stability] headline quantities across partitions")
        print(tabs["stability_summary"].round(4).to_string(index=False))
        print("\n[significance] distribution matching vs each alternative "
              "(paired on seed × source × target, Holm-corrected)")
        sig = tabs["stability_significance"].copy()
        for c in ("mean_diff_f1", "cohens_dz"):
            sig[c] = sig[c].round(4)
        for c in ("p_raw", "p_holm"):          # these go far below 1e-5
            sig[c] = sig[c].map(lambda v: f"{v:.2e}")
        print(sig[["comparison", "n_pairs", "mean_diff_f1", "cohens_dz",
                   "p_raw", "p_holm", "significant_05", "reading"]]
              .to_string(index=False))
        print("\n[per-pair] where distribution matching helps least")
        print(tabs["stability_per_pair"].round(3).head(3).to_string(index=False))
        P.stability_box(tabs["_long"], f"Stability across partitions ({mode})",
                        "fig12_stability.png", is_real)

        s = tabs["stability_summary"].set_index("quantity")
        summary["stability"] = {
            "n_seeds": len(seeds),
            "transfer_gap_pp": s.loc["transfer_gap_pp", ["mean", "sd", "ci95_lo", "ci95_hi"]].to_dict(),
            "rec_dm": s.loc["rec_dm", ["mean", "sd", "ci95_lo", "ci95_hi"]].to_dict(),
            "f1_dm": s.loc["f1_dm", ["mean", "sd", "ci95_lo", "ci95_hi"]].to_dict(),
            "significance": tabs["stability_significance"].set_index("comparison")[
                ["mean_diff_f1", "cohens_dz", "p_raw", "p_holm",
                 "significant_05", "reading"]].to_dict("index"),
            "dm_beats_stale_on_every_pair": bool(
                (tabs["stability_per_pair"]["win_rate"] == 1.0).all()),
        }

    summary["runtime_sec"] = round(time.time() - t0, 1)
    (RESULTS / "extended_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"\n[done] extended studies in {summary['runtime_sec']}s  |  "
          f"results/extended_summary.json + figures written\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=STUDIES,
                    help="run only these studies (default: ablations robustness neural)")
    ap.add_argument("--stability", action="store_true",
                    help="also run the multi-seed stability + significance study")
    ap.add_argument("--seeds", type=int, default=10,
                    help="number of partitions for the stability study")
    ap.add_argument("--control-perms", type=int, default=8,
                    help="permutation replicates for the shuffled-label control; "
                         "a permutation p cannot go below 1/(n+1), so use 19 "
                         "if a p < 0.05 is wanted in the write-up")
    ap.add_argument("--logreg", action="store_true")
    a = ap.parse_args()
    main(only=a.only or ["ablations", "robustness", "neural"],
         n_seeds=a.seeds, kind="logreg" if a.logreg else "xgboost",
         with_stability=a.stability, control_perms=a.control_perms)
