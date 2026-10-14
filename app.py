"""genphish-guard  —  Detection Dashboard
=========================================
Streamlit live demo.  Run:

    pip install streamlit plotly
    cd genphish-guard
    streamlit run app.py

Tabs
----
1. Live Detect   — paste / upload email → confidence gauge → zero-trust tier
2. Transfer Map  — pre-computed N×N AUC / F1 heatmaps
3. Drift Monitor — score stream and martingale alarm
4. SOAR Log      — incident queue and simulated playbook responses
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ page config
st.set_page_config(
    page_title="genphish-guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ lazy imports
@st.cache_resource(show_spinner="Loading model pipeline…")
def _load():
    """Load corpus + detectors once. Cached across reruns."""
    from genphish.corpus import load_corpus
    from genphish.detectors import train_all
    from genphish.config import GENERATORS, SEED
    from genphish import transfer as T, calibration as CAL, features as F

    df, is_real = load_corpus()
    dets, splits = train_all(df, GENERATORS, kind="xgboost", seed=SEED)
    ham_tr, ham_te, ph_tr, ph_te = splits
    cache = T.score_cache(dets, ham_te, ph_te, GENERATORS)
    return df, dets, cache, splits, GENERATORS, is_real, F, T, CAL


# ------------------------------------------------------------------ sidebar
st.sidebar.image("https://img.shields.io/badge/genphish--guard-Review%202-blue", width=200)
st.sidebar.title("genphish-guard 🛡️")
st.sidebar.caption("Generator-agnostic phishing detection with label-free recalibration")

try:
    summary_path = Path("results/extended_summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        mode = summary.get("data_mode", "UNKNOWN")
        st.sidebar.metric("Data mode", mode)
        if "stability" in summary:
            gap = summary["stability"]["transfer_gap_pp"]["mean"]
            rec = summary["stability"]["rec_dm"]["mean"] * 100
            st.sidebar.metric("Transfer gap", f"{gap:.1f} pp")
            st.sidebar.metric("DM recovery", f"{rec:.1f} %")
except Exception:
    pass

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Zenodo corpus**: drop `corpus_features.csv` from "
    "[DOI 10.5281/zenodo.20250116](https://zenodo.org/records/20250116) "
    "into `data/` and restart."
)

# ------------------------------------------------------------------ load
with st.spinner("Initialising pipeline…"):
    try:
        df, dets, cache, splits, GENERATORS, is_real, F, T, CAL = _load()
        pipeline_ok = True
    except Exception as e:
        st.error(f"Pipeline failed to load: {e}")
        pipeline_ok = False

if not pipeline_ok:
    st.stop()

from genphish.config import FEATURES, LABEL, SOURCE, SEED
from genphish import drift_monitor as DM, zerotrust as ZT
from genphish import soar as SOAR

mode_badge = "🟢 REAL" if is_real else "🟡 SYNTHETIC"
st.title(f"genphish-guard Detection Dashboard  {mode_badge}")
if not is_real:
    st.info(
        "Running on **synthetic** corpus. Drop `corpus_features.csv` from Zenodo into "
        "`data/` and restart to switch to REAL data — all numbers update automatically."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Live Detect", "📊 Transfer Map", "📡 Drift Monitor", "🚨 SOAR Log"]
)


# ══════════════════════════════════════════════════════════════════════ TAB 1
with tab1:
    st.header("Live Email Classification")
    st.caption(
        "Paste email text or upload a CSV with email features. "
        "The pipeline extracts features, scores with the pooled detector, "
        "and assigns a zero-trust tier."
    )

    col_in, col_out = st.columns([1, 1])

    with col_in:
        mode_choice = st.radio("Input mode", ["Paste email text", "Upload feature CSV"],
                               horizontal=True)

        detector_choice = st.selectbox(
            "Source generator (detector to use)",
            ["pooled"] + list(GENERATORS),
            help="'pooled' is trained on all generators. Single-generator detectors "
                 "show what happens under cross-generator transfer."
        )
        cal_method = st.selectbox(
            "Calibration method",
            ["stale (no recalibration)", "distribution matching", "oracle (labelled)"],
        )

        if mode_choice == "Paste email text":
            raw_text = st.text_area(
                "Email body",
                height=200,
                placeholder="Paste the email body here. Features will be extracted automatically.",
                value=(
                    "URGENT: Your account has been compromised! "
                    "Click here immediately to verify your credentials: http://malicious.example.com "
                    "Failure to act within 24 hours will result in permanent suspension!!!"
                ),
            )
            run_btn = st.button("Classify", type="primary", use_container_width=True)

            features_df = None
            if run_btn and raw_text.strip():
                try:
                    features_df = F.extract_frame(pd.DataFrame({"body": [raw_text]}))
                except Exception as ex:
                    st.error(f"Feature extraction failed: {ex}")

        else:
            uploaded = st.file_uploader(
                "Upload feature CSV (same schema as corpus_features.csv)",
                type=["csv"],
            )
            run_btn = st.button("Classify batch", type="primary", use_container_width=True)
            features_df = None
            if uploaded and run_btn:
                try:
                    raw = pd.read_csv(uploaded)
                    # keep only feature columns present
                    keep = [c for c in FEATURES if c in raw.columns]
                    if len(keep) < len(FEATURES):
                        missing = set(FEATURES) - set(keep)
                        st.warning(f"Missing features filled with 0: {missing}")
                        for m in missing:
                            raw[m] = 0
                    features_df = raw
                except Exception as ex:
                    st.error(f"CSV parse error: {ex}")

    with col_out:
        if features_df is not None and run_btn:
            det = dets.get(detector_choice, dets["pooled"])

            # raw scores
            scores = det.score(features_df)

            # calibration
            if cal_method.startswith("distribution") and detector_choice in GENERATORS:
                # find a source calibration pair
                src = detector_choice
                tgt_scores_ref = []
                for tgt in GENERATORS:
                    if tgt != src:
                        ss, sy = cache.get((src, src), (scores, np.ones(len(scores))))
                        thr_raw = CAL.threshold_at_recall(ss, sy)
                        cal_scores = CAL.dm_threshold(ss, thr_raw, scores)
                        tgt_scores_ref.append(cal_scores)
                calibrated_thr = float(np.mean(tgt_scores_ref)) if tgt_scores_ref else 0.5
                threshold = calibrated_thr
                method_label = "Distribution Matching"
            else:
                # stale: use pooled default
                threshold = 0.5
                method_label = "Stale (no recalibration)" if "stale" in cal_method else "Oracle"

            predictions = (scores >= threshold).astype(int)
            mean_score = float(scores.mean())

            # zero-trust tier
            tier_rows = []
            for s in scores:
                tier = ZT.classify(s, threshold)
                tier_rows.append(tier)

            # headline metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean confidence", f"{mean_score:.3f}")
            c2.metric("Threshold", f"{threshold:.3f}", help=method_label)
            phish_pct = float(predictions.mean()) * 100
            c3.metric("% flagged phishing", f"{phish_pct:.0f}%")

            # confidence gauge (plotly)
            try:
                import plotly.graph_objects as go
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mean_score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Phishing confidence"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#e63946"},
                        "steps": [
                            {"range": [0, 0.35], "color": "#2a9d8f"},
                            {"range": [0.35, 0.65], "color": "#e9c46a"},
                            {"range": [0.65, 1.0], "color": "#e76f51"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": threshold,
                        },
                    },
                ))
                fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.progress(mean_score)

            # zero-trust tier badge
            tier_counts = pd.Series(tier_rows).value_counts()
            st.subheader("Zero-trust decision")
            for tier, count in tier_counts.items():
                if tier == "BLOCK":
                    st.error(f"🚫 BLOCK — {count} email(s)")
                elif tier == "DEFUSE":
                    st.warning(f"⚠️ DEFUSE / QUARANTINE — {count} email(s)")
                else:
                    st.success(f"✅ DELIVER — {count} email(s)")

            # per-email table
            result_df = features_df[FEATURES].copy() if set(FEATURES).issubset(features_df.columns) else features_df.head()
            result_df["confidence"] = scores.round(4)
            result_df["tier"] = tier_rows
            result_df["flagged"] = predictions
            st.dataframe(result_df[["confidence", "tier", "flagged"] +
                                   ["urgency_term_count", "url_count", "exclamation_count"]
                                   [:len([c for c in ["urgency_term_count", "url_count", "exclamation_count"]
                                          if c in result_df.columns])]],
                         use_container_width=True)

            # SOAR trigger
            if phish_pct > 0:
                incidents = SOAR.incidents_from_scores(
                    scores, tier_rows, method_label, detector_choice
                )
                SOAR.log_to_session(incidents)
                st.caption(f"↗ {len(incidents)} incident(s) raised to SOAR log (Tab 4)")
        else:
            st.info("Configure input and click Classify.")


# ══════════════════════════════════════════════════════════════════════ TAB 2
with tab2:
    st.header("Transfer Matrix")
    st.caption(
        "Off-diagonal = cross-generator transfer. AUC (threshold-free) stays high; "
        "F1 at the stale threshold collapses — that is the core finding."
    )

    auc_path = Path("results/transfer_auc.csv")
    f1_path  = Path("results/transfer_f1_default.csv")

    if auc_path.exists() and f1_path.exists():
        try:
            import plotly.graph_objects as go
            auc_df = pd.read_csv(auc_path, index_col=0)
            f1_df  = pd.read_csv(f1_path, index_col=0)

            col_a, col_b = st.columns(2)
            for col, mat, title, fmt in (
                (col_a, auc_df, "AUC (threshold-free)", ".3f"),
                (col_b, f1_df,  "F1 at stale threshold", ".3f"),
            ):
                with col:
                    fig = go.Figure(go.Heatmap(
                        z=mat.values,
                        x=list(mat.columns),
                        y=list(mat.index),
                        colorscale="RdYlGn",
                        zmin=0, zmax=1,
                        text=mat.round(3).values,
                        texttemplate="%{text}",
                        showscale=True,
                    ))
                    fig.update_layout(
                        title=title,
                        xaxis_title="Target generator",
                        yaxis_title="Source generator",
                        height=350,
                        margin=dict(t=50, b=30, l=80, r=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # gap summary
            off_auc = []
            off_f1  = []
            for r in auc_df.index:
                for c in auc_df.columns:
                    if r != c:
                        off_auc.append(auc_df.loc[r, c])
                        off_f1.append(f1_df.loc[r, c])
            diag_f1 = [f1_df.loc[r, r] for r in f1_df.index]

            ga, gb, gc = st.columns(3)
            ga.metric("Mean off-diagonal AUC",  f"{np.mean(off_auc):.3f}")
            gb.metric("Mean off-diagonal F1",   f"{np.mean(off_f1):.3f}")
            gc.metric("Transfer gap (F1)",       f"{np.mean(diag_f1)-np.mean(off_f1):.1f} pp")
        except Exception as e:
            st.error(f"Could not render heatmaps: {e}")
    else:
        st.warning("Run `python -m genphish.run_all` first to generate transfer matrices.")


# ══════════════════════════════════════════════════════════════════════ TAB 3
with tab3:
    st.header("Drift Monitor (conformal martingale)")
    st.caption(
        "The martingale watches the unlabelled score stream. When exchangeability breaks "
        "(a new generator is arriving) it raises an anytime-valid alarm, triggering recalibration."
    )

    drift_path = Path("results/drift_monitor.csv")
    if drift_path.exists():
        try:
            dm_df = pd.read_csv(drift_path)
            import plotly.express as px

            fig = px.line(
                dm_df, x=dm_df.index, y="martingale_value",
                title="Betting martingale value (log scale)",
                labels={"index": "Email index", "martingale_value": "Martingale"},
                log_y=True,
            )
            alarm_threshold = 1 / 0.01  # 1/δ, δ=0.01
            fig.add_hline(
                y=alarm_threshold, line_dash="dash", line_color="red",
                annotation_text=f"Alarm threshold (1/δ = {alarm_threshold:.0f})",
            )
            # mark alarms
            alarms = dm_df[dm_df["alarm"] == True] if "alarm" in dm_df.columns else pd.DataFrame()
            if not alarms.empty:
                fig.add_scatter(
                    x=alarms.index, y=alarms["martingale_value"],
                    mode="markers", marker=dict(color="red", size=10, symbol="x"),
                    name="Alarm",
                )
            fig.update_layout(height=380, margin=dict(t=50, b=30, l=60, r=20))
            st.plotly_chart(fig, use_container_width=True)

            n_alarms = int(dm_df["alarm"].sum()) if "alarm" in dm_df.columns else 0
            c1, c2 = st.columns(2)
            c1.metric("Emails processed", len(dm_df))
            c2.metric("Alarms raised", n_alarms)
        except Exception as e:
            st.error(f"Could not render drift plot: {e}")
    else:
        st.warning("Run `python -m genphish.run_all` first.")

    st.markdown("---")
    st.subheader("Simulate incoming stream")
    n_sim = st.slider("Emails to simulate", 50, 500, 200, 50)
    shift_at = st.slider("Inject generator shift at email #", 10, n_sim - 10, n_sim // 2)
    if st.button("Run simulation"):
        rng = np.random.default_rng(42)
        pre  = rng.beta(2, 5, shift_at)        # low phishing scores (ham-like)
        post = rng.beta(6, 2, n_sim - shift_at) # high phishing after shift
        stream = np.concatenate([pre, post])

        monitor = DM.ConformalMartingale()
        mart_vals, alarms = [], []
        for i, s in enumerate(stream):
            v, alarm = monitor.update(s)
            mart_vals.append(v)
            alarms.append(alarm)

        sim_df = pd.DataFrame({"score": stream, "martingale": mart_vals, "alarm": alarms})

        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=mart_vals, mode="lines", name="Martingale",
                line=dict(color="#2196f3"),
            ))
            fig2.add_hline(y=100, line_dash="dash", line_color="red",
                          annotation_text="Alarm (1/0.01)")
            alarm_idx = [i for i, a in enumerate(alarms) if a]
            if alarm_idx:
                fig2.add_scatter(
                    x=alarm_idx,
                    y=[mart_vals[i] for i in alarm_idx],
                    mode="markers", marker=dict(color="red", size=10, symbol="x"),
                    name="Alarm",
                )
            fig2.add_vline(x=shift_at, line_dash="dot", line_color="orange",
                          annotation_text="Generator shift injected")
            fig2.update_layout(
                title="Simulated stream — martingale response",
                yaxis_type="log", height=350,
                margin=dict(t=50, b=30, l=60, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.success(
                f"Alarm raised at email #{alarm_idx[0] if alarm_idx else 'never'} "
                f"(shift injected at #{shift_at})"
            )
        except ImportError:
            st.line_chart(pd.DataFrame({"martingale": mart_vals}))


# ══════════════════════════════════════════════════════════════════════ TAB 4
with tab4:
    st.header("SOAR Integration — Incident Log")
    st.caption(
        "Incidents raised by the Live Detect tab appear here. "
        "In production, these POST to a Splunk SOAR / generic webhook endpoint."
    )

    incidents = SOAR.get_session_log()
    if not incidents:
        st.info("No incidents yet. Classify a phishing email in Tab 1.")
    else:
        inc_df = pd.DataFrame([i.to_dict() for i in incidents])
        st.dataframe(inc_df, use_container_width=True)

        # simulated playbook response
        st.subheader("Simulated playbook execution")
        for inc in incidents[:5]:  # show up to 5
            with st.expander(f"Incident {inc.id}  —  tier: {inc.tier}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.json(inc.to_dict())
                with col2:
                    playbook = SOAR.playbook_response(inc)
                    st.markdown("**Playbook actions**")
                    for step in playbook:
                        icon = "✅" if step["status"] == "executed" else "⏭️"
                        st.write(f"{icon} `{step['action']}` — {step['detail']}")

    st.markdown("---")
    st.subheader("SOAR connector configuration")
    webhook_url = st.text_input(
        "Webhook endpoint (leave blank for simulation mode)",
        placeholder="https://your-soar.example.com/api/incidents",
    )
    if webhook_url:
        st.info(
            f"In production, incidents will POST to `{webhook_url}`. "
            "See `src/genphish/soar.py → GenericWebhookConnector`."
        )
    else:
        st.caption("Running in **simulation mode** — no real SOAR endpoint needed for the demo.")
