import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)

import ml_model as M

st.set_page_config(
    page_title="GNSS Spoofing Detection Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Fixed raw dataset path
DATA_FILE = Path(__file__).resolve().parent / "raw_dataset.csv"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

STATUS_COLORS = {
    "Clear": "#2ca02c",
    "Clean": "#2ca02c",
    "Spoofed": "#d62728",
    "Jammed": "#ff7f0e",
    "No fix (<4 sats)": "#7f7f7f"
}

# --------------------------------------------------------------------------- #
# Cached Data Loaders and ML Trainers (80/20 train/test split handled internally)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading raw_dataset.csv ...")
def load_scene(path: str, mtime: float):
    raw = pd.read_csv(path)
    return len(raw), M.build_scene(raw)


@st.cache_resource(show_spinner="Training machine learning model ...")
def train_model_internal(path: str, mtime: float, model_name: str,
                         seed: int, threshold_m: float, label_mode: str, n_trees: int):
    _, scene = load_scene(path, mtime)
    # Internally splits 80% train / 20% test randomly
    return M.run_80_20_train_test(
        scene, model_name=model_name, test_size=0.20, seed=seed,
        threshold_m=threshold_m, label_mode=label_mode, n_trees=n_trees,
        compare_all=True
    )


@st.cache_data(show_spinner="Running Isolation Forest ...")
def run_isolation(path: str, mtime: float, contamination: float, seed: int):
    _, scene = load_scene(path, mtime)
    return M.run_isolation_forest(scene.epoch, contamination=contamination, seed=seed)


# --------------------------------------------------------------------------- #
# Session State for Retraining
# --------------------------------------------------------------------------- #
if "active_seed" not in st.session_state:
    st.session_state.active_seed = 42

# --------------------------------------------------------------------------- #
# Sidebar: Controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🛰️ GNSS Detection Control")

st.sidebar.header("1. Dataset")
if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}\n\n"
             "Place raw_dataset.csv in the same folder as dashboard.py and restart.")
    st.stop()

try:
    n_raw_rows, scene = load_scene(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as e:
    st.error(f"Could not prepare dataset: {e}")
    st.stop()

epoch = scene.epoch.copy()
valid_mask = epoch["Valid_fix"]
n_valid = int(valid_mask.sum())
n_nofix = int((~valid_mask).sum())

st.sidebar.success(f"Loaded: `{DATA_FILE.name}`")
st.sidebar.write(f"Raw rows: **{n_raw_rows:,}** | Epochs: **{len(epoch):,}**")
st.sidebar.write(f"Valid Fixes (≥4 sats): **{n_valid:,}** | No Fix: **{n_nofix:,}**")

st.sidebar.header("2. Model Configuration")
model_choice = st.sidebar.selectbox(
    "Classifier", M.MODEL_NAMES, index=0,
    help="Select the machine learning classifier to train."
)

label_mode_choice = st.sidebar.radio(
    "Ground Truth Labeling",
    ["Physical Threshold (Clear vs Spoofed)", "Simulated Attacks (Clean vs Spoofed vs Jammed)"],
    index=0,
    help="Physical Threshold uses GNSS distance & coordinate physics. Simulated Attacks injects synthetic attacks."
)
label_mode_internal = "threshold" if label_mode_choice.startswith("Physical") else "simulated"

threshold = st.sidebar.slider(
    "Distance Threshold (m)", min_value=10.0, max_value=5000.0, value=500.0, step=10.0,
    help="Distance deviation threshold from resting position to flag spoofing."
)
n_trees = st.sidebar.slider("Trees (Random Forest only)", 50, 300, 100, 25)

if st.sidebar.button("🔄 Retrain Model"):
    st.session_state.active_seed = int(np.random.randint(1, 999999))
    st.rerun()

st.sidebar.header("3. Model Persistence")
if st.sidebar.button(f"💾 Save Current {model_choice} Model"):
    with st.spinner(f"Saving {model_choice} ..."):
        res_current = train_model_internal(
            str(DATA_FILE), DATA_FILE.stat().st_mtime, model_choice,
            st.session_state.active_seed, threshold,
            label_mode_internal, n_trees
        )
        saved_p = M.save_model(
            M.default_model_path(APP_DIR, model_choice), res_current, scene,
            {"model": model_choice, "seed": st.session_state.active_seed, "threshold_m": threshold}
        )
    st.sidebar.success(f"Saved: models/{saved_p.name}")

saved_files = sorted(MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else []
st.sidebar.caption(f"Saved models: {', '.join(p.name for p in saved_files) if saved_files else 'none yet'}")

# --------------------------------------------------------------------------- #
# Main Model Execution (80/20 train/test split executed internally)
# --------------------------------------------------------------------------- #
res = train_model_internal(
    str(DATA_FILE), DATA_FILE.stat().st_mtime, model_choice,
    st.session_state.active_seed, threshold,
    label_mode_internal, n_trees
)

# --------------------------------------------------------------------------- #
# Header & Top KPI Metrics
# --------------------------------------------------------------------------- #
st.title("🛰️ GNSS Spoofing & Jamming Detection System")
st.caption("AI-Powered Multi-Model Detection Framework for Satellite Navigation Security")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("Raw Observations", f"{n_raw_rows:,}")
col_m2.metric("Total Epochs", f"{len(epoch):,}")
col_m3.metric("Valid Fixes (≥4 sats)", f"{n_valid:,}")
col_m4.metric("Model Accuracy", f"{res.metrics['accuracy']*100:.1f}%")
col_m5.metric("Weighted F1-Score", f"{res.metrics['f1']:.3f}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 Model Performance",
    "🛰️ Spoofed vs Clear Analysis",
    "🧪 Real-time Dataset Predictor",
    "📊 Dataset Overview",
    "📍 Position & 3D Deviation",
    "🏷️ Ground Truth & Anomaly Explorer"
])

# --------------------------------------------------------------------------- #
# TAB 1: Model Performance
# --------------------------------------------------------------------------- #
with tab1:
    st.subheader(f"Model Diagnostic & Validation Metrics ({model_choice})")
    st.write(
        f"Evaluation metrics computed on held-out test data. "
        f"The classifier predicts whether an epoch is genuine or anomalous."
    )

    t1_c1, t1_c2, t1_c3, t1_c4 = st.columns(4)
    t1_c1.metric("Accuracy", f"{res.metrics['accuracy']*100:.2f}%")
    t1_c2.metric("Precision", f"{res.metrics['precision']:.3f}")
    t1_c3.metric("Recall", f"{res.metrics['recall']:.3f}")
    t1_c4.metric("F1-Score", f"{res.metrics['f1']:.3f}")

    col_cm, col_cr = st.columns([1, 1])
    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig_cm = px.imshow(
            res.confusion, text_auto=True, color_continuous_scale="Blues",
            title=f"Confusion Matrix ({model_choice})"
        )
        fig_cm.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label")
        st.plotly_chart(fig_cm, width='stretch')

    with col_cr:
        st.markdown("#### Detailed Classification Report")
        st.dataframe(res.report.round(3), width='stretch')

    st.markdown("---")
    st.subheader("Model Benchmark Comparison")
    st.write("Performance comparison across machine learning algorithms:")

    col_bench_tbl, col_bench_fig = st.columns([1, 1])
    with col_bench_tbl:
        if not res.model_comparison.empty:
            st.dataframe(
                res.model_comparison.sort_values("Accuracy", ascending=False).round(4),
                width='stretch', hide_index=True
            )
    with col_bench_fig:
        if not res.model_comparison.empty:
            fig_cmp = px.bar(
                res.model_comparison, x="Model", y="Accuracy", color="Model",
                title="Model Accuracy Comparison", text_auto=".3f"
            )
            fig_cmp.update_layout(yaxis_range=[0.8, 1.02], showlegend=False)
            st.plotly_chart(fig_cmp, width='stretch')

    st.markdown("---")
    st.subheader("Feature Permutation Importance")
    st.write("Relative importance of GNSS features in detecting spoofing/jamming anomalies:")
    fig_imp = px.bar(
        res.feature_importance, x="Importance", y="Feature", orientation="h",
        title=f"Feature Importance ({model_choice})", color="Importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, width='stretch')

    buf = io.BytesIO()
    joblib.dump({
        "model": res.model, "model_name": res.model_name,
        "features": M.FEATURES, "classes": res.class_labels
    }, buf)
    st.download_button(
        f"📥 Download Trained {model_choice} Model (.joblib)",
        data=buf.getvalue(), file_name=f"gnss_{model_choice.lower().replace(' ', '_')}_model.joblib",
        mime="application/octet-stream"
    )

# --------------------------------------------------------------------------- #
# TAB 2: Spoofed vs Clear Analysis
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Dataset Classification: How Much is Spoofed vs Clear?")
    st.write(
        "Breakdown of genuine (Clear) versus anomalous (Spoofed) epochs across the baseline dataset:"
    )

    r_sum = res.raw_summary
    v_n = max(r_sum["valid_epochs"], 1)

    c_raw1, c_raw2, c_raw3, c_raw4 = st.columns(4)
    c_raw1.metric("🟢 Clear / Genuine Epochs", f"{r_sum['clear_count']:,}", f"{r_sum['clear_pct']:.1f}%")
    c_raw2.metric("🔴 Spoofed Epochs", f"{r_sum['spoofed_count']:,}", f"{r_sum['spoofed_pct']:.1f}%")
    if r_sum["jammed_count"] > 0:
        c_raw3.metric("🟠 Jammed Epochs", f"{r_sum['jammed_count']:,}", f"{r_sum['jammed_pct']:.1f}%")
    else:
        c_raw3.metric("⚪ No Fix (<4 sats)", f"{r_sum['no_fix_epochs']:,}", f"{r_sum['no_fix_epochs']/r_sum['total_epochs']*100:.1f}%")
    c_raw4.metric("📊 Total Epochs", f"{r_sum['total_epochs']:,}", "100%")

    st.markdown("---")
    col_donut, col_timeline = st.columns([1, 2])

    with col_donut:
        st.markdown("#### Status Distribution")
        donut_labels = ["Clear", "Spoofed"]
        donut_values = [r_sum["clear_count"], r_sum["spoofed_count"]]
        donut_colors = [STATUS_COLORS["Clear"], STATUS_COLORS["Spoofed"]]
        if r_sum["jammed_count"] > 0:
            donut_labels.append("Jammed")
            donut_values.append(r_sum["jammed_count"])
            donut_colors.append(STATUS_COLORS["Jammed"])
        if r_sum["no_fix_epochs"] > 0:
            donut_labels.append("No fix (<4 sats)")
            donut_values.append(r_sum["no_fix_epochs"])
            donut_colors.append(STATUS_COLORS["No fix (<4 sats)"])

        fig_pie = go.Figure(data=[go.Pie(
            labels=donut_labels, values=donut_values, hole=0.5,
            marker=dict(colors=donut_colors), textinfo="label+percent"
        )])
        fig_pie.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width='stretch')

    with col_timeline:
        st.markdown(f"#### Attack Probability Timeline ({model_choice})")
        valid_raw_preds = res.raw_pred_table[res.raw_pred_table["Valid_fix"]].copy()
        fig_timeline = px.scatter(
            valid_raw_preds, x="Timestamp", y="P_attack", color="ML_status",
            color_discrete_map=STATUS_COLORS,
            hover_data=["Satellite_count", "PDOP", "Distance_from_true_m"],
            title=f"Attack Probability per Epoch ({r_sum['spoofed_pct']:.1f}% Spoofed)"
        )
        fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)")
        fig_timeline.update_layout(yaxis_title="P(Spoofed / Attack)", yaxis_range=[-0.05, 1.05])
        st.plotly_chart(fig_timeline, width='stretch')

    st.markdown("#### Flagged Spoofed Epochs")
    flagged_epochs = res.raw_pred_table[res.raw_pred_table["ML_status"] == "Spoofed"].sort_values(
        "P_attack", ascending=False
    )
    if flagged_epochs.empty:
        st.success("No epoch was classified as spoofed in the raw dataset.")
    else:
        st.write(f"Displaying **{len(flagged_epochs)} flagged epochs** (highest attack probability first):")
        show_cols = [
            "Timestamp", "ML_status", "P_attack", "Distance_from_true_m",
            "Satellite_count", "PDOP", "Movement_since_previous_m", "Res_std_m"
        ]
        disp = flagged_epochs[show_cols].copy()
        nums = disp.select_dtypes(include=[np.number]).columns
        disp[nums] = disp[nums].round(3)
        st.dataframe(disp, width='stretch', hide_index=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            "📥 Download Epoch Predictions (CSV)",
            data=res.raw_pred_table.to_csv(index=False).encode("utf-8"),
            file_name="raw_dataset_epoch_predictions.csv", mime="text/csv"
        )
    with d_col2:
        raw_with_status = M.attach_status(pd.read_csv(DATA_FILE), res.raw_pred_table)
        st.download_button(
            "📥 Download Full Raw Dataset with ML Status (CSV)",
            data=raw_with_status.to_csv(index=False).encode("utf-8"),
            file_name="raw_dataset_with_ml_status.csv", mime="text/csv"
        )

# --------------------------------------------------------------------------- #
# TAB 3: Real-time Dataset Predictor
# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("🧪 Real-Time / Live Generated Dataset Predictor")
    st.write(
        "Upload your **real-time generated dataset** or any GNSS recording to immediately find out "
        "how much of it is **Spoofed** and how much is **Clear** using the trained machine learning model."
    )

    col_rt_cfg1, col_rt_cfg2 = st.columns([1, 1])
    with col_rt_cfg1:
        model_rt_source = st.radio(
            "Model to use for inference",
            ["Currently trained in-memory model (Recommended)", "Load saved model from disk (.joblib)"],
            horizontal=True
        )
    with col_rt_cfg2:
        rt_pos_mode = st.radio(
            "Receiver Resting Position for uploaded data",
            ["Auto-estimate median from file (Recommended)", "Same as raw_dataset.csv", "Enter coordinates manually"],
            horizontal=True
        )

    active_rt_model = res.model
    if model_rt_source.startswith("Load"):
        if not saved_files:
            st.warning("No saved model found on disk. Train and save one in Section 3 of the sidebar first.")
        else:
            names = [p.name for p in saved_files]
            pick = st.selectbox("Select saved .joblib model", names)
            try:
                card = M.load_model(MODEL_DIR / pick)
                active_rt_model = card["model"]
                st.success(f"Using saved model: `{card['model_name']}` (trained at {card.get('trained_at', 'unknown')})")
            except Exception as e:
                st.error(f"Could not load model: {e}")

    uploaded_file = st.file_uploader(
        "Upload Real-Time Generated GNSS Dataset (CSV)",
        type=["csv"], key="realtime_file_uploader"
    )

    if uploaded_file is None:
        st.info("👆 Upload your CSV file (e.g. real-time sensor log or simulated recording) to inspect.")
    else:
        try:
            rt_raw = pd.read_csv(uploaded_file)
            rt_cadence = M.estimate_cadence(rt_raw)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        st.write(f"Uploaded: **{len(rt_raw):,} raw rows** | Estimated interval: **{rt_cadence:.0f} s**")

        rt_true_pos = None
        if rt_pos_mode == "Same as raw_dataset.csv":
            rt_true_pos = scene.true_pos
        elif rt_pos_mode == "Enter coordinates manually":
            mc1, mc2, mc3 = st.columns(3)
            lat_in = mc1.number_input("Latitude (deg)", value=48.39006, format="%.6f")
            lon_in = mc2.number_input("Longitude (deg)", value=-123.48682, format="%.6f")
            alt_in = mc3.number_input("Altitude (m)", value=74.4)
            rt_true_pos = M.lla_to_ecef(lat_in, lon_in, alt_in)

        try:
            rt_ep, rt_summary = M.score_dataset(rt_raw, active_rt_model, true_pos=rt_true_pos)
        except Exception as e:
            st.error(f"Could not score dataset: {e}")
            st.stop()

        rt_valid_n = max(rt_summary["valid"], 1)
        st.markdown(f"### Real-Time Inference Results ({rt_summary['epochs']:,} epochs evaluated)")

        rt_c1, rt_c2, rt_c3, rt_c4 = st.columns(4)
        rt_c1.metric("🟢 Clear / Genuine", f"{rt_summary['Clear']:,}", f"{rt_summary['Clear_pct']:.1f}%")
        rt_c2.metric("🔴 Spoofed", f"{rt_summary['Spoofed']:,}", f"{rt_summary['Spoofed_pct']:.1f}%")
        if rt_summary.get("Jammed", 0) > 0:
            rt_c3.metric("🟠 Jammed", f"{rt_summary['Jammed']:,}", f"{rt_summary['Jammed_pct']:.1f}%")
        else:
            rt_c3.metric("⚪ No Fix (<4 sats)", f"{rt_summary['no_fix']:,}")
        rt_c4.metric("Median Deviation", f"{rt_summary['median_dev_m']:,.1f} m")

        col_rt_d, col_rt_t = st.columns([1, 2])
        with col_rt_d:
            st.markdown("#### Real-Time Status Breakdown")
            rt_donut_labels = ["Clear", "Spoofed"]
            rt_donut_values = [rt_summary["Clear"], rt_summary["Spoofed"]]
            rt_donut_colors = [STATUS_COLORS["Clear"], STATUS_COLORS["Spoofed"]]
            if rt_summary.get("Jammed", 0) > 0:
                rt_donut_labels.append("Jammed")
                rt_donut_values.append(rt_summary["Jammed"])
                rt_donut_colors.append(STATUS_COLORS["Jammed"])
            if rt_summary["no_fix"] > 0:
                rt_donut_labels.append("No fix (<4 sats)")
                rt_donut_values.append(rt_summary["no_fix"])
                rt_donut_colors.append(STATUS_COLORS["No fix (<4 sats)"])

            fig_rt_pie = go.Figure(data=[go.Pie(
                labels=rt_donut_labels, values=rt_donut_values, hole=0.5,
                marker=dict(colors=rt_donut_colors), textinfo="label+percent"
            )])
            fig_rt_pie.update_layout(margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig_rt_pie, width='stretch')

        with col_rt_t:
            st.markdown("#### Real-Time Attack Probability Timeline")
            rt_tl = rt_ep[rt_ep["Valid_fix"]]
            if len(rt_tl):
                fig_rt_tl = px.scatter(
                    rt_tl, x="Timestamp", y="P_attack", color="ML_status",
                    color_discrete_map=STATUS_COLORS,
                    hover_data=["Satellite_count", "PDOP", "Distance_from_true_m"],
                    title="Real-Time Epoch Attack Probability"
                )
                fig_rt_tl.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="0.5")
                st.plotly_chart(fig_rt_tl, width='stretch')

        # Flagged epochs
        rt_flagged = rt_ep[rt_ep["ML_status"].isin(["Spoofed", "Jammed"])].sort_values("P_attack", ascending=False)
        if rt_flagged.empty:
            st.success("No epoch in this real-time dataset was detected as spoofed or jammed! All data appears Clear.")
        else:
            st.warning(f"⚠️ Detected **{len(rt_flagged)} spoofed/jammed epochs**:")
            rt_disp = rt_flagged[["Timestamp", "ML_status", "P_attack", "Distance_from_true_m",
                                   "Satellite_count", "PDOP", "Movement_since_previous_m", "Res_std_m"]].copy()
            rt_nums = rt_disp.select_dtypes(include=[np.number]).columns
            rt_disp[rt_nums] = rt_disp[rt_nums].round(3)
            st.dataframe(rt_disp, width='stretch', hide_index=True)

        # Accuracy check if file has ground truth
        st.markdown("---")
        st.markdown("#### Optional: Ground-Truth Verification (if file contains labels)")
        label_col = st.selectbox(
            "Select column containing actual ground truth label (optional)",
            ["(None - dataset has no labels)"] + list(rt_raw.columns),
            key="rt_label_col"
        )
        if not label_col.startswith("(None"):
            try:
                ev = M.evaluate_labeled(rt_ep, M.epoch_labels(rt_raw, label_col))
                eva1, eva2, eva3, eva4 = st.columns(4)
                eva1.metric("Test Accuracy", f"{ev['accuracy']*100:.1f}%")
                eva2.metric("Attack Precision", f"{ev['attack_vs_clean']['precision']:.3f}")
                eva3.metric("Attack Recall", f"{ev['attack_vs_clean']['recall']:.3f}")
                eva4.metric("Attack F1", f"{ev['attack_vs_clean']['f1']:.3f}")
                st.write("Confusion matrix:")
                st.dataframe(ev["confusion"], width='stretch')
            except Exception as e:
                st.error(f"Could not compute ground truth metrics: {e}")

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Download Scored Real-Time Epochs (CSV)",
                data=rt_ep.to_csv(index=False).encode("utf-8"),
                file_name="realtime_epoch_results.csv", mime="text/csv"
            )
        with dl2:
            st.download_button(
                "📥 Download Original Rows + ML Status (CSV)",
                data=M.attach_status(rt_raw, rt_ep).to_csv(index=False).encode("utf-8"),
                file_name="realtime_with_status.csv", mime="text/csv"
            )

# --------------------------------------------------------------------------- #
# TAB 4: Dataset Overview
# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Dataset Structure & Raw Epochs")
    st.write(
        "Each timestamp is grouped into an epoch. Coordinates `Xr (m)`, `Yr (m)`, `Zr (m)` represent "
        "the receiver's position in ECEF meters. Satellites and pseudoranges determine the fix quality."
    )
    show_cols = ["Timestamp", "Satellite_count", "PDOP", "Distance_from_true_m",
                 "Movement_since_previous_m", "Res_std_m", "Valid_fix"]
    st.dataframe(epoch[show_cols].head(30), width='stretch')

    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        st.subheader("Satellite Visibility Distribution")
        fig_sats = px.histogram(epoch, x="Satellite_count", nbins=15, title="Satellites per Epoch")
        st.plotly_chart(fig_sats, width='stretch')

    with col_ov2:
        st.subheader("PDOP Distribution (Geometry Dilution)")
        fig_pdop = px.histogram(epoch[epoch["Valid_fix"]], x="PDOP", nbins=30, title="PDOP (Lower is better)")
        st.plotly_chart(fig_pdop, width='stretch')

    st.write(f"Receiver Resting Position (Median of valid epochs): "
             f"**X = {scene.true_pos[0]:,.2f} m, Y = {scene.true_pos[1]:,.2f} m, Z = {scene.true_pos[2]:,.2f} m**")

# --------------------------------------------------------------------------- #
# TAB 5: Position & 3D Deviation
# --------------------------------------------------------------------------- #
with tab5:
    st.subheader("3D Position Deviation from Resting Position")
    st.write("Measures spatial displacement between each epoch's estimated coordinates and the baseline station position.")

    valid_epochs = epoch[epoch["Valid_fix"]].copy()
    fig_dev = px.line(
        valid_epochs, x="Timestamp", y="Distance_from_true_m",
        title="3D Distance from Station True Position over Time (m)"
    )
    fig_dev.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold}m)")
    st.plotly_chart(fig_dev, width='stretch')

    c_dev1, c_dev2 = st.columns(2)
    with c_dev1:
        st.subheader("Vertical Deviation (|Up|)")
        fig_v = px.line(valid_epochs, x="Timestamp", y="Vertical_dev_m", title="Vertical (Height) Deviation (m)")
        st.plotly_chart(fig_v, width='stretch')
    with c_dev2:
        st.subheader("Inter-Epoch Step Jump")
        fig_m = px.line(valid_epochs, x="Timestamp", y="Movement_since_previous_m", title="Movement Between Fixes (m)")
        st.plotly_chart(fig_m, width='stretch')

# --------------------------------------------------------------------------- #
# TAB 6: Ground Truth & Anomaly Explorer
# --------------------------------------------------------------------------- #
with tab6:
    st.subheader("Ground Truth Detector & Isolation Forest Cross-Check")
    st.write(
        "Threshold rule: Flag an epoch as spoofed if 3D deviation > threshold or altitude/latitude physics violated."
    )

    iso = run_isolation(str(DATA_FILE), DATA_FILE.stat().st_mtime, 0.03, 42)
    merged_iso = valid_epochs.merge(iso, on="Timestamp", how="left")

    fig_iso = px.scatter(
        merged_iso, x="Timestamp", y="Anomaly_score", color="IF_anomaly",
        title="Isolation Forest Anomaly Score (Higher = Outlier)"
    )
    st.plotly_chart(fig_iso, width='stretch')

    st.download_button(
        "📥 Download Baseline Rule-Labeled Dataset",
        data=epoch.to_csv(index=False).encode("utf-8"),
        file_name="epoch_with_baseline_labels.csv", mime="text/csv"
    )

st.divider()
st.caption(
    "Minor Project System: Built with Python, Streamlit, and Scikit-Learn."
)