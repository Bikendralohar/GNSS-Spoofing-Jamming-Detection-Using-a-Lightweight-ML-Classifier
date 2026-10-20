import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_score, recall_score)

import ml_model as M

st.set_page_config(page_title="GNSS Spoofing Detection Dashboard", page_icon="🛰️", layout="wide")

# The dataset is fixed: raw_dataset.csv must sit in the SAME folder as dashboard.py.
DATA_FILE = Path(__file__).resolve().parent / "raw_dataset.csv"

st.title("🛰️ GNSS Spoofing & Jamming Detection Dashboard")
st.caption("Workflow: epoch construction → true position → deviation analysis → threshold detector → "
           "features → Random Forest / Isolation Forest (Clean · Spoofed · Jammed)")


# --------------------------------------------------------------------------- #
# Cached loaders / trainers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading raw_dataset.csv ...")
def load_scene(path: str, mtime: float):
    raw = pd.read_csv(path)
    return len(raw), M.build_scene(raw)


@st.cache_resource(show_spinner="Training Random Forest with leave-one-day-out validation (≈10 s) ...")
def train_supervised(path: str, mtime: float, n_trees: int, attack_frac: float, seed: int):
    _, scene = load_scene(path, mtime)
    return M.run_supervised(scene, n_trees=n_trees, attack_frac=attack_frac, seed=seed)


@st.cache_data(show_spinner="Running Isolation Forest ...")
def run_isolation(path: str, mtime: float, contamination: float, seed: int):
    _, scene = load_scene(path, mtime)
    return M.run_isolation_forest(scene.epoch, contamination=contamination, seed=seed)


# --------------------------------------------------------------------------- #
# Sidebar: data (fixed) + detector + ML settings
# --------------------------------------------------------------------------- #
st.sidebar.header("1. Data")
if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}\n\n"
             "Place raw_dataset.csv in the same folder as dashboard.py and restart.")
    st.stop()

try:
    n_raw_rows, scene = load_scene(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except Exception as e:
    st.error(f"Could not prepare the dataset: {e}")
    st.stop()

epoch = scene.epoch.copy()
true_pos = pd.Series(scene.true_pos, index=M.XYZ)
valid = epoch["Valid_fix"]
n_nofix = int((~valid).sum())

st.sidebar.success(f"Dataset: {DATA_FILE.name} (fixed)")
st.sidebar.write(f"Raw rows: **{n_raw_rows:,}**")
st.sidebar.write(f"Epochs: **{len(epoch):,}**  (≥4 satellites: {int(valid.sum()):,})")

st.sidebar.header("2. Threshold detector")
threshold = st.sidebar.slider(
    "Distance threshold (m)", min_value=10.0, max_value=10000.0, value=500.0, step=10.0,
    help="Baseline threshold. Final value should be justified from clean data / validation.")
exclude_nofix = st.sidebar.checkbox(
    "Treat epochs with <4 satellites as 'No fix' (not spoofing)", value=True,
    help="With fewer than 4 satellites the position cannot be solved, so the huge deviation "
         "(thousands of km) is a geometry problem, not necessarily an attack.")
use_altitude = st.sidebar.checkbox("Flag impossible altitude", value=True)
use_latlon = st.sidebar.checkbox("Flag impossible latitude/longitude", value=True)

st.sidebar.header("3. Machine learning")
n_trees = st.sidebar.slider("Trees (Random Forest)", 50, 500, 200, 50)
attack_frac = st.sidebar.slider("Simulated attack share of epochs", 0.20, 0.50, 0.35, 0.05,
                                help="Fraction of epochs turned into simulated attacks for training/testing.")
contamination = st.sidebar.slider("Isolation Forest: expected outlier share", 0.01, 0.10, 0.03, 0.01)
seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42))

# Threshold rule
epoch["Threshold_flag"] = epoch["Distance_from_true_m"] > threshold
if exclude_nofix:
    epoch["Threshold_flag"] &= epoch["Valid_fix"]
epoch["Rule_flag"] = epoch["Threshold_flag"]
if use_altitude:
    epoch["Rule_flag"] |= epoch["Impossible_altitude"]
if use_latlon:
    epoch["Rule_flag"] |= epoch["Impossible_latlon"]
epoch["Working_label"] = epoch["Rule_flag"].astype(int)

# KPIs
total = len(epoch)
flagged = int(epoch["Rule_flag"].sum())
flag_pct = flagged / total * 100 if total else 0
vd = epoch.loc[valid, "Distance_from_true_m"]
median_dev = float(vd.median())
max_dev = float(vd.max())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Epochs", f"{total:,}")
c2.metric("No fix (<4 sats)", f"{n_nofix:,}", f"{n_nofix / total * 100:.1f}%", delta_color="off")
c3.metric("Rule-flagged", f"{flagged:,}", f"{flag_pct:.1f}%", delta_color="off")
c4.metric("Median deviation", f"{median_dev:,.1f} m")
c5.metric("Maximum deviation", f"{max_dev:,.0f} m")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📍 Position & Deviation", "🚨 Detector", "🏷️ Labels", "🤖 ML Detection"])

# --------------------------------------------------------------------------- #
with tab1:
    st.subheader("Dataset overview")
    st.write(
        "The dashboard builds one row per timestamp and uses the repeated receiver coordinates "
        "Xr/Yr/Zr to measure deviation from the median resting position (median of epochs that "
        "have ≥4 satellites).")
    show_cols = ["Timestamp", "Satellite_count", "PDOP", "Distance_from_true_m",
                 "Movement_since_previous_m", "Res_std_m", "Valid_fix"]
    st.dataframe(epoch[show_cols].head(20), width="stretch")

    st.subheader("Satellite visibility per epoch")
    fig = px.histogram(epoch, x="Satellite_count", nbins=max(5, int(epoch["Satellite_count"].max()) + 1))
    fig.update_layout(xaxis_title="Visible satellites", yaxis_title="Epoch count")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Epochs without a position fix, by hour of day")
    hourly = (epoch.assign(Hour=epoch["Timestamp"].dt.hour, Day=epoch["Day"])
                   .groupby(["Day", "Hour"])["Valid_fix"].apply(lambda s: float((~s).mean()))
                   .reset_index(name="Share_no_fix"))
    fig = px.line(hourly, x="Hour", y="Share_no_fix", color="Day", markers=True)
    fig.update_layout(yaxis_title="Share of epochs with <4 satellites")
    st.plotly_chart(fig, width="stretch")
    st.caption("If the same hours lose their fix every day, the cause is satellite visibility / geometry "
               "(the constellation repeats daily) rather than an attack.")

# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("True resting position")
    st.write({k: float(v) for k, v in true_pos.items()})

    plot_df = epoch[valid].copy()
    plot_df["Status"] = np.where(plot_df["Rule_flag"], "Flagged", "Normal")
    fig = px.line(plot_df, x="Timestamp", y="Distance_from_true_m",
                  title="3-D receiver-position deviation over time (epochs with ≥4 satellites)",
                  labels={"Distance_from_true_m": "Distance from median true position (m)"},
                  log_y=True)
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"Threshold = {threshold:g} m")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{n_nofix:,} epochs with <4 satellites are not drawn: their position is not solvable.")

    if {"Latitude_deg", "Longitude_deg"}.issubset(epoch.columns):
        map_df = epoch[valid & (epoch["Distance_from_true_m"] <= 50_000)].dropna(
            subset=["Latitude_deg", "Longitude_deg"]).copy()
        map_df["Rule_flag"] = map_df["Rule_flag"].astype(str)
        fig = px.scatter_map(
            map_df, lat="Latitude_deg", lon="Longitude_deg",
            color="Rule_flag", size="Satellite_count",
            hover_data=["Timestamp", "Distance_from_true_m"], zoom=12)
        fig.update_layout(map_style="open-street-map", height=550)
        st.plotly_chart(fig, width="stretch")
        st.caption("Map shows epochs within 50 km of the resting position.")

# --------------------------------------------------------------------------- #
with tab3:
    st.subheader("Threshold-based baseline detector")
    st.write("Baseline: flag a reading when it is farther than X metres from the true position, "
             "with optional impossible-value rules. Epochs with <4 satellites can be reported as 'No fix' "
             "(sidebar) instead of being counted as spoofing.")

    rule_counts = pd.DataFrame({
        "Class": ["Normal", "Flagged"],
        "Count": [int((epoch["Working_label"] == 0).sum()), int((epoch["Working_label"] == 1).sum())]})
    st.plotly_chart(px.bar(rule_counts, x="Class", y="Count", title="Baseline detector result"),
                    width="stretch")

    flagged_rows = epoch[epoch["Rule_flag"]].sort_values("Distance_from_true_m", ascending=False)
    st.dataframe(flagged_rows[["Timestamp", "Satellite_count", "PDOP", "Distance_from_true_m",
                               "Movement_since_previous_m", "Res_std_m", "Valid_fix"]].head(100),
                 width="stretch")

    st.download_button("Download baseline-labeled epoch table",
                       data=epoch.to_csv(index=False).encode("utf-8"),
                       file_name="epoch_with_baseline_labels.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
with tab4:
    st.subheader("Manual clean/spoofed labeling")
    st.warning(
        "Working_label is only a temporary rule-based label. Do not present it as ground truth in the "
        "final report. If you have real knowledge of which periods were attacked, label them here and "
        "use the download in the ML tab ('My manual labels').")
    label_view = epoch[["Timestamp", "Distance_from_true_m", "Movement_since_previous_m",
                        "Clock_bias_change_s", "Satellite_count", "Impossible_altitude",
                        "Impossible_latlon", "Working_label"]].copy()
    label_view["Manual_label"] = label_view["Working_label"]

    edited = st.data_editor(
        label_view, width="stretch", height=500,
        column_config={"Manual_label": st.column_config.SelectboxColumn(
            "Manual label (0=clean, 1=spoofed)", options=[0, 1], required=True)},
        disabled=[c for c in label_view.columns if c != "Manual_label"], hide_index=True)

    labeled = epoch.copy()
    labeled["Manual_label"] = edited["Manual_label"].astype(int).values
    st.write("Manual labels:", labeled["Manual_label"].value_counts().rename(index={0: "Clean", 1: "Spoofed"}))
    st.download_button("Download manually labeled dataset",
                       data=labeled.to_csv(index=False).encode("utf-8"),
                       file_name="epoch_manually_labeled.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
STATUS_COLORS = {"Clean": "#2e9e5b", "Spoofed": "#d62728", "Jammed": "#ff7f0e", "No fix (<4 sats)": "#9e9e9e"}

with tab5:
    st.subheader("Machine-learning detection: Clean · Spoofed · Jammed")
    source = st.radio(
        "Training labels",
        ["Simulated spoofing & jamming (recommended)", "My manual labels (upload CSV)"],
        horizontal=True)

    # ======================================================================= #
    if source.startswith("Simulated"):
        st.info(
            "**Why simulated attacks?** raw_dataset.csv contains no spoofed/jammed labels. A model trained on the "
            "threshold rule's own output would just re-learn the rule (≈100 % accuracy, meaningless). Instead, "
            "attacks are injected into the real epochs (spoofing = coherent position push/drift, jamming = lost "
            "satellites + noisy pseudoranges), and the Random Forest is validated with **leave-one-day-out** "
            "cross-validation: it is always tested on a day it never saw. The scores below measure detection of "
            "*these simulated attacks*, not of real ones.")

        res = train_supervised(str(DATA_FILE), DATA_FILE.stat().st_mtime, n_trees, attack_frac, seed)

        st.markdown("#### 1 · How well does the Random Forest detect simulated attacks?")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{res.metrics['accuracy']:.3f}")
        m2.metric("Macro precision", f"{res.metrics['macro_precision']:.3f}")
        m3.metric("Macro recall", f"{res.metrics['macro_recall']:.3f}")
        m4.metric("Macro F1", f"{res.metrics['macro_f1']:.3f}")

        left, right = st.columns(2)
        with left:
            labels = list(M.CLASS_NAMES.values())
            cm_df = pd.DataFrame(res.confusion, index=[f"Actual {l}" for l in labels],
                                 columns=[f"Pred {l}" for l in labels])
            fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                            title="Confusion matrix (rows = actual, columns = predicted)")
            st.plotly_chart(fig, width="stretch")
        with right:
            st.write("Per-class scores")
            st.dataframe(res.report.round(3), width="stretch")
            st.write("Accuracy per held-out day")
            st.dataframe(pd.DataFrame(res.folds).round(3), width="stretch", hide_index=True)

        st.markdown("#### 2 · Does ML beat the threshold rule? (attack vs clean, same test epochs)")
        base_cmp = M.baseline_threshold_metrics(res.sim_test, threshold)
        st.dataframe(pd.DataFrame(base_cmp).T.round(3), width="stretch")
        st.caption(f"Threshold rule uses the sidebar threshold ({threshold:g} m).")

        st.markdown("#### 3 · Detection limit: how strong must an attack be?")
        st.dataframe(res.by_severity.round(3), width="stretch", hide_index=True)
        st.caption("Weak attacks close to the normal noise (~100 m) are hard to detect; this table "
                   "shows exactly where that limit is. Use it in your report.")

        st.markdown("#### 4 · Which features matter?")
        fig = px.bar(res.feature_importance, x="Importance", y="Feature", orientation="h",
                     title="Random Forest feature importance (final model)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

        buf = io.BytesIO()
        joblib.dump({"model": res.model, "features": M.FEATURES, "classes": M.CLASS_NAMES}, buf)
        st.download_button("Download trained model (.joblib)", data=buf.getvalue(),
                           file_name="gnss_rf_spoof_jam.joblib", mime="application/octet-stream")

        # ------------------------------------------------------------------- #
        st.markdown("---")
        st.markdown("#### 5 · Result on the REAL data")
        st.caption("Each real epoch is classified by a model trained on the *other* four days "
                   "(no in-sample memorisation).")

        keep = ["Timestamp", "Pred", "P_Clean", "P_Spoofed", "P_Jammed"]
        results = epoch.merge(res.real_pred[keep], on="Timestamp", how="left")
        results["ML_status"] = np.where(
            ~results["Valid_fix"], "No fix (<4 sats)",
            results["Pred"].map(lambda v: M.CLASS_NAMES.get(int(v)) if pd.notna(v) else None))
        results["P_attack"] = 1.0 - results["P_Clean"]

        iso = run_isolation(str(DATA_FILE), DATA_FILE.stat().st_mtime, contamination, seed)
        results = results.merge(iso, on="Timestamp", how="left")
        results["IF_anomaly"] = results["IF_anomaly"].fillna(False).astype(bool)

        counts = results["ML_status"].value_counts().reindex(list(STATUS_COLORS)).fillna(0).astype(int)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clean", f"{counts['Clean']:,}")
        k2.metric("Spoofed", f"{counts['Spoofed']:,}")
        k3.metric("Jammed", f"{counts['Jammed']:,}")
        k4.metric("No fix (<4 sats)", f"{counts['No fix (<4 sats)']:,}")

        tl = results[results["Valid_fix"]]
        fig = px.scatter(tl, x="Timestamp", y="P_attack", color="ML_status",
                         color_discrete_map=STATUS_COLORS,
                         hover_data=["Satellite_count", "PDOP", "Distance_from_true_m"],
                         title="Probability of attack per epoch (Random Forest)")
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="0.5")
        fig.update_layout(yaxis_title="P(spoofed or jammed)")
        st.plotly_chart(fig, width="stretch")

        st.write("Epochs the Random Forest flags (sorted by attack probability)")
        flagged_ml = results[results["ML_status"].isin(["Spoofed", "Jammed"])].sort_values(
            "P_attack", ascending=False)
        if flagged_ml.empty:
            st.success("No epoch was classified as spoofed or jammed.")
        else:
            st.dataframe(flagged_ml[["Timestamp", "ML_status", "P_attack", "Satellite_count", "PDOP",
                                     "Distance_from_true_m", "Movement_since_previous_m",
                                     "Res_std_m"]].round(3), width="stretch", hide_index=True)
        st.caption("Low P(attack) values near 0.5–0.9 on real data mean 'borderline': the model is not "
                   "confident. A PDOP above ~6 usually explains a poor fix without any attack.")

        st.markdown("##### Unsupervised cross-check: Isolation Forest (needs no labels)")
        fig = px.scatter(tl.merge(iso[["Timestamp"]], on="Timestamp"), x="Timestamp", y="Anomaly_score",
                         color="IF_anomaly", title="Isolation Forest anomaly score (higher = more unusual)")
        st.plotly_chart(fig, width="stretch")
        agree = pd.crosstab(tl["ML_status"].isin(["Spoofed", "Jammed"]).rename("RF flags attack"),
                            tl["IF_anomaly"].rename("Isolation Forest outlier"))
        st.write("Agreement between the two detectors")
        st.dataframe(agree, width="stretch")
        st.caption("Isolation Forest always flags the share you set in the sidebar, even on perfectly clean data. "
                   "Epochs flagged by BOTH detectors are the most credible candidates.")

        st.download_button("Download ML results for every epoch",
                           data=results.drop(columns=["Pred"]).to_csv(index=False).encode("utf-8"),
                           file_name="epoch_ml_results.csv", mime="text/csv")

    # ======================================================================= #
    else:
        st.write("Upload `epoch_manually_labeled.csv` (downloaded from the **Labels** tab after you set "
                 "Manual_label yourself). Only epochs with ≥4 satellites are used. This trains a binary "
                 "clean/spoofed Random Forest.")
        lab_file = st.file_uploader("Upload epoch_manually_labeled.csv", type=["csv"], key="labels")
        if lab_file is None:
            st.info("Upload your manually labeled epoch file.")
        else:
            lab_df = pd.read_csv(lab_file)
            if "Manual_label" not in lab_df.columns:
                st.error("The uploaded file must contain a Manual_label column (0=clean, 1=spoofed).")
            elif not set(M.FEATURES).issubset(lab_df.columns):
                st.error("This file lacks the new feature columns. Download the labeled file again "
                         "from the Labels tab of this version of the dashboard.")
            else:
                test_size = st.slider("Test-set fraction", 0.2, 0.4, 0.3, 0.05)
                try:
                    rf, y_test, pred = M.train_on_manual_labels(lab_df, test_size, seed, n_trees)
                except ValueError as e:
                    st.error(str(e))
                else:
                    a, b, c = st.columns(3)
                    a.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
                    b.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
                    c.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
                    cm = confusion_matrix(y_test, pred, labels=[0, 1])
                    st.write("Confusion matrix — rows = actual, columns = predicted")
                    st.dataframe(pd.DataFrame(cm, index=["Actual clean", "Actual spoofed"],
                                              columns=["Pred clean", "Pred spoofed"]))
                    imp = pd.DataFrame({"Feature": M.FEATURES, "Importance": rf.feature_importances_})
                    st.plotly_chart(px.bar(imp.sort_values("Importance"), x="Importance", y="Feature",
                                           orientation="h", title="Random Forest feature importance"),
                                    width="stretch")
                    st.text(classification_report(y_test, pred, target_names=["Clean", "Spoofed"],
                                                  zero_division=0))
                    st.caption("Random train/test split: neighbouring epochs are correlated, so these scores "
                               "are optimistic. Prefer the simulated-attack mode for an honest estimate.")

st.divider()
st.caption(
    "Data notes: receiver position uses Xr/Yr/Zr (X/Y/Z are satellite coordinates). Distances are computed "
    "in metres before any scaling. Epochs with <4 satellites cannot produce a position fix and are reported "
    "as 'No fix' rather than as attacks.")
