"""
ml_model.py - GNSS spoofing / jamming detection (feature engineering + ML)

Why this module exists
----------------------
raw_dataset.csv has NO ground-truth labels (nobody told us which epochs are
spoofed or jammed). Training a classifier on labels produced by the threshold
rule would only teach it the rule back (accuracy ~100 %, scientifically
meaningless). So this module uses two complementary approaches:

1. Supervised  - Gradient Boosting (default) or Random Forest (Clean / Spoofed / Jammed)
   Labels come from *simulated attacks* injected into the real, clean-looking
   epochs (see `simulate_attacks`). Evaluated with leave-one-day-out
   cross-validation, so the model is always tested on a day it never saw.

2. Unsupervised - Isolation Forest
   Needs no labels at all. Scores how unusual each real epoch is.

All features are computed by the SAME function for real and simulated epochs,
so the model cannot cheat by recognising "how the data was produced".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

C_LIGHT = 299_792_458.0  # m/s
MIN_SATS = 4             # minimum satellites for a position fix (3 coords + clock)

CLASS_NAMES = {0: "Clean", 1: "Spoofed", 2: "Jammed"}
BINARY_CLASS_NAMES = {0: "Clear", 1: "Spoofed"}

# Features fed to the ML models (all computed per epoch).
FEATURES = [
    "Satellite_count",           # how many satellites contributed
    "PDOP",                      # geometry quality (explains "bad but innocent" fixes)
    "Distance_from_true_m",      # 3-D distance from the known resting position
    "Vertical_dev_m",            # |height error| at the resting position
    "Movement_since_previous_m", # jump between consecutive fixes
    "Clock_bias_change_s",       # |change| of receiver clock bias
    "Res_median_m",              # common-mode pseudorange residual
    "Res_std_m",                 # spread of residuals across satellites (RAIM-like)
    "Res_max_dev_m",             # worst single-satellite residual
]

RAW_RENAME = {
    "X (km)": "X_km", "Y (km)": "Y_km", "Z (km)": "Z_km",
    "Clock (s)": "Clock_s", "Xr (m)": "Xr_m", "Yr (m)": "Yr_m",
    "Zr (m)": "Zr_m", "br (s)": "br_s",
    "Latitude (deg)": "Latitude_deg", "Longitude (deg)": "Longitude_deg",
    "Altitude (m)": "Altitude_m",
}
REQUIRED = ["Satellite", "X_km", "Y_km", "Z_km", "Clock_s", "P_correction",
            "Xr_m", "Yr_m", "Zr_m", "br_s"]
XYZ = ["Xr_m", "Yr_m", "Zr_m"]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _ecef_to_lat_lon(p):
    """ECEF (m) -> geodetic latitude / longitude in radians (WGS-84)."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    x, y, z = p
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - e2))
    for _ in range(6):
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * n * np.sin(lat), r)
    return lat, lon


def _enu_axes(true_pos):
    lat, lon = _ecef_to_lat_lon(true_pos)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


def pdop(u):
    """Position dilution of precision from unit line-of-sight vectors (n x 3)."""
    if len(u) < MIN_SATS:
        return np.nan
    g = np.hstack([-u, np.ones((len(u), 1))])
    try:
        q = np.linalg.inv(g.T @ g)
    except np.linalg.LinAlgError:
        return 100.0
    return float(min(np.sqrt(max(q[0, 0] + q[1, 1] + q[2, 2], 0.0)), 100.0))


# --------------------------------------------------------------------------- #
# Scene = everything we know about the dataset, built once
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    epoch: pd.DataFrame            # one row per timestamp (features + legacy columns)
    u_list: list                   # per epoch: (n_sat, 3) unit line-of-sight vectors
    res_list: list                 # per epoch: (n_sat,) pseudorange residuals [m]
    true_pos: np.ndarray           # resting position (ECEF, m)
    east: np.ndarray = field(repr=False, default=None)
    north: np.ndarray = field(repr=False, default=None)
    up: np.ndarray = field(repr=False, default=None)
    cadence_s: float = 300.0

    @property
    def max_gap_s(self):
        return 1.5 * self.cadence_s


@dataclass
class Base:
    """Valid-fix epochs only; the raw material the attack simulator works on."""
    t: np.ndarray
    pos: np.ndarray
    br: np.ndarray
    u_list: list
    res_list: list
    day: np.ndarray
    timestamp: np.ndarray

    def select(self, mask):
        idx = np.flatnonzero(mask)
        return Base(self.t[idx], self.pos[idx], self.br[idx],
                    [self.u_list[i] for i in idx], [self.res_list[i] for i in idx],
                    self.day[idx], self.timestamp[idx])


def _timestamps(df):
    cols = set(df.columns)
    if "Datetime" in cols:
        return pd.to_datetime(df["Datetime"], errors="coerce")
    if {"DATE", "Time"} <= cols:
        return pd.to_datetime(df["DATE"].astype(str) + " " + df["Time"].astype(str),
                              dayfirst=True, errors="coerce")
    if "Timestamp" in cols:
        return pd.to_datetime(df["Timestamp"], errors="coerce")
    raise ValueError("No DATE/Time, Datetime, or Timestamp column was found.")


def compute_features(scene, t, pos, br, u_list, res_list):
    """Epoch-level features. Used identically for real and simulated epochs."""
    n_ep = len(t)
    n = np.array([len(r) for r in res_list])
    valid = n >= MIN_SATS

    dvec = pos - scene.true_pos
    dist = np.linalg.norm(dvec, axis=1)
    vert = np.abs(dvec @ scene.up)

    step = np.full(n_ep, np.nan)
    clk = np.full(n_ep, np.nan)
    if n_ep > 1:
        adjacent = valid[1:] & valid[:-1] & (np.diff(t) <= scene.max_gap_s)
        step[1:][adjacent] = np.linalg.norm(np.diff(pos, axis=0), axis=1)[adjacent]
        clk[1:][adjacent] = np.abs(np.diff(br))[adjacent]

    r_med = np.full(n_ep, np.nan)
    r_std = np.full(n_ep, np.nan)
    r_max = np.full(n_ep, np.nan)
    for i, r in enumerate(res_list):
        if len(r) >= 2:
            med = np.median(r)
            r_med[i] = med
            r_std[i] = r.std(ddof=1)
            r_max[i] = np.max(np.abs(r - med))

    return pd.DataFrame({
        "Satellite_count": n,
        "PDOP": [pdop(u) for u in u_list],
        "Distance_from_true_m": dist,
        "Vertical_dev_m": vert,
        "Movement_since_previous_m": np.nan_to_num(step, nan=0.0),
        "Clock_bias_change_s": np.nan_to_num(clk, nan=0.0),
        "Res_median_m": r_med,
        "Res_std_m": r_std,
        "Res_max_dev_m": r_max,
        "Valid_fix": valid,
    })


def build_scene(raw: pd.DataFrame, true_pos=None) -> Scene:
    """raw rows -> Scene (one epoch per timestamp + per-satellite residuals).

    true_pos: known resting position (ECEF, m). If None, the median receiver position
    of epochs with >= 4 satellites is used.
    """
    df = raw.rename(columns=RAW_RENAME).copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df["Timestamp"] = _timestamps(df)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp", kind="stable").reset_index(drop=True)

    starts = np.flatnonzero(df["Timestamp"].ne(df["Timestamp"].shift()).to_numpy())
    ends = np.append(starts[1:], len(df))

    keep = XYZ + ["br_s"] + [c for c in ("Latitude_deg", "Longitude_deg", "Altitude_m") if c in df.columns]
    epoch = df.iloc[starts][["Timestamp"] + keep].reset_index(drop=True)

    # Receiver coordinates (Xr/Yr/Zr) are repeated on every satellite row.
    # X/Y/Z are SATELLITE coordinates and must not be used as receiver position.
    n_sat = (ends - starts)
    valid = n_sat >= MIN_SATS
    if true_pos is None:
        if not valid.any():
            raise ValueError("No epoch has >= 4 satellites; cannot establish a resting position.")
        true_pos = epoch.loc[valid, XYZ].median().to_numpy()   # robust: ignore unsolvable epochs
    else:
        true_pos = np.asarray(true_pos, dtype=float)
    east, north, up = _enu_axes(true_pos)

    sat = df[["X_km", "Y_km", "Z_km"]].to_numpy() * 1000.0
    vec = sat - true_pos
    rng_m = np.linalg.norm(vec, axis=1)
    u_all = vec / rng_m[:, None]
    # residual = measured pseudorange - geometric range to the TRUE position
    #            (+ satellite clock term); a consistent signal gives a small band.
    res_all = df["P_correction"].to_numpy() - rng_m + C_LIGHT * df["Clock_s"].to_numpy()
    u_list = [u_all[s:e] for s, e in zip(starts, ends)]
    res_list = [res_all[s:e] for s, e in zip(starts, ends)]

    t = (epoch["Timestamp"] - epoch["Timestamp"].min()).dt.total_seconds().to_numpy()
    cadence = float(np.median(np.diff(t))) if len(t) > 1 else 300.0

    scene = Scene(epoch=epoch, u_list=u_list, res_list=res_list, true_pos=true_pos,
                  east=east, north=north, up=up, cadence_s=cadence)

    feats = compute_features(scene, t, epoch[XYZ].to_numpy(), epoch["br_s"].to_numpy(),
                             u_list, res_list)
    epoch = pd.concat([epoch, feats], axis=1)
    epoch["Day"] = epoch["Timestamp"].dt.strftime("%Y-%m-%d")
    epoch["Impossible_altitude"] = (
        (epoch["Altitude_m"] < -1000) | (epoch["Altitude_m"] > 100000)
        if "Altitude_m" in epoch.columns else False)
    epoch["Impossible_latlon"] = (
        (epoch["Latitude_deg"].abs() > 90) | (epoch["Longitude_deg"].abs() > 180)
        if {"Latitude_deg", "Longitude_deg"} <= set(epoch.columns) else False)
    scene.epoch = epoch
    return scene


def valid_base(scene: Scene) -> Base:
    ep = scene.epoch
    idx = np.flatnonzero(ep["Valid_fix"].to_numpy())
    t = (ep["Timestamp"] - ep["Timestamp"].min()).dt.total_seconds().to_numpy()
    return Base(
        t=t[idx], pos=ep[XYZ].to_numpy()[idx], br=ep["br_s"].to_numpy()[idx],
        u_list=[scene.u_list[i] for i in idx], res_list=[scene.res_list[i] for i in idx],
        day=ep["Day"].to_numpy()[idx], timestamp=ep["Timestamp"].to_numpy()[idx])


# --------------------------------------------------------------------------- #
# Attack simulation
# --------------------------------------------------------------------------- #
def simulate_attacks(scene: Scene, base: Base, seed: int, attack_frac: float = 0.35,
                     win_len=(6, 36)) -> pd.DataFrame:
    """
    Inject simulated attacks into windows of real epochs and recompute features.

    Spoofed  - coherent bias: the position is pushed away from the truth, either
               suddenly ("jump") or slowly ("drift"). Pseudoranges stay mutually
               consistent with the FALSE position, so residuals against the true
               position shift by -u.offset (different for each satellite).
               Optionally a receiver-clock jump.
    Jammed   - noise: some satellites are lost and the remaining pseudoranges get
               random noise; position error is erratic (scaled by PDOP).

    Label: 0 clean, 1 spoofed, 2 jammed.
    NOTE: the clean class assumes the real data contains no attack. Metrics
    therefore measure detection of *these simulated* attacks, not real ones.
    """
    rng = np.random.default_rng(seed)
    n_ep = len(base.t)
    pos = base.pos.copy()
    br = base.br.copy()
    u_list = list(base.u_list)
    res_list = [r.copy() for r in base.res_list]

    label = np.zeros(n_ep, dtype=int)
    atype = np.full(n_ep, "clean", dtype=object)
    magnitude = np.zeros(n_ep)

    target = int(attack_frac * n_ep)
    placed, attempts = 0, 0
    lo, hi = win_len
    while placed < target and attempts < 2000 and n_ep > hi + 2:
        attempts += 1
        length = int(rng.integers(lo, hi + 1))
        s = int(rng.integers(1, n_ep - length - 1))
        w = np.arange(s, s + length)
        if np.any(np.diff(base.t[s - 1:s + length + 1]) > scene.max_gap_s):
            continue                      # window must be contiguous in time
        if label[s - 1:s + length + 1].any():
            continue                      # keep a clean margin between windows

        if rng.random() < 0.5:            # ---------------- spoofing ----------------
            mag = float(np.exp(rng.uniform(np.log(100), np.log(3000))))   # metres
            az, vert = rng.uniform(0, 2 * np.pi), rng.normal(0, 0.15)
            direction = scene.east * np.cos(az) + scene.north * np.sin(az) + scene.up * vert
            direction /= np.linalg.norm(direction)
            drift = rng.random() < 0.5
            profile = np.linspace(1 / length, 1, length) if drift else np.ones(length)
            off = (profile * mag)[:, None] * direction
            pos[w] += off
            clock_jump = (rng.choice([-1, 1]) * np.exp(rng.uniform(np.log(2e-7), np.log(2e-6)))
                          if rng.random() < 0.5 else 0.0)
            for j, k in enumerate(w):
                res_list[k] = res_list[k] - u_list[k] @ off[j] + C_LIGHT * clock_jump * profile[j]
            br[w] += clock_jump * profile
            label[w], magnitude[w] = 1, mag
            atype[w] = "spoof-drift" if drift else "spoof-jump"
        else:                             # ----------------- jamming ----------------
            sigma = float(np.exp(rng.uniform(np.log(40), np.log(400))))   # metres
            keep_q = rng.uniform(0.5, 0.95)
            for k in w:
                m = len(res_list[k])
                keep = max(MIN_SATS, int(round(m * keep_q)))
                if keep < m:
                    sub = np.sort(rng.choice(m, keep, replace=False))
                    res_list[k], u_list[k] = res_list[k][sub], u_list[k][sub]
                res_list[k] = res_list[k] + rng.normal(0, sigma, len(res_list[k]))
                p = min(pdop(u_list[k]), 10.0)
                pos[k] += rng.normal(0, sigma * p / np.sqrt(3), 3)
                br[k] += rng.normal(0, sigma / C_LIGHT)
            label[w], magnitude[w], atype[w] = 2, sigma, "jam"
        placed += length

    out = compute_features(scene, base.t, pos, br, u_list, res_list)
    out["Timestamp"] = base.timestamp
    out["Day"] = base.day
    out["Label"] = label
    out["Attack_type"] = atype
    out["Attack_magnitude"] = magnitude
    return out


# --------------------------------------------------------------------------- #
# Supervised models (sklearn): Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, SVM
# --------------------------------------------------------------------------- #
MODEL_NAMES = [
    "Random Forest",
    "Gradient Boosting",
    "Decision Tree",
    "Logistic Regression",
    "Support Vector Machine (SVM)",
]


def _new_model(name: str, n_trees: int = 100, seed: int = 42):
    """Instantiate sklearn classifiers for GNSS spoofing detection."""
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=n_trees, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed)
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    if name == "Support Vector Machine (SVM)":
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(random_state=seed), ensemble=False))
    raise ValueError(f"Unknown model: {name}")


@dataclass
class TrainTestResult:
    model: object                           # trained sklearn model
    model_name: str
    metrics: dict                           # accuracy, precision, recall, f1, train_epochs, test_epochs
    report: pd.DataFrame                    # classification report table
    confusion: pd.DataFrame                 # confusion matrix table
    feature_importance: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_test_pred: np.ndarray
    y_test_proba: np.ndarray | None
    test_size: float
    train_size: float
    seed: int
    class_labels: dict
    raw_pred_table: pd.DataFrame            # epoch predictions for the whole raw_dataset
    raw_summary: dict                       # counts & percentages of clean, spoofed, no_fix
    model_comparison: pd.DataFrame          # benchmark table of RF, GB, DT, LR, SVM on the 80/20 split
    label_mode: str
    threshold_m: float


def run_80_20_train_test(scene: Scene, model_name="Random Forest", test_size=0.2,
                         seed=42, threshold_m=500.0, label_mode="threshold",
                         attack_frac=0.35, n_trees=100, compare_all=True) -> TrainTestResult:
    """
    Train an sklearn model using an 80% train / 20% test random split.
    Directly complies with the evaluator's requirement:
    1. Model trained on raw_dataset via sklearn.
    2. 80% of dataset to training, 20% to testing randomly.
    3. Identifies exact counts & percentages of Clear vs Spoofed in raw_dataset.
    """
    valid_mask = scene.epoch["Valid_fix"].to_numpy()
    valid_ep = scene.epoch[valid_mask].copy().reset_index(drop=True)
    if len(valid_ep) == 0:
        raise ValueError("No epoch with valid position fix (>=4 satellites) found.")

    if label_mode == "threshold":
        # GNSS physics rule: distance > threshold or impossible altitude/latlon
        imp_alt = valid_ep["Impossible_altitude"] if "Impossible_altitude" in valid_ep.columns else False
        imp_ll = valid_ep["Impossible_latlon"] if "Impossible_latlon" in valid_ep.columns else False
        y_raw = ((valid_ep["Distance_from_true_m"] > threshold_m) | imp_alt | imp_ll).astype(int)
        class_labels = {0: "Clear", 1: "Spoofed"}
        X_full = valid_ep[FEATURES].copy()
        y_full = y_raw
    else:
        # Simulated attack injection mode
        base = valid_base(scene)
        sim_df = simulate_attacks(scene, base, seed=seed, attack_frac=attack_frac).reset_index(drop=True)
        class_labels = {0: "Clean", 1: "Spoofed", 2: "Jammed"}
        X_full = sim_df[FEATURES].copy()
        y_full = sim_df["Label"]

    strat = y_full if (y_full.value_counts().min() >= 2 and y_full.nunique() > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=seed, stratify=strat
    )

    # Train chosen model
    model = _new_model(model_name, n_trees=n_trees, seed=seed)
    model.fit(X_train, y_train)

    # Predict on test set
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    acc = float(accuracy_score(y_test, y_test_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "train_epochs": len(X_train),
        "test_epochs": len(X_test),
    }

    labels_list = sorted(class_labels.keys())
    names_list = [class_labels[k] for k in labels_list]
    cm = confusion_matrix(y_test, y_test_pred, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=[f"Actual {n}" for n in names_list],
                         columns=[f"Pred {n}" for n in names_list])

    rep_dict = classification_report(y_test, y_test_pred, labels=labels_list,
                                     target_names=names_list, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(rep_dict).T

    # Feature importance via permutation
    try:
        pi = permutation_importance(model, X_test, y_test, scoring="accuracy",
                                    n_repeats=5, random_state=seed, n_jobs=1)
        feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean}).sort_values(
            "Importance", ascending=False).reset_index(drop=True)
    except Exception:
        feat_imp = pd.DataFrame({"Feature": FEATURES, "Importance": [0.0] * len(FEATURES)})

    # Benchmark of all sklearn models on the same split
    cmp_rows = []
    if compare_all:
        for m_name in MODEL_NAMES:
            try:
                clf = _new_model(m_name, n_trees=n_trees, seed=seed)
                clf.fit(X_train, y_train)
                pred_i = clf.predict(X_test)
                p_i, r_i, f1_i, _ = precision_recall_fscore_support(y_test, pred_i, average="weighted", zero_division=0)
                cmp_rows.append({
                    "Model": m_name,
                    "Accuracy": float(accuracy_score(y_test, pred_i)),
                    "Precision": float(p_i),
                    "Recall": float(r_i),
                    "F1-Score": float(f1_i),
                })
            except Exception:
                pass
    model_comparison = pd.DataFrame(cmp_rows)

    # Classify whole raw dataset
    raw_pred_table = scene.epoch.copy()
    raw_pred_table["ML_status"] = "No fix (<4 sats)"
    for c_name in class_labels.values():
        raw_pred_table[f"P_{c_name}"] = np.nan

    if valid_mask.any():
        x_raw_valid = raw_pred_table.loc[valid_mask, FEATURES]
        raw_preds = model.predict(x_raw_valid)
        raw_pred_table.loc[valid_mask, "ML_status"] = [class_labels.get(int(p), "Unknown") for p in raw_preds]

        if hasattr(model, "predict_proba"):
            raw_probas = model.predict_proba(x_raw_valid)
            model_classes = getattr(model, "classes_", labels_list)
            for idx_c, c_val in enumerate(model_classes):
                c_label = class_labels.get(int(c_val))
                if c_label:
                    raw_pred_table.loc[valid_mask, f"P_{c_label}"] = raw_probas[:, idx_c]

    if "P_Clear" in raw_pred_table.columns and raw_pred_table["P_Clear"].notna().any():
        raw_pred_table["P_attack"] = 1.0 - raw_pred_table["P_Clear"]
    elif "P_Clean" in raw_pred_table.columns and raw_pred_table["P_Clean"].notna().any():
        raw_pred_table["P_attack"] = 1.0 - raw_pred_table["P_Clean"]
    else:
        raw_pred_table["P_attack"] = (raw_pred_table["ML_status"].isin(["Spoofed", "Jammed"])).astype(float)

    total_ep = len(raw_pred_table)
    valid_n = int(valid_mask.sum())
    nofix_n = total_ep - valid_n
    counts = raw_pred_table["ML_status"].value_counts()
    clear_n = int(counts.get("Clear", counts.get("Clean", 0)))
    spoofed_n = int(counts.get("Spoofed", 0))
    jammed_n = int(counts.get("Jammed", 0))

    raw_summary = {
        "total_epochs": total_ep,
        "valid_epochs": valid_n,
        "no_fix_epochs": nofix_n,
        "clear_count": clear_n,
        "clear_pct": (clear_n / valid_n * 100) if valid_n > 0 else 0.0,
        "spoofed_count": spoofed_n,
        "spoofed_pct": (spoofed_n / valid_n * 100) if valid_n > 0 else 0.0,
        "jammed_count": jammed_n,
        "jammed_pct": (jammed_n / valid_n * 100) if valid_n > 0 else 0.0,
        "threshold_m": threshold_m,
        "cadence_s": float(scene.cadence_s),
        "true_pos": scene.true_pos,
    }

    return TrainTestResult(
        model=model, model_name=model_name, metrics=metrics, report=report_df,
        confusion=cm_df, feature_importance=feat_imp, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, y_test_pred=y_test_pred, y_test_proba=y_test_proba,
        test_size=test_size, train_size=1.0 - test_size, seed=seed,
        class_labels=class_labels, raw_pred_table=raw_pred_table, raw_summary=raw_summary,
        model_comparison=model_comparison, label_mode=label_mode, threshold_m=threshold_m
    )


def _training_table(scene, base, seed, attack_frac, repeats):
    """One full clean+attack pass plus extra attack-only passes (more attack variety)."""
    parts = [simulate_attacks(scene, base, seed, attack_frac)]
    for i in range(1, repeats):
        extra = simulate_attacks(scene, base, seed + 101 * i, attack_frac)
        parts.append(extra[extra["Label"] > 0])
    return pd.concat(parts, ignore_index=True)


@dataclass
class SupervisedResult:
    model: object                           # final model, fitted on all days
    feature_importance: pd.DataFrame
    sim_test: pd.DataFrame                  # out-of-fold predictions on simulated epochs
    metrics: dict
    report: pd.DataFrame
    confusion: np.ndarray
    by_severity: pd.DataFrame
    real_pred: pd.DataFrame                 # out-of-fold predictions on the REAL epochs
    folds: list
    model_name: str = ""


def run_supervised(scene: Scene, model_name="Gradient Boosting", n_trees=200, attack_frac=0.35,
                   train_repeats=4, test_repeats=6, seed=42) -> SupervisedResult:
    base = valid_base(scene)
    days = sorted(set(base.day))
    if len(days) < 2:
        raise ValueError("Leave-one-day-out evaluation needs at least 2 different days.")

    sim_parts, real_parts, folds, perm = [], [], [], []
    for d in days:
        test_mask = base.day == d
        train_base, test_base = base.select(~test_mask), base.select(test_mask)
        train = _training_table(scene, train_base, seed, attack_frac, train_repeats)
        rf = _new_model(model_name, n_trees, seed).fit(train[FEATURES], train["Label"])

        test = pd.concat([simulate_attacks(scene, test_base, seed + 9000 + 7 * i, attack_frac)
                          for i in range(test_repeats)], ignore_index=True)
        test["Pred"] = rf.predict(test[FEATURES])
        sim_parts.append(test)
        # Permutation importance on the held-out day (works for any model, no leakage).
        pi = permutation_importance(rf, test[FEATURES], test["Label"], scoring="f1_macro",
                                    n_repeats=3, random_state=seed, n_jobs=1)
        perm.append(pi.importances_mean)
        folds.append({"held_out_day": d,
                      "accuracy": accuracy_score(test["Label"], test["Pred"]),
                      "train_rows": len(train), "test_rows": len(test)})

        # Real (un-modified) epochs of the held-out day, predicted by a model that
        # never saw that day -> no in-sample memorisation.
        real = pd.DataFrame({"Timestamp": test_base.timestamp, "Day": test_base.day})
        real_feats = compute_features(scene, test_base.t, test_base.pos, test_base.br,
                                      test_base.u_list, test_base.res_list)
        proba = rf.predict_proba(real_feats[FEATURES])
        for c_i, cls in enumerate(rf.classes_):
            real[f"P_{CLASS_NAMES[int(cls)]}"] = proba[:, c_i]
        real["Pred"] = rf.predict(real_feats[FEATURES])
        real_parts.append(pd.concat([real, real_feats], axis=1))

    sim_test = pd.concat(sim_parts, ignore_index=True)
    real_pred = pd.concat(real_parts, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
    for name in ("Clean", "Spoofed", "Jammed"):
        if f"P_{name}" not in real_pred:
            real_pred[f"P_{name}"] = 0.0

    y, p = sim_test["Label"].to_numpy(), sim_test["Pred"].to_numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(y, p), "macro_precision": prec,
               "macro_recall": rec, "macro_f1": f1}
    rep = classification_report(y, p, labels=[0, 1, 2], target_names=list(CLASS_NAMES.values()),
                                zero_division=0, output_dict=True)
    report = pd.DataFrame(rep).T.loc[list(CLASS_NAMES.values())][["precision", "recall", "f1-score", "support"]]
    cm = confusion_matrix(y, p, labels=[0, 1, 2])

    # Final model on ALL days (this is the one you would save / deploy).
    full_train = _training_table(scene, base, seed, attack_frac, train_repeats)
    final = _new_model(model_name, n_trees, seed).fit(full_train[FEATURES], full_train["Label"])
    importance = (pd.DataFrame({"Feature": FEATURES, "Importance": np.mean(perm, axis=0)})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    return SupervisedResult(final, importance, sim_test, metrics, report, cm,
                            severity_table(sim_test), real_pred, folds, model_name)


def severity_table(sim_test: pd.DataFrame) -> pd.DataFrame:
    """How detection depends on attack strength (shows the detection limit)."""
    rows = []
    specs = [("Spoofed", 1, "offset (m)", [100, 200, 500, 1000, 3001]),
             ("Jammed", 2, "extra noise sigma (m)", [40, 80, 150, 401])]
    for name, lab, unit, edges in specs:
        sub = sim_test[sim_test["Label"] == lab]
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = sub[(sub["Attack_magnitude"] >= lo) & (sub["Attack_magnitude"] < hi)]
            if len(s) == 0:
                continue
            rows.append({
                "Attack": name, "Strength": f"{lo}-{min(hi, edges[-1]) - (1 if hi == edges[-1] else 0)} {unit}",
                "Epochs": len(s),
                "Detected as any attack": float((s["Pred"] > 0).mean()),
                "Correct type": float((s["Pred"] == lab).mean()),
            })
    return pd.DataFrame(rows)


def baseline_threshold_metrics(sim_test: pd.DataFrame, threshold_m: float,
                               model_name: str = "Learned model") -> dict:
    """The original rule (distance > T) scored on the same simulated test epochs (attack vs clean)."""
    y = (sim_test["Label"] > 0).astype(int)
    rule = (sim_test["Distance_from_true_m"] > threshold_m).astype(int)
    ml = (sim_test["Pred"] > 0).astype(int)
    out = {}
    for name, pred in (("Threshold rule", rule), (model_name, ml)):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        out[name] = {"Precision": pr, "Recall": rc, "F1": f1, "Accuracy": accuracy_score(y, pred)}
    return out


# --------------------------------------------------------------------------- #
# Unsupervised model: Isolation Forest (no labels needed)
# --------------------------------------------------------------------------- #
def run_isolation_forest(epoch: pd.DataFrame, contamination=0.05, n_trees=300, seed=42) -> pd.DataFrame:
    """Anomaly score for every valid-fix epoch. Higher score = more unusual."""
    valid = epoch[epoch["Valid_fix"]].copy()
    iso = IsolationForest(n_estimators=n_trees, contamination=contamination, random_state=seed)
    iso.fit(valid[FEATURES])
    valid["Anomaly_score"] = -iso.score_samples(valid[FEATURES])
    valid["IF_anomaly"] = iso.predict(valid[FEATURES]) == -1
    return valid[["Timestamp", "Anomaly_score", "IF_anomaly"]]


# --------------------------------------------------------------------------- #
# Optional: supervised model on the user's own manual labels (binary)
# --------------------------------------------------------------------------- #
def train_on_manual_labels(df: pd.DataFrame, test_size=0.3, seed=42, n_trees=300,
                           model_name="Gradient Boosting"):
    from sklearn.model_selection import train_test_split
    d = df[df["Valid_fix"].astype(bool)] if "Valid_fix" in df.columns else df
    d = d.dropna(subset=FEATURES + ["Manual_label"])
    x, y = d[FEATURES], d["Manual_label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Training requires both clean (0) and spoofed (1) examples.")
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)
    rf = _new_model(model_name, n_trees, seed).fit(xtr, ytr)
    imp = permutation_importance(rf, xte, yte, scoring="f1", n_repeats=5, random_state=seed, n_jobs=1)
    return rf, yte, rf.predict(xte), imp.importances_mean


# --------------------------------------------------------------------------- #
# Scoring a NEW dataset with a trained model
# --------------------------------------------------------------------------- #
def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """Geodetic latitude/longitude/height (WGS-84) -> ECEF metres."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + alt_m) * np.cos(lat) * np.cos(lon),
                     (n + alt_m) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + alt_m) * np.sin(lat)])


def _raw_timestamps(raw):
    return _timestamps(raw.rename(columns=RAW_RENAME))


def estimate_cadence(raw) -> float:
    """Median spacing (seconds) between distinct timestamps."""
    ts = np.sort(_raw_timestamps(raw).dropna().unique())
    return float(np.median(np.diff(ts)) / np.timedelta64(1, "s")) if len(ts) > 1 else float("nan")


def thin_raw(raw, target_s: float):
    """Keep roughly one epoch per `target_s` seconds (match the training sampling interval)."""
    ts = _raw_timestamps(raw)
    keep, last = [], None
    for t in np.sort(ts.dropna().unique()):
        if last is None or (t - last) / np.timedelta64(1, "s") >= 0.9 * target_s:
            keep.append(t)
            last = t
    return raw[ts.isin(pd.DatetimeIndex(keep))]


def score_dataset(raw: pd.DataFrame, model, true_pos=None, thin_to_s=None):
    """
    Classify every epoch of a new / real-time dataset with an already-trained model.
    Returns (epoch_table, summary_dict). Status is one of
    Clear (Clean) / Spoofed / Jammed / No fix (<4 sats).
    """
    if thin_to_s:
        raw = thin_raw(raw, thin_to_s)
    scene = build_scene(raw, true_pos)
    ep = scene.epoch.copy()
    ep["ML_status"] = "No fix (<4 sats)"

    # Detect model classes
    classes = getattr(model, "classes_", [0, 1])
    if len(classes) == 2:
        class_map = {0: "Clear", 1: "Spoofed"}
    else:
        class_map = {0: "Clean", 1: "Spoofed", 2: "Jammed"}

    for name in ["Clear", "Clean", "Spoofed", "Jammed"]:
        ep[f"P_{name}"] = np.nan

    ok = ep["Valid_fix"].to_numpy()
    if ok.any():
        x = ep.loc[ok, FEATURES]
        pred = model.predict(x)
        ep.loc[ok, "ML_status"] = [class_map.get(int(p), "Unknown") for p in pred]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)
            for j, cls in enumerate(classes):
                c_name = class_map.get(int(cls))
                if c_name:
                    ep.loc[ok, f"P_{c_name}"] = proba[:, j]

    if "P_Clear" in ep.columns and ep["P_Clear"].notna().any():
        ep["P_attack"] = 1.0 - ep["P_Clear"]
    elif "P_Clean" in ep.columns and ep["P_Clean"].notna().any():
        ep["P_attack"] = 1.0 - ep["P_Clean"]
    else:
        ep["P_attack"] = (ep["ML_status"].isin(["Spoofed", "Jammed"])).astype(float)

    counts = ep["ML_status"].value_counts()
    clear_c = int(counts.get("Clear", counts.get("Clean", 0)))
    spoofed_c = int(counts.get("Spoofed", 0))
    jammed_c = int(counts.get("Jammed", 0))
    valid_n = int(ok.sum())

    summary = {
        "epochs": len(ep), "valid": valid_n, "no_fix": int((~ok).sum()),
        "Clean": clear_c, "Clear": clear_c,
        "Clear_pct": (clear_c / valid_n * 100) if valid_n else 0.0,
        "Spoofed": spoofed_c,
        "Spoofed_pct": (spoofed_c / valid_n * 100) if valid_n else 0.0,
        "Jammed": jammed_c,
        "Jammed_pct": (jammed_c / valid_n * 100) if valid_n else 0.0,
        "cadence_s": scene.cadence_s, "true_pos": scene.true_pos,
        "median_dev_m": float(ep.loc[ok, "Distance_from_true_m"].median()) if ok.any() else float("nan"),
    }
    return ep, summary


def attach_status(raw: pd.DataFrame, epoch_result: pd.DataFrame) -> pd.DataFrame:
    """Add ML_status / P_attack to every original raw row (matched by timestamp)."""
    out = raw.copy()
    out["_ts"] = _raw_timestamps(raw)
    m = epoch_result[["Timestamp", "ML_status", "P_attack"]].rename(columns={"Timestamp": "_ts"})
    out = out.merge(m, on="_ts", how="left").drop(columns="_ts")
    out["ML_status"] = out["ML_status"].fillna("Not scored")
    return out


# --------------------------------------------------------------------------- #
# Saving / loading a trained model
# --------------------------------------------------------------------------- #
def default_model_path(base_dir, model_name: str) -> Path:
    slug = model_name.lower().replace(" ", "_")
    return Path(base_dir) / "models" / f"gnss_{slug}.joblib"


def build_model_card(result: object, scene: Scene, settings: dict,
                     training_file: str = "raw_dataset.csv") -> dict:
    """Everything needed to reuse the model later, plus how it was trained."""
    if isinstance(result, TrainTestResult):
        recall = {}
        if "recall" in result.report.columns:
            recall = {str(k): float(v) for k, v in result.report["recall"].items()}
        return {
            "model": result.model,
            "model_name": result.model_name,
            "features": list(FEATURES),
            "classes": dict(result.class_labels),
            "train_cadence_s": float(scene.cadence_s),
            "train_true_pos": np.asarray(scene.true_pos, dtype=float),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_file": training_file,
            "n_train_epochs": len(result.X_train),
            "n_test_epochs": len(result.X_test),
            "split": {"train_size": result.train_size, "test_size": result.test_size, "seed": result.seed},
            "label_mode": settings.get("label_mode", "threshold"),
            "settings": dict(settings),
            "metrics_80_20_test": {**result.metrics, "recall_by_class": recall},
            "raw_summary": result.raw_summary,
            "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__},
        }
    else:
        recall = {k: float(v) for k, v in result.report["recall"].items()}
        return {
            "model": result.model,
            "model_name": result.model_name,
            "features": list(FEATURES),
            "classes": dict(CLASS_NAMES),
            "train_cadence_s": float(scene.cadence_s),
            "train_true_pos": np.asarray(scene.true_pos, dtype=float),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "training_file": training_file,
            "n_train_epochs": int(scene.epoch["Valid_fix"].sum()),
            "label_mode": "simulated",
            "protocol": "final fit on all days; metrics = leave-one-day-out on simulated attacks",
            "settings": dict(settings),
            "cv_metrics_simulated": {**{k: float(v) for k, v in result.metrics.items()}, "recall": recall},
            "ref_flag_rate": float(result.real_pred["Pred"].isin([1, 2]).mean()) if hasattr(result, "real_pred") else 0.0,
            "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__},
        }


def save_model(path, result: "SupervisedResult", scene: Scene, settings: dict,
               training_file: str = "raw_dataset.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_model_card(result, scene, settings, training_file), path)
    return path


def load_model(path) -> dict:
    """Load a model saved by save_model(). Only load files you created yourself."""
    card = joblib.load(path)
    need = {"model", "model_name", "features", "train_cadence_s", "train_true_pos"}
    if not isinstance(card, dict) or not need <= set(card):
        raise ValueError("This is not a model file saved by this project.")
    if list(card["features"]) != FEATURES:
        raise ValueError("The saved model uses a different feature list than ml_model.FEATURES. "
                         "Retrain and save the model again.")
    saved_v = card.get("versions", {}).get("scikit-learn")
    card["warnings"] = ([f"Model was saved with scikit-learn {saved_v}, you have {sklearn.__version__}. "
                         "Results may differ or loading may fail; retrain if in doubt."]
                        if saved_v and saved_v != sklearn.__version__ else [])
    return card


# --------------------------------------------------------------------------- #
# Accuracy on a dataset that has ground-truth labels
# --------------------------------------------------------------------------- #
_CLEAN = {"0", "0.0", "clean", "normal", "false", "no", "genuine", "authentic"}
_SPOOF = {"1", "1.0", "spoofed", "spoof", "spoofing", "true", "yes", "attack"}
_JAM = {"2", "2.0", "jammed", "jam", "jamming"}


def _map_label(x):
    if pd.isna(x):
        return np.nan
    sx = str(x).strip().lower()
    if sx in _CLEAN:
        return 0
    if sx in _SPOOF:
        return 1
    if sx in _JAM:
        return 2
    raise ValueError(f"Unrecognised label value {x!r}. Use 0/1/2 or clean/spoofed/jammed.")


def epoch_labels(raw: pd.DataFrame, column: str) -> pd.Series:
    """Per-epoch ground truth from a per-row label column (most severe label in an epoch wins)."""
    df = pd.DataFrame({"Timestamp": _raw_timestamps(raw), "Label": raw[column].map(_map_label)}).dropna()
    return df.groupby("Timestamp")["Label"].max().astype(int)


def evaluate_labeled(ep: pd.DataFrame, labels: pd.Series) -> dict:
    """Compare predictions with ground truth on epochs that have a position fix."""
    d = ep[ep["Valid_fix"]].merge(labels.rename("True_label"), left_on="Timestamp",
                                  right_index=True, how="inner")
    if d.empty:
        raise ValueError("No labelled epoch with a position fix was found.")
    ids = {v: k for k, v in CLASS_NAMES.items()}
    y = d["True_label"].to_numpy()
    p = d["ML_status"].map(ids).to_numpy()
    names = list(CLASS_NAMES.values())
    cm = pd.DataFrame(confusion_matrix(y, p, labels=[0, 1, 2]),
                      index=[f"Actual {n}" for n in names], columns=[f"Pred {n}" for n in names])
    rep = pd.DataFrame(classification_report(y, p, labels=[0, 1, 2], target_names=names,
                                             zero_division=0, output_dict=True)).T
    rep = rep.loc[names][["precision", "recall", "f1-score", "support"]]
    ya, pa = (y > 0).astype(int), (p > 0).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(ya, pa, average="binary", zero_division=0)
    return {
        "n_labelled_epochs": len(d),
        "n_true_attack": int(ya.sum()), "n_true_clean": int((ya == 0).sum()),
        "accuracy": float(accuracy_score(y, p)),
        "attack_vs_clean": {"accuracy": float(accuracy_score(ya, pa)), "precision": float(pr),
                            "recall": float(rc), "f1": float(f1)},
        "confusion": cm, "report": rep,
        "n_valid_unlabelled": int(len(ep[ep["Valid_fix"]]) - len(d)),
    }