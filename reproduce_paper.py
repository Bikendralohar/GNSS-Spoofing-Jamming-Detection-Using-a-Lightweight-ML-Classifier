"""Reproduce every number in the paper and (re)save the deployable models.

Run:  python reproduce_paper.py
Writes: results/*.csv, results/summary.json and models/gnss_<model>.joblib

Protocol (same as the paper)
  * labels   : simulated attacks injected into the real epochs (clean / spoofed / jammed)
  * Table 1/2: 10 seeds (0-9), 80/20 stratified split, macro metrics, mean +- sample std
  * LODO     : leave-one-day-out over the 5 recording days (seed 42), macro metrics
  * models   : final fit on all 5 days for every classifier, saved with a model card
  * test set : saved Gradient Boosting scored on synthetic_1hz_test_set.csv
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

import ml_model as M

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
TRAIN_CSV, TEST_CSV = ROOT / "raw_dataset.csv", ROOT / "synthetic_1hz_test_set.csv"
SEEDS, N_TREES, ATTACK_FRAC = range(10), 100, 0.35
summary = {}

raw = pd.read_csv(TRAIN_CSV)
scene = M.build_scene(raw)
base = M.valid_base(scene)
summary["training_data"] = {
    "rows": int(len(raw)), "satellites": int(raw["Satellite"].nunique()),
    "epochs": int(len(scene.epoch)), "valid_epochs": int(scene.epoch["Valid_fix"].sum()),
    "cadence_s": float(scene.cadence_s)}

# ---- A. 80/20 split repeated over ten seeds (macro metrics) ------------------------------
rows, pooled = [], {m: np.zeros((3, 3), int) for m in M.MODEL_NAMES}
for seed in SEEDS:
    sim = M.simulate_attacks(scene, base, seed=seed, attack_frac=ATTACK_FRAC).reset_index(drop=True)
    X, y = sim[M.FEATURES], sim["Label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    for name in M.MODEL_NAMES:
        pred = M._new_model(name, n_trees=N_TREES, seed=seed).fit(Xtr, ytr).predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                     average="macro", zero_division=0)
        pooled[name] += confusion_matrix(yte, pred, labels=[0, 1, 2])
        rows.append(dict(seed=seed, model=name, accuracy=accuracy_score(yte, pred),
                         macro_precision=p, macro_recall=r, macro_f1=f))
per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT / "table1_per_seed.csv", index=False)
agg = per_seed.groupby("model")[["accuracy", "macro_precision", "macro_recall", "macro_f1"]].agg(["mean", "std"])
agg.to_csv(OUT / "table1_mean_std.csv")
names = ["Clean", "Spoofed", "Jammed"]
for name in ("Gradient Boosting", "Random Forest"):
    cm = pd.DataFrame(pooled[name], index=names, columns=names)
    cm.to_csv(OUT / f"table2_confusion_{name.lower().replace(' ', '_')}.csv")
print("== Table 1 (mean/std over seeds)\n", agg.round(3).to_string())
print("== Table 2 (GB pooled)\n", pd.DataFrame(pooled["Gradient Boosting"], index=names, columns=names))

# ---- B. leave-one-day-out for every model + final fit saved to models/ -------------------
lodo_rows = []
for name in M.MODEL_NAMES:
    res = M.run_supervised(scene, name, n_trees=N_TREES, attack_frac=ATTACK_FRAC, seed=42)
    M.save_model(M.default_model_path(ROOT, name), res, scene,
                 {"model": name, "seed": 42, "n_trees": N_TREES, "label_mode": "simulated",
                  "attack_frac": ATTACK_FRAC}, training_file=TRAIN_CSV.name)
    lodo_rows.append({"model": name, **{k: float(v) for k, v in res.metrics.items()},
                      "fold_acc_min": min(f["accuracy"] for f in res.folds),
                      "fold_acc_max": max(f["accuracy"] for f in res.folds)})
    if name == "Gradient Boosting":
        gb = res
        res.report.to_csv(OUT / "table3_lodo_per_class_gb.csv")
        pd.DataFrame(res.folds).to_csv(OUT / "lodo_folds_gb.csv", index=False)
        res.by_severity.to_csv(OUT / "table4_by_strength_gb.csv", index=False)
        res.feature_importance.to_csv(OUT / "lodo_importance_gb.csv", index=False)
        pd.DataFrame(res.confusion, index=names, columns=names).to_csv(OUT / "lodo_confusion_gb.csv")
lodo = pd.DataFrame(lodo_rows)
lodo.to_csv(OUT / "lodo_all_models.csv", index=False)
print("== LODO, all models\n", lodo.round(3).to_string())

base_cmp = M.baseline_threshold_metrics(gb.sim_test, 500.0, model_name="Gradient Boosting")
summary["baseline_vs_learned"] = base_cmp
flag = gb.real_pred["Pred"].value_counts().to_dict()
summary["lodo_real_epochs_flagged"] = {CN: int(flag.get(k, 0)) for k, CN in M.CLASS_NAMES.items()}
iso = M.run_isolation_forest(scene.epoch)
summary["isolation_forest_flagged"] = [int(iso["IF_anomaly"].sum()), int(len(iso))]
print("== baseline vs learned", json.dumps(base_cmp, indent=1))

# ---- C. synthetic 1 Hz test set scored with the SAVED Gradient Boosting model -------------
test = pd.read_csv(TEST_CSV)
test["_ts"] = pd.to_datetime(test["DATE"] + " " + test["Time"])
truth = test.groupby("_ts")["true_label"].max()
card = M.load_model(M.default_model_path(ROOT, "Gradient Boosting"))
ep, _ = M.score_dataset(test, card["model"], true_pos=None)  # reference = the file's own median position
ep["truth"] = ep["Timestamp"].map(truth)
v = ep[ep["Valid_fix"]].copy()
code = {"Clean": 0, "Spoofed": 1, "Jammed": 2}
y_true, y_pred = v["truth"].astype(int), v["ML_status"].map(code)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
attack_true, attack_pred = y_true > 0, y_pred > 0
summary["synthetic_test_set"] = {
    "rows": int(len(test)), "epochs": int(len(ep)), "valid_epochs": int(len(v)),
    "label_epochs": {str(k): int(x) for k, x in y_true.value_counts().sort_index().items()},
    "accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
    "macro_recall": float(r), "macro_f1": float(f),
    "attack_detection_recall": float((attack_pred & attack_true).sum() / attack_true.sum()),
    "false_alarms_on_clean": int((attack_pred & ~attack_true).sum()),
    "confusion_rows_true_cols_pred": cm.tolist()}
pd.DataFrame(cm, index=names, columns=names).to_csv(OUT / "synthetic_test_confusion_gb.csv")
print("== synthetic 1 Hz test set\n", json.dumps(summary["synthetic_test_set"], indent=1))

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("saved models:", sorted(p.name for p in (ROOT / "models").glob("*.joblib")))
