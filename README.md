# GNSS Spoofing & Jamming Detection System

Detects GNSS spoofing and jamming per epoch from receiver measurements, using scikit-learn
classifiers trained on attacks injected into real epochs (labels: Clean / Spoofed / Jammed).

## Data
| File | What it is |
|------|------------|
| `raw_dataset.csv` | **Training data**: 7,383 satellite rows, 28 GPS satellites, 1,379 epochs at 300 s, 6-10 Jan 2025, stationary receiver. *Source station / website: add here.* `P_correction` = pseudorange in metres as supplied with the recording; the residual is `P_correction - range_to_reference + c * satellite_clock`. |
| `synthetic_1hz_test_set.csv` | **Synthetic 1 Hz test set** (was `realtime_satellite_data_with_attacks.csv`): 20,076 rows, 1,854 epochs, 14 Oct 2026, with a `true_label` column (0 clean, 1 spoofed, 2 jammed). Used only for testing, never for training. |

## Reproduce the paper
```bash
pip install -r requirements.txt
python reproduce_paper.py      # all tables -> results/, all five models -> models/
python -m streamlit run dashboard.py
```
`reproduce_paper.py` runs: 10-seed 80/20 evaluation (macro metrics, mean +- std), leave-one-day-out
validation (the more reliable result), the 500 m threshold baseline, the Isolation Forest, and scoring of the
synthetic 1 Hz test set. Every model in `models/` is a 3-class simulated-attack model refit on all five days;
each model card records `label_mode`, seed, metrics and library versions.

## Project structure
| File | Purpose |
|------|---------|
| `ml_model.py` | Feature engineering, attack simulator, 80/20 and leave-one-day-out pipelines, scoring |
| `reproduce_paper.py` | One command that regenerates every number in the paper and the saved models |
| `dashboard.py` | Streamlit dashboard (default labelling: simulated attacks) |
| `models/` | Five saved models (`.joblib`), one per classifier |
| `results/` | CSV/JSON outputs written by `reproduce_paper.py` |

## Features per epoch
Satellite_count, PDOP, Distance_from_true_m, Vertical_dev_m, Movement_since_previous_m,
Clock_bias_change_s, Res_median_m, Res_std_m, Res_max_dev_m.

## Note on the "Physical Threshold" labelling mode
Its label is computed from `Distance_from_true_m`, which is also an input feature, so tree models reach 100%
trivially. Use it only as a baseline, not as evidence of detection skill.
