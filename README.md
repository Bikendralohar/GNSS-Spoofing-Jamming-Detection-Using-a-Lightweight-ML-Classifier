# GNSS Spoofing & Jamming Detection System

## Overview
This project detects GNSS spoofing and jamming anomalies using **Machine Learning (Scikit-Learn)**. 

### Evaluator Specification Compliant:
1. **Model Training via Scikit-Learn (`sklearn`)**:
   - Random Forest (`RandomForestClassifier`)
   - Gradient Boosting (`HistGradientBoostingClassifier`)
   - Decision Tree (`DecisionTreeClassifier`)
   - Logistic Regression (`LogisticRegression`)
   - Support Vector Machine (`SVC`)
2. **80% Training / 20% Testing Random Partition**:
   - Splitting is performed randomly every time using `sklearn.model_selection.train_test_split(..., test_size=0.20, random_state=seed)`.
   - Dedicated "🎲 Random Split & Retrain" button to dynamically randomize the 80/20 split and re-evaluate on the fly.
3. **Raw Dataset Analysis**:
   - Clearly computes and displays exact counts and percentages of **Clear** vs **Spoofed** epochs in `raw_dataset.csv`.
4. **Real-Time Dataset Predictor**:
   - Upload any real-time generated or recorded GNSS CSV to classify each epoch as **Clear** or **Spoofed**, with probability plots and downloadable results.

---

## How to Run

1. Make sure Python 3.9+ is installed.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the dashboard:
```bash
python -m streamlit run dashboard.py
```

---

## Project Structure

| File | Purpose |
|------|---------|
| `dashboard.py` | Interactive Streamlit Dashboard with 80/20 evaluation, raw dataset analysis, and real-time predictor |
| `ml_model.py`  | GNSS feature engineering, 80/20 train-test pipeline, sklearn models, dataset scoring |
| `raw_dataset.csv` | Baseline GNSS observation dataset (7,383 rows, 1,379 epochs) |
| `models/` | Saved trained models (`.joblib`) for reuse and deployment |
| `requirements.txt` | Python library requirements |

---

## Features Extracted per Epoch
- **Satellite_count**: Contributing visible satellites in the epoch fix
- **PDOP**: Position Dilution of Precision (satellite geometry quality)
- **Distance_from_true_m**: 3D spatial displacement from estimated resting position
- **Vertical_dev_m**: Altitude / vertical axis deviation
- **Movement_since_previous_m**: Step displacement from consecutive epoch
- **Clock_bias_change_s**: Receiver clock bias jump
- **Res_median_m**: Median pseudorange residual
- **Res_std_m**: Pseudorange residual dispersion
- **Res_max_dev_m**: Peak single-satellite residual error
