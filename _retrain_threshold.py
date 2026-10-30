"""Re-derive ``best_threshold`` with a corrected F1 objective, out-of-fold, leak-free.

Why this script exists
----------------------
The notebook's threshold cell computed

    f1 = 2 * (precisions - recalls) / (precisions + recalls + 1e-9)

which is not F1.  With both terms in [0, 1] the numerator is almost always
negative, so ``argmax`` landed on an arbitrary early index and produced
``best_threshold = 0.695``.

It also selected that threshold on ``X_test``/``y_test``, i.e. the same rows
used for the final reported metrics -- selection leakage.

This script redoes both steps properly:

  A. uncalibrated, out-of-fold on the training set   -> threshold_A (reference)
  B. calibrated,   out-of-fold on the training set   -> threshold_B (shipped)

Stage B uses nested CV (5 outer folds x 5 inner calibration folds) so the
calibrated probabilities used to pick the threshold never come from a model
that saw the row.  ``threshold_B`` is what gets written to best_threshold.pkl,
because it is derived on the same probability scale the API actually serves.
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

NUMERIC_COLS = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
CATEGORICAL_COLS = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file",
]

# Recovered from the previously serialized model, so the deployed estimator is
# unchanged -- only the threshold and the calibration fit are redone.
TUNED_PARAMS = dict(
    n_estimators=366,
    max_depth=5,
    learning_rate=0.1316690691287814,
    subsample=0.845303589935636,
    colsample_bytree=0.9374256499933686,
    min_child_weight=7,
    gamma=2.391359792655321,
    scale_pos_weight=3.6312213039485766,
    random_state=SEED,
)


def build_preprocessor():
    """Same ColumnTransformer the notebook trained with (no scaling for trees)."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), NUMERIC_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_COLS),
        ]
    )


def build_pipeline(scale_pos_weight):
    params = dict(TUNED_PARAMS, scale_pos_weight=scale_pos_weight)
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(**params)),
    ])


def best_f1_threshold(y_true, y_prob):
    """Threshold maximizing F1 -- corrected formula: 2PR/(P+R)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    idx = int(np.argmax(f1[:-1]))  # last PR point has no matching threshold
    return float(thresholds[idx]), f1[:-1]


def legacy_f1_threshold(y_true, y_prob):
    """The buggy objective, kept only to quantify how far off it was."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precisions - recalls) / (precisions + recalls + 1e-9)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx])


def report(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "stage": name,
        "threshold": round(threshold, 6),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "average_precision": round(average_precision_score(y_true, y_prob), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    # ---- reproduce the notebook's cleaning + split exactly -------------------
    df = pd.read_csv("credit_risk_dataset.csv")
    df_copy = df.copy()
    df_copy.drop_duplicates(inplace=True)
    df_copy = df_copy[df_copy["person_age"] >= 18]
    df_copy = df_copy[df_copy["person_age"] <= 100]
    df_copy = df_copy[df_copy["person_emp_length"] <= df_copy["person_age"]]
    df_copy = df_copy[df_copy["person_emp_length"] <= 60]
    df_copy = df_copy[df_copy["loan_amnt"] > 0]

    X = df_copy.drop("loan_status", axis=1)
    y = df_copy["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"rows={len(df_copy)} train={len(X_train)} test={len(X_test)} "
          f"positive_rate={y_train.mean():.4f}")

    neg, pos = np.bincount(y_train)
    scale_weight = neg / pos
    print(f"scale_pos_weight = {scale_weight:.6f} (recovered {TUNED_PARAMS['scale_pos_weight']:.6f})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- Stage A: uncalibrated OOF probabilities on the training set --------
    pipe = build_pipeline(scale_weight)
    oof_uncal = cross_val_predict(
        pipe, X_train, y_train, cv=skf, method="predict_proba", n_jobs=1
    )[:, 1]
    th_a, _ = best_f1_threshold(y_train, oof_uncal)
    old_th = float(joblib.load("best_threshold.pkl"))
    print(f"[A] corrected OOF threshold   = {th_a:.6f}")
    print(f"[A] legacy-buggy OOF threshold = {legacy_f1_threshold(y_train, oof_uncal):.6f}")
    print(f"[A] previously saved threshold = {old_th:.6f}")
    row_a = report("A_train_oof_uncalibrated", y_train, oof_uncal, th_a)

    # ---- Stage B: calibrated OOF probabilities, nested CV -------------------
    # Inner cv=skf keeps calibration folds stratified; outer cv=skf keeps the
    # evaluated row out of BOTH the base fit and the calibrator fit.
    calib = CalibratedClassifierCV(
        estimator=build_pipeline(scale_weight), method="sigmoid", cv=skf
    )
    oof_cal = cross_val_predict(
        calib, X_train, y_train, cv=skf, method="predict_proba", n_jobs=1
    )[:, 1]
    th_b, _ = best_f1_threshold(y_train, oof_cal)
    print(f"[B] calibrated OOF threshold  = {th_b:.6f}")
    row_b = report("B_train_oof_calibrated", y_train, oof_cal, th_b)

    # ---- Final refit: same recipe the notebook used, on all of X_train ------
    final_model = CalibratedClassifierCV(
        estimator=build_pipeline(scale_weight), method="sigmoid", cv=skf
    )
    final_model.fit(X_train, y_train)
    print(f"[B] refit CalibratedClassifierCV on {len(X_train)} rows "
          f"({len(final_model.calibrated_classifiers_)} calibrators)")

    # ---- Reporting on the untouched hold-out set (report only, no selection) -
    test_prob = final_model.predict_proba(X_test)[:, 1]
    rows = [
        row_a,
        row_b,
        report("C_test_at_new_threshold", y_test, test_prob, th_b),
        report("C_test_at_old_threshold", y_test, test_prob, old_th),
    ]

    # ---- Persist ------------------------------------------------------------
    backup = Path("_backup")
    backup.mkdir(exist_ok=True)
    for f in ("credit_risk_model.pkl", "best_threshold.pkl"):
        if Path(f).exists():
            shutil.copy2(f, backup / f)

    joblib.dump(final_model, "credit_risk_model.pkl")
    joblib.dump(th_b, "best_threshold.pkl")
    print(f"saved credit_risk_model.pkl + best_threshold.pkl ({th_b:.6f})")

    # ---- Verify the shipped pair round-trips and behaves ---------------------
    reloaded_model = joblib.load("credit_risk_model.pkl")
    reloaded_th = joblib.load("best_threshold.pkl")
    rt = reloaded_model.predict_proba(X_test)[:, 1]
    assert np.allclose(rt, test_prob), "reloaded model predicts differently"
    assert abs(reloaded_th - th_b) < 1e-12, "threshold round-trip mismatch"
    print("round-trip check OK")

    summary = {
        "rows_after_cleaning": int(len(df_copy)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "scale_pos_weight": round(float(scale_weight), 6),
        "stage_a_threshold_uncalibrated_oof": round(th_a, 6),
        "stage_b_threshold_calibrated_oof": round(th_b, 6),
        "previous_threshold": round(old_th, 6),
        "metrics": rows,
    }
    Path("_threshold_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print()
    print(f"{'stage':30s} {'thr':>8s} {'prec':>6s} {'rec':>6s} {'f1':>6s} "
          f"{'auc':>6s} {'ap':>6s} {'fp':>5s} {'fn':>5s}")
    for r in rows:
        print(f"{r['stage']:30s} {r['threshold']:8.4f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f} {r['roc_auc']:6.3f} "
              f"{r['average_precision']:6.3f} {r['fp']:5d} {r['fn']:5d}")


if __name__ == "__main__":
    main()
