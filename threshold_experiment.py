from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# 1. CAU HINH THI NGHIEM
# ============================================================

TARGET_SENSITIVITY = 0.95
OUTER_SPLITS = 5
INNER_SPLITS = 5
SEEDS = range(10)

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "results" / "threshold_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. TAI DU LIEU
# ============================================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

# Trong sklearn:
# target = 0: malignant (ac tinh)
# target = 1: benign (lanh tinh)
# Chuyen thanh malignant = 1, benign = 0
y = (data.target == 0).astype(int)

print("=" * 70)
print("THRESHOLD EXPERIMENT - BREAST CANCER")
print("=" * 70)
print(f"So mau: {len(X)}")
print(f"So dac trung: {X.shape[1]}")
print(f"So mau ac tinh (y=1): {int(y.sum())}")
print(f"So mau lanh tinh (y=0): {int((y == 0).sum())}")
print(f"Muc Sensitivity muc tieu: {TARGET_SENSITIVITY:.2f}")
print(f"Thu muc ket qua: {OUTPUT_DIR}")
print()


# ============================================================
# 3. KHAI BAO HAI MO HINH
# ============================================================

def build_models():
    """
    Tao cac pipeline tien xu ly va mo hinh.
    Cac gia tri C duoi day duoc co dinh = 1.0.
    """
    return {
        "SVM Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="linear",
                C=1.0
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                max_iter=5000,
                random_state=42
            )),
        ]),
    }


def get_score_method(model_name):
    """
    SVM Linear su dung decision_function.
    Logistic Regression su dung xac suat lop ac tinh.
    """
    if model_name == "SVM Linear":
        return "decision_function"
    return "predict_proba"


def get_positive_scores(model, model_name, X_data):
    """
    Lay diem du doan cho lop ac tinh (y=1).
    """
    if model_name == "SVM Linear":
        return model.decision_function(X_data)

    # Lop 1 la malignant theo cach ma hoa y o tren
    positive_index = list(model.classes_).index(1)
    return model.predict_proba(X_data)[:, positive_index]


# ============================================================
# 4. CHON NGUONG CHI TU DU LIEU HUAN LUYEN
# ============================================================

def choose_threshold(y_true, scores, target_sensitivity):
    """
    Chon nguong cao nhat ma Sensitivity tren du lieu OOF
    van dat muc tieu.

    Du doan ac tinh neu score >= threshold.

    Khong su dung outer test de chon nguong.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # Thu cac nguong score duy nhat theo thu tu tang dan.
    # Them -inf de co truong hop du doan tat ca la ac tinh.
    candidates = np.concatenate((
        [-np.inf],
        np.sort(np.unique(scores))
    ))

    valid_thresholds = []

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        sensitivity = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        if sensitivity >= target_sensitivity:
            valid_thresholds.append(threshold)

    if not valid_thresholds:
        # Ve ly thuyet -inf se du doan tat ca la ac tinh.
        # Truong hop nay chi de bao ve chuong trinh.
        return -np.inf, np.nan

    # Nguong cao nhat van dat Sensitivity muc tieu.
    selected_threshold = max(valid_thresholds)

    selected_predictions = (
        scores >= selected_threshold
    ).astype(int)

    selected_sensitivity = recall_score(
        y_true,
        selected_predictions,
        zero_division=0
    )

    return float(selected_threshold), float(selected_sensitivity)


# ============================================================
# 5. HAM TINH CAC CHI SO
# ============================================================

def calculate_metrics(y_true, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, predictions, zero_division=0
        ),
        "specificity": specificity,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 6. NESTED CROSS-VALIDATION VOI CHON NGUONG
# ============================================================

all_fold_results = []
start_time = time.time()

total_runs = len(SEEDS) * OUTER_SPLITS * 2
run_number = 0

for seed in SEEDS:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=seed
    )

    for outer_fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        models = build_models()

        for model_name, model in models.items():
            run_number += 1

            print(
                f"[{run_number}/{total_runs}] "
                f"Seed={seed}, Outer fold={outer_fold}, "
                f"Model={model_name}"
            )

            # ------------------------------------------------
            # A. Tao du doan out-of-fold chi tren outer train
            # ------------------------------------------------
            inner_cv = StratifiedKFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=seed * 100 + outer_fold
            )

            method = get_score_method(model_name)

            if method == "decision_function":
                inner_scores = cross_val_predict(
                    model,
                    X_train,
                    y_train,
                    cv=inner_cv,
                    method="decision_function",
                    n_jobs=1
                )
            else:
                inner_probabilities = cross_val_predict(
                    model,
                    X_train,
                    y_train,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=1
                )

                # predict_proba tra ve cac cot theo model.classes_.
                # Trong bai toan nay, lop 1 la malignant.
                inner_scores = inner_probabilities[:, 1]

            # ------------------------------------------------
            # B. Chon nguong tu inner OOF scores
            # ------------------------------------------------
            threshold, inner_sensitivity = choose_threshold(
                y_train,
                inner_scores,
                TARGET_SENSITIVITY
            )

            # ------------------------------------------------
            # C. Fit lai tren toan bo outer train
            # ------------------------------------------------
            model.fit(X_train, y_train)

            # ------------------------------------------------
            # D. Danh gia mot lan tren outer test
            # ------------------------------------------------
            test_scores = get_positive_scores(
                model,
                model_name,
                X_test
            )

            test_predictions = (
                test_scores >= threshold
            ).astype(int)

            metrics = calculate_metrics(
                y_test,
                test_predictions,
                test_scores
            )

            row = {
                "seed": seed,
                "outer_fold": outer_fold,
                "model": model_name,
                "target_sensitivity": TARGET_SENSITIVITY,
                "threshold": threshold,
                "inner_oof_sensitivity": inner_sensitivity,
                **metrics,
            }

            all_fold_results.append(row)

            print(
                f"    Threshold={threshold:.6f} | "
                f"Inner Sensitivity={inner_sensitivity:.4f} | "
                f"Outer Sensitivity={metrics['sensitivity']:.4f} | "
                f"Outer Specificity={metrics['specificity']:.4f} | "
                f"Outer F1={metrics['f1']:.4f}"
            )

            # Luu lien tuc de giam nguy co mat ket qua neu bi loi
            fold_df = pd.DataFrame(all_fold_results)
            fold_path = OUTPUT_DIR / "threshold_fold_results.csv"
            fold_df.to_csv(
                fold_path,
                index=False,
                encoding="utf-8-sig"
            )


# ============================================================
# 7. TAO BANG TONG HOP TRUNG BINH VA DO LECH CHUAN
# ============================================================

fold_results = pd.DataFrame(all_fold_results)

metric_columns = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "threshold",
    "inner_oof_sensitivity",
]

summary_rows = []

for model_name, group in fold_results.groupby("model"):
    row = {
        "model": model_name,
        "target_sensitivity": TARGET_SENSITIVITY,
        "n_outer_folds": len(group),
    }

    for metric in metric_columns:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)

    # Bao cao them so mau ac tinh bi bo sot va so mau lanh tinh
    # bi phan loai nham la ac tinh tren tat ca outer test folds.
    row["total_fn"] = int(group["fn"].sum())
    row["total_fp"] = int(group["fp"].sum())
    row["total_tp"] = int(group["tp"].sum())
    row["total_tn"] = int(group["tn"].sum())

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary_path = OUTPUT_DIR / "threshold_summary.csv"
fold_path = OUTPUT_DIR / "threshold_fold_results.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

fold_results.to_csv(
    fold_path,
    index=False,
    encoding="utf-8-sig"
)

elapsed_minutes = (time.time() - start_time) / 60

print("\n" + "=" * 70)
print("THRESHOLD EXPERIMENT COMPLETED")
print("=" * 70)
print(f"Thoi gian: {elapsed_minutes:.2f} phut")
print(f"Ket qua tung fold: {fold_path}")
print(f"Bang tong hop: {summary_path}")

display_columns = [
    "model",
    "accuracy_mean",
    "sensitivity_mean",
    "specificity_mean",
    "f1_mean",
    "roc_auc_mean",
    "threshold_mean",
    "threshold_std",
]

print("\nSUMMARY:")
print(
    summary[display_columns]
    .sort_values("f1_mean", ascending=False)
    .to_string(index=False, float_format=lambda value: f"{value:.4f}")
)