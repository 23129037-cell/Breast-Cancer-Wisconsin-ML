
"""
ĐỀ TÀI:
Chẩn đoán ung thư vú bằng học máy cổ điển kết hợp lựa chọn đặc trưng và XAI

Chạy:
    python main.py

Pipeline:
1. Load WDBC từ scikit-learn
2. Kiểm tra dữ liệu
3. Train/Test split
4. Baseline: Logistic Regression, KNN, SVM, Random Forest
5. Feature Selection bằng Mutual Information
6. So sánh trước/sau Feature Selection
7. GridSearchCV tối ưu SVM + Feature Selection
8. Đánh giá mô hình cuối
9. Giải thích bằng SHAP
10. Lưu bảng và hình vào results/ và plots/
"""
import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    make_scorer,
)

# =========================
# CẤU HÌNH
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
np.random.seed(RANDOM_STATE)


def specificity_score(y_true, y_pred):
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train + đánh giá mô hình trên test."""
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall_Sensitivity": recall_score(y_test, pred, zero_division=0),
        "Specificity": specificity_score(y_test, pred),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
    }


def main():
    print("=" * 70)
    print("CHẨN ĐOÁN UNG THƯ VÚ - ML CỔ ĐIỂN + FEATURE SELECTION + XAI")
    print("=" * 70)

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    print("\n[1/9] Đang tải Breast Cancer Wisconsin Diagnostic...")

    data = load_breast_cancer(as_frame=True)

    X = data.data.copy()

    # sklearn: 0 = malignant, 1 = benign
    # Quy ước đề tài: 1 = malignant, 0 = benign
    y = (data.target == 0).astype(int)

    print(f"Số mẫu: {X.shape[0]}")
    print(f"Số đặc trưng: {X.shape[1]}")
    print("Benign:", int((y == 0).sum()))
    print("Malignant:", int((y == 1).sum()))

    # Lưu dataset đã chuẩn bị
    df = X.copy()
    df["Diagnosis"] = y.map({0: "Benign", 1: "Malignant"})
    df.to_csv(RESULTS_DIR / "wdbc_prepared.csv", index=False, encoding="utf-8-sig")

    # ============================================================
    # 2. KIỂM TRA DỮ LIỆU
    # ============================================================
    print("\n[2/9] Kiểm tra dữ liệu...")
    missing = int(X.isna().sum().sum())
    duplicated = int(X.duplicated().sum())

    print("Giá trị thiếu:", missing)
    print("Dòng trùng:", duplicated)

    class_counts = y.map({0: "Benign", 1: "Malignant"}).value_counts()
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    ax.set_title("Phân bố lớp Benign / Malignant")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Số mẫu")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png", dpi=200)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(14, 11))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - 30 đặc trưng")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_correlation_heatmap.png", dpi=200)
    plt.close()

    # ============================================================
    # 3. TRAIN / TEST SPLIT
    # ============================================================
    print("\n[3/9] Chia Train/Test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    # ============================================================
    # 4. BASELINE MODELS
    # ============================================================
    print("\n[4/9] Huấn luyện mô hình cơ sở...")
    baseline_models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "Linear SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),

    "RBF SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=2.0,
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
}

    baseline_results = []

    for name, model in baseline_models.items():
        print("  >", name)
        scores = evaluate_model(model, X_train, y_train, X_test, y_test)
        scores["Model"] = name
        baseline_results.append(scores)

    baseline_df = pd.DataFrame(baseline_results).set_index("Model")
    baseline_df.to_csv(
        RESULTS_DIR / "baseline_results.csv",
        encoding="utf-8-sig"
    )

    print("\nBASELINE:")
    print(baseline_df.round(4))

    # Baseline ROC
    plt.figure(figsize=(8, 6))
    for name, model in baseline_models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Baseline Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_baseline_roc.png", dpi=200)
    plt.close()

    # ============================================================
    # 5. FEATURE SELECTION - MUTUAL INFORMATION
    # ============================================================
    print("\n[5/9] Tìm số lượng đặc trưng tốt nhất bằng Mutual Information...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    k_values = [5, 10, 15, 20, 25, 30]
    selector_results = []

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }

    for k in k_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(
                score_func=mutual_info_classif,
                k=k
            )),
            ("model", SVC(
                kernel="rbf",
                C=2.0,
                probability=True,
                random_state=RANDOM_STATE
            ))
        ])

        cv_result = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        selector_results.append({
            "K": k,
            "Accuracy_mean": cv_result["test_accuracy"].mean(),
            "Precision_mean": cv_result["test_precision"].mean(),
            "Recall_mean": cv_result["test_recall"].mean(),
            "F1_mean": cv_result["test_f1"].mean(),
            "AUC_mean": cv_result["test_roc_auc"].mean(),
        })

    k_df = pd.DataFrame(selector_results)
    k_df.to_csv(
        RESULTS_DIR / "feature_selection_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best_k = int(k_df.loc[k_df["F1_mean"].idxmax(), "K"])
    print("K tốt nhất theo F1:", best_k)

    plt.figure(figsize=(8, 5))
    plt.plot(k_df["K"], k_df["Accuracy_mean"], marker="o", label="Accuracy")
    plt.plot(k_df["K"], k_df["Recall_mean"], marker="o", label="Recall")
    plt.plot(k_df["K"], k_df["F1_mean"], marker="o", label="F1")
    plt.plot(k_df["K"], k_df["AUC_mean"], marker="o", label="AUC")
    plt.xlabel("Số lượng đặc trưng K")
    plt.ylabel("CV score")
    plt.title("Ảnh hưởng của số lượng đặc trưng")
    plt.xticks(k_values)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_feature_selection_k.png", dpi=200)
    plt.close()

    # Fit selector chỉ trên TRAIN
    final_selector = SelectKBest(
        score_func=mutual_info_classif,
        k=best_k
    )

    X_train_selected = final_selector.fit_transform(X_train, y_train)
    X_test_selected = final_selector.transform(X_test)

    selected_features = X.columns[final_selector.get_support()].tolist()

    feature_scores = pd.DataFrame({
        "Feature": X.columns,
        "MI_Score": final_selector.scores_
    }).sort_values("MI_Score", ascending=False)

    feature_scores.to_csv(
        RESULTS_DIR / "mutual_information_scores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "Selected_Feature": selected_features
    }).to_csv(
        RESULTS_DIR / "selected_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nĐẶC TRƯNG ĐƯỢC CHỌN:")
    for i, f in enumerate(selected_features, 1):
        print(f"  {i:02d}. {f}")

    top_scores = feature_scores.head(best_k).sort_values("MI_Score")
    plt.figure(figsize=(9, max(4, best_k * 0.4)))
    plt.barh(top_scores["Feature"], top_scores["MI_Score"])
    plt.xlabel("Mutual Information Score")
    plt.ylabel("Feature")
    plt.title(f"Top {best_k} đặc trưng")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_selected_features_mi.png", dpi=200)
    plt.close()

    # ============================================================
    # 6. MODELS AFTER FEATURE SELECTION
    # ============================================================
    print("\n[6/9] Huấn luyện lại mô hình sau Feature Selection...")

    selected_models = {
        "Logistic Regression + FS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
        ]),
        "KNN + FS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7))
        ]),
        "SVM + FS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=2.0,
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
        "Random Forest + FS": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
    }

    selected_results = []

    for name, model in selected_models.items():
        print("  >", name)
        scores = evaluate_model(
            model,
            X_train_selected,
            y_train,
            X_test_selected,
            y_test
        )
        scores["Model"] = name
        selected_results.append(scores)

    selected_df = pd.DataFrame(selected_results).set_index("Model")
    selected_df.to_csv(
        RESULTS_DIR / "feature_selected_model_results.csv",
        encoding="utf-8-sig"
    )

    comparison_df = pd.concat([
        baseline_df.assign(Stage="Baseline - 30 features"),
        selected_df.assign(Stage=f"Feature Selection - {best_k} features")
    ])
    comparison_df.to_csv(
        RESULTS_DIR / "baseline_vs_feature_selection.csv",
        encoding="utf-8-sig"
    )

    print("\nSAU FEATURE SELECTION:")
    print(selected_df.round(4))

    # ============================================================
    # 7. GRID SEARCH - SVM + FEATURE SELECTION
    # ============================================================
    print("\n[7/9] Tối ưu SVM + Feature Selection bằng GridSearchCV...")

    svm_fs_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=mutual_info_classif)),
        ("model", SVC(
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    param_grid = {
        "select__k": [5, 10, 15, 20, 25, 30],
        "model__C": [0.1, 1, 2, 5, 10],
        "model__gamma": ["scale", 0.01, 0.001],
        "model__kernel": ["rbf"]
    }

    grid = GridSearchCV(
        estimator=svm_fs_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)

    # Lưu mô hình SVM tốt nhất để sử dụng trên website
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        grid.best_estimator_,
        "models/svm_model.joblib"
    )

    # Lưu tên đặc trưng đầu vào
    with open("models/feature_names.json", "w", encoding="utf-8") as f:
        json.dump(
            list(X_train.columns),
            f,
            ensure_ascii=False,
            indent=4
        )

    # Lưu giá trị trung vị để điền sẵn trên website
    with open("models/feature_medians.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                key: float(value)
                for key, value in X_train.median().to_dict().items()
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    print("Da luu mo hinh cho website.")

    print("\nBest parameters:")
    print(grid.best_params_)
    print("Best CV F1:", round(grid.best_score_, 4))

    pd.DataFrame([grid.best_params_]).to_csv(
        RESULTS_DIR / "best_parameters.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # 8. FINAL EVALUATION
    # ============================================================
    print("\n[8/9] Đánh giá mô hình cuối trên TEST...")

    best_model = grid.best_estimator_

    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    final_metrics = {
        "Accuracy": accuracy_score(y_test, test_pred),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Sensitivity_Recall": recall_score(y_test, test_pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_test, test_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, test_prob),
    }

    final_df = pd.DataFrame([final_metrics])
    final_df.to_csv(
        RESULTS_DIR / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nKẾT QUẢ CUỐI:")
    for key, value in final_metrics.items():
        print(f"{key:22s}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["Benign", "Malignant"],
        digits=4
    ))

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )
    plt.title("Confusion Matrix - Final Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_final_confusion_matrix.png", dpi=200)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {final_metrics['ROC_AUC']:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Final SVM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_final_roc.png", dpi=200)
    plt.close()

    # Predictions
    predictions_df = X_test.copy()
    predictions_df["True_Label"] = y_test.values
    predictions_df["Predicted_Label"] = test_pred
    predictions_df["Probability_Malignant"] = test_prob
    predictions_df["True_Name"] = predictions_df["True_Label"].map({
        0: "Benign", 1: "Malignant"
    })
    predictions_df["Predicted_Name"] = predictions_df["Predicted_Label"].map({
        0: "Benign", 1: "Malignant"
    })

    predictions_df.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ============================================================
    # 9. XAI - SHAP
    # ============================================================
    print("\n[9/9] Tính SHAP XAI...")

    # Để SHAP nhanh và ổn định, ta giải thích pipeline cuối.
    # Chỉ dùng một phần nhỏ làm background.
    background_size = min(40, len(X_train))
    explain_size = min(30, len(X_test))

    background = X_train.sample(
        n=background_size,
        random_state=RANDOM_STATE
    )
    X_explain = X_test.iloc[:explain_size].copy()

    def predict_proba_for_shap(data_array):
        data_df = pd.DataFrame(
            data_array,
            columns=X.columns
        )
        return best_model.predict_proba(data_df)

    print("Đang tạo SHAP Explainer...")
    explainer = shap.Explainer(
        predict_proba_for_shap,
        background
    )

    print("Đang tính SHAP values...")
    shap_exp = explainer(X_explain)

    values = shap_exp.values

    if values.ndim == 3:
        shap_malignant = values[:, :, 1]
        base_values = shap_exp.base_values[:, 1]
    else:
        shap_malignant = values
        base_values = shap_exp.base_values

    shap_malignant_exp = shap.Explanation(
        values=shap_malignant,
        base_values=base_values,
        data=X_explain.values,
        feature_names=X.columns.tolist()
    )

    # SHAP beeswarm
    plt.figure()
    shap.plots.beeswarm(
        shap_malignant_exp,
        max_display=15,
        show=False
    )
    plt.title("SHAP Summary - Ảnh hưởng đến dự đoán Malignant")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # SHAP bar
    plt.figure()
    shap.plots.bar(
        shap_malignant_exp,
        max_display=15,
        show=False
    )
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_malignant).mean(axis=0)

    shap_importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    shap_importance.to_csv(
        RESULTS_DIR / "shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTOP 15 SHAP FEATURES:")
    print(shap_importance.head(15).round(6).to_string(index=False))

    # Waterfall cho mẫu đầu tiên
    sample_shap_values = shap_malignant[0]

    sample_exp = shap.Explanation(
        values=sample_shap_values,
        base_values=base_values[0],
        data=X_explain.iloc[0].values,
        feature_names=X.columns.tolist()
    )

    plt.figure()
    shap.plots.waterfall(
        sample_exp,
        max_display=12,
        show=False
    )
    plt.title("SHAP Waterfall - Mẫu test đầu tiên")
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "10_shap_waterfall_sample_1.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # TÓM TẮT
    # ============================================================
    print("\n" + "=" * 70)
    print("HOÀN THÀNH PIPELINE 1 -> 9")
    print("=" * 70)
    print(f"Best K: {best_k}")
    print(f"Best parameters: {grid.best_params_}")
    print("\nCác thư mục kết quả:")
    print("  results/  -> bảng CSV")
    print("  plots/    -> biểu đồ PNG")
    print("\nĐây là các file quan trọng cho báo cáo:")
    print("  - baseline_results.csv")
    print("  - feature_selection_results.csv")
    print("  - selected_features.csv")
    print("  - baseline_vs_feature_selection.csv")
    print("  - best_parameters.csv")
    print("  - final_metrics.csv")
    print("  - shap_feature_importance.csv")
    print("  - test_predictions.csv")


if __name__ == "__main__":
    main()
