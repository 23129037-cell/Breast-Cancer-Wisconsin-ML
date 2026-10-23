
"""
Nested Cross-Validation experiment for Breast Cancer Wisconsin Diagnostic.

- Dataset: Breast Cancer Wisconsin Diagnostic (WDBC)
- Positive class (1): Malignant
- Negative class (0): Benign
- Outer CV: 5-fold StratifiedKFold, repeated with 10 seeds
- Inner CV: 5-fold StratifiedKFold for hyperparameter tuning
- Compare 6 algorithms, with and without Mutual Information feature selection
- Output is saved to results/nested_cv/
"""

from pathlib import Path
from functools import partial
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEEDS = list(range(10))
OUTER_FOLDS = 5
INNER_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "nested_cv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate evaluation metrics. Malignant is the positive class."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


# ============================================================
# 3. MODEL DEFINITIONS AND PARAMETER GRIDS
# ============================================================

def get_model_configs(seed):
    """
    Each algorithm is evaluated twice:
    1. All 30 features
    2. Mutual Information feature selection

    The scaler and selector remain inside the Pipeline.
    """

    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            },
        },
        {
            "name": "KNN",
            "estimator": KNeighborsClassifier(),
            "params": {
                "model__n_neighbors": [5, 7, 11],
                "model__weights": ["uniform", "distance"],
            },
        },
        {
            "name": "SVM Linear",
            "estimator": SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
            },
        },
        {
            "name": "SVM RBF",
            "estimator": SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__gamma": ["scale", 0.01],
            },
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=seed,
            ),
            "params": {
                "model__max_depth": [None, 5, 10],
                "model__min_samples_leaf": [1, 2],
            },
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            "params": {
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [1, 2],
            },
        },
    ]


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset():
    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )

    # sklearn's original target:
    # 0 = malignant, 1 = benign
    # Convert to the project's convention:
    # 1 = malignant, 0 = benign
    y = (dataset.target == 0).astype(int)

    print("=" * 65)
    print("BREAST CANCER - NESTED CROSS-VALIDATION")
    print("=" * 65)
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Benign (0): {(y == 0).sum()}")
    print(f"Malignant (1): {(y == 1).sum()}")
    print(f"Outer folds per seed: {OUTER_FOLDS}")
    print(f"Number of seeds: {len(RANDOM_SEEDS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 65)

    return X, y


# ============================================================
# 5. RUN NESTED CROSS-VALIDATION
# ============================================================

def run_experiment():
    X, y = load_dataset()

    all_results = []
    start_time = time.time()

    total_outer_folds = len(RANDOM_SEEDS) * OUTER_FOLDS
    current_outer_fold = 0

    for seed in RANDOM_SEEDS:
        print(f"\n{'#' * 65}")
        print(f"RANDOM SEED: {seed}")
        print(f"{'#' * 65}")

        outer_cv = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y), start=1
        ):
            current_outer_fold += 1

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx] if isinstance(
                y, pd.Series
            ) else y[train_idx]
            y_test = y.iloc[test_idx] if isinstance(
                y, pd.Series
            ) else y[test_idx]

            print(
                f"\nOuter fold {outer_fold}/{OUTER_FOLDS} "
                f"| Seed {seed} "
                f"| Overall {current_outer_fold}/{total_outer_folds}"
            )

            inner_cv = StratifiedKFold(
                n_splits=INNER_FOLDS,
                shuffle=True,
                random_state=seed * 100 + outer_fold,
            )

            configs = get_model_configs(seed)

            for config in configs:
                for use_feature_selection in [False, True]:

                    if use_feature_selection:
                        selector = SelectKBest(
                            score_func=partial(
                                mutual_info_classif,
                                random_state=seed,
                            )
                        )
                        feature_mode = "MI selection"
                        selector_params = {
                            "select__k": [5, 10, 15, 20, 30]
                        }
                    else:
                        selector = "passthrough"
                        feature_mode = "All features"
                        selector_params = {}

                    pipeline = Pipeline([
                        ("scaler", StandardScaler()),
                        ("select", selector),
                        ("model", config["estimator"]),
                    ])

                    param_grid = {
                        **config["params"],
                        **selector_params,
                    }

                    search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        scoring="f1",
                        cv=inner_cv,
                        refit=True,
                        n_jobs=-1,
                        error_score="raise",
                    )

                    # Hyperparameter tuning uses only the outer
                    # training partition.
                    search.fit(X_train, y_train)

                    # Evaluate once on the untouched outer test fold.
                    best_model = search.best_estimator_
                    y_pred = best_model.predict(X_test)

                    # Explicitly locate the probability column for
                    # malignant class 1.
                    classes = list(best_model.classes_)
                    malignant_col = classes.index(1)
                    y_prob = best_model.predict_proba(X_test)[
                        :, malignant_col
                    ]

                    metrics = calculate_metrics(
                        y_test, y_pred, y_prob
                    )

                    row = {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": config["name"],
                        "feature_mode": feature_mode,
                        "best_inner_f1": search.best_score_,
                        "best_params": str(search.best_params_),
                        **metrics,
                    }

                    all_results.append(row)

                    print(
                        f"  {config['name']:20s} | "
                        f"{feature_mode:14s} | "
                        f"F1={metrics['f1']:.4f} | "
                        f"Recall={metrics['sensitivity']:.4f} | "
                        f"Spec={metrics['specificity']:.4f}"
                    )

            # Save progress after every outer fold, so intermediate
            # results are not lost if the experiment is interrupted.
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "nested_cv_fold_results.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # ========================================================
    # 6. SAVE FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_DIR / "nested_cv_fold_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
    ]

    grouped = results_df.groupby(
        ["model", "feature_mode"]
    )[metric_columns].agg(["mean", "std"])

    # Flatten the multi-level column names.
    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat in grouped.columns
    ]

    summary_df = grouped.reset_index()

    summary_df.to_csv(
        OUTPUT_DIR / "nested_cv_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = (time.time() - start_time) / 60

    print("\n" + "=" * 80)
    print("NESTED CROSS-VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Elapsed time: {elapsed:.2f} minutes")
    print(f"Fold-level results: {OUTPUT_DIR / 'nested_cv_fold_results.csv'}")
    print(f"Summary results:    {OUTPUT_DIR / 'nested_cv_summary.csv'}")

    display_columns = [
        "model",
        "feature_mode",
        "f1_mean",
        "f1_std",
        "sensitivity_mean",
        "specificity_mean",
        "roc_auc_mean",
    ]

    print("\nSUMMARY (mean and standard deviation across outer folds):")
    print(
        summary_df[display_columns]
        .sort_values("f1_mean", ascending=False)
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    run_experiment()
