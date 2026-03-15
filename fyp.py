# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None  # type: ignore[assignment]

RANDOM_STATE = 42
TARGET_BUCKET_PRECISION = 0.80
MIN_THRESHOLD_GAP = 0.08
MIN_BUCKET_SHARE = 0.03
RISK_CONFIG_PATH = Path("risk_thresholds.json")

COMPANY_FEATURES = [
    "Industry",
    "workforce_impacted %",
    "Funds_Raised",
    "Funds_Raised_Log",
    "Stage",
    "Country",
    "Year",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
]

EMPLOYEE_FEATURES = [
    "Education",
    "JoiningYear",
    "City",
    "PaymentTier",
    "Age",
    "Gender",
    "EverBenched",
    "ExperienceInCurrentDomain",
    "Performance_Score",
    "Skill_Count",
    "Experience_Age_Ratio",
    "Skills",
]


@dataclass
class TrainOutput:
    model: CalibratedClassifierCV
    scored_df: pd.DataFrame
    thresholds: dict[str, float]
    selected_model: str
    cv_roc_auc: float


def _safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None


def _choose_bucket_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    bucket: str,
    target_precision: float,
    min_share: float,
) -> tuple[float, float, float, float]:
    if bucket not in {"high", "low"}:
        raise ValueError("bucket must be either 'high' or 'low'.")

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    thresholds = np.unique(np.round(p, 6))
    if thresholds.size == 0:
        return 0.5, 0.0, 0.0, 0.0

    best_above_target: tuple[float, float, float, float] | None = None
    best_precision: tuple[float, float, float, float] | None = None

    positives = float(y.sum())
    negatives = float((1 - y).sum())

    for thr in thresholds:
        mask = p >= thr if bucket == "high" else p <= thr
        share = float(mask.mean())
        if share < min_share:
            continue
        if not np.any(mask):
            continue

        if bucket == "high":
            precision = float(y[mask].mean())
            recall = float(y[mask].sum() / positives) if positives > 0 else 0.0
        else:
            precision = float((1 - y[mask]).mean())
            recall = float((1 - y[mask]).sum() / negatives) if negatives > 0 else 0.0

        candidate = (float(thr), precision, recall, share)

        if precision >= target_precision:
            if (
                best_above_target is None
                or candidate[2] > best_above_target[2]
                or (np.isclose(candidate[2], best_above_target[2]) and candidate[1] > best_above_target[1])
            ):
                best_above_target = candidate

        if (
            best_precision is None
            or candidate[1] > best_precision[1]
            or (np.isclose(candidate[1], best_precision[1]) and candidate[2] > best_precision[2])
        ):
            best_precision = candidate

    selected = best_above_target or best_precision
    if selected is None:
        if bucket == "high":
            precision, recall, pr_thresholds = precision_recall_curve(y, p)
        else:
            precision, recall, pr_thresholds = precision_recall_curve(1 - y, 1 - p)
        if pr_thresholds.size == 0:
            return 0.5, 0.0, 0.0, 0.0
        idx = int(np.argmax(precision[:-1]))
        if bucket == "high":
            return float(pr_thresholds[idx]), float(precision[idx]), float(recall[idx]), 0.0
        return float(1.0 - pr_thresholds[idx]), float(precision[idx]), float(recall[idx]), 0.0

    return selected


def learn_risk_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_precision: float = TARGET_BUCKET_PRECISION,
    min_gap: float = MIN_THRESHOLD_GAP,
    min_share: float = MIN_BUCKET_SHARE,
) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(y_proba, dtype=float)

    high_min, high_precision, high_recall, high_share = _choose_bucket_threshold(
        y_true=y,
        scores=p,
        bucket="high",
        target_precision=target_precision,
        min_share=min_share,
    )

    low_max, low_precision, low_recall, low_share = _choose_bucket_threshold(
        y_true=y,
        scores=p,
        bucket="low",
        target_precision=target_precision,
        min_share=min_share,
    )

    low_max = float(np.clip(low_max, 0.05, 0.90))
    high_min = float(np.clip(high_min, 0.10, 0.95))

    if low_max + min_gap >= high_min:
        midpoint = (low_max + high_min) / 2.0
        low_max = float(np.clip(midpoint - (min_gap / 2.0), 0.05, 0.90))
        high_min = float(np.clip(midpoint + (min_gap / 2.0), 0.10, 0.95))
        if low_max >= high_min:
            low_max, high_min = 0.40, 0.70

    return {
        "low_max": round(low_max, 6),
        "high_min": round(high_min, 6),
        "target_precision": round(float(target_precision), 6),
        "min_bucket_share": round(float(min_share), 6),
        "low_bucket_precision": round(low_precision, 6),
        "low_bucket_recall": round(low_recall, 6),
        "low_bucket_share": round(low_share, 6),
        "high_bucket_precision": round(high_precision, 6),
        "high_bucket_recall": round(high_recall, 6),
        "high_bucket_share": round(high_share, 6),
    }


def score_to_risk_label(score: float, low_max: float, high_min: float) -> str:
    if score >= high_min:
        return "High"
    if score <= low_max:
        return "Low"
    return "Medium"


def _bucket_coverage(y_true: pd.Series, y_proba: np.ndarray, low_max: float, high_min: float) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(y_proba, dtype=float)

    low_mask = p <= low_max
    high_mask = p >= high_min
    med_mask = ~(low_mask | high_mask)

    low_precision = float((1 - y[low_mask]).mean()) if np.any(low_mask) else 0.0
    high_precision = float(y[high_mask].mean()) if np.any(high_mask) else 0.0

    return {
        "low_share": float(low_mask.mean()),
        "medium_share": float(med_mask.mean()),
        "high_share": float(high_mask.mean()),
        "low_bucket_precision": low_precision,
        "high_bucket_precision": high_precision,
    }


def _print_metrics(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    thresholds: dict[str, float],
) -> None:
    print(f"\n{name} Test Results")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Balanced Accuracy:", round(balanced_accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_true, y_pred, zero_division=0), 4))
    auc = _safe_roc_auc(y_true, y_proba)
    if auc is not None:
        print("ROC-AUC:", round(auc, 4))
    print("PR-AUC:", round(average_precision_score(y_true, y_proba), 4))
    print("Brier Score:", round(brier_score_loss(y_true, y_proba), 4))

    low_max = float(thresholds["low_max"])
    high_min = float(thresholds["high_min"])
    bucket = _bucket_coverage(y_true, y_proba, low_max=low_max, high_min=high_min)
    print(f"Risk buckets => Low <= {low_max:.3f}, Medium ({low_max:.3f}-{high_min:.3f}), High >= {high_min:.3f}")
    print(
        "Coverage =>",
        f"Low: {bucket['low_share']:.1%},",
        f"Medium: {bucket['medium_share']:.1%},",
        f"High: {bucket['high_share']:.1%}",
    )
    print(
        "Bucket precision =>",
        f"Low bucket correctness: {bucket['low_bucket_precision']:.1%},",
        f"High bucket correctness: {bucket['high_bucket_precision']:.1%}",
    )


def _normalize_skills(skills_text: str) -> str:
    tokens = [token.strip().lower() for token in str(skills_text).split(",") if token.strip()]
    if not tokens:
        return "none"
    unique_sorted = sorted(set(tokens))
    return ",".join(unique_sorted)


def _evaluate_candidates(
    dataset_name: str,
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[str, Pipeline, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name = ""
    best_score = -1.0

    for name, pipeline in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
            )

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        print(f"{dataset_name} candidate '{name}' CV ROC-AUC: {mean_score:.4f} +/- {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    assert best_name in candidates, "No candidate model was selected."
    print(f"{dataset_name} selected model: {best_name}")
    return best_name, candidates[best_name], best_score


def _fit_calibrated_model(base_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    model.fit(X, y)
    return model


def train_company_model() -> TrainOutput:
    company_df = pd.read_csv("company-level.csv")

    company_df["Laid_Off_Count"] = company_df["Laid_Off_Count"].fillna(0)
    company_df["Layoff_Flag"] = (company_df["Laid_Off_Count"] > 0).astype(int)

    company_df["Date"] = pd.to_datetime(company_df["Date"], dayfirst=True, errors="coerce")
    company_df["Year"] = company_df["Date"].dt.year
    company_df["Month"] = company_df["Date"].dt.month
    company_df["Quarter"] = company_df["Date"].dt.quarter
    company_df["Month_Sin"] = np.sin(2 * np.pi * company_df["Month"] / 12.0)
    company_df["Month_Cos"] = np.cos(2 * np.pi * company_df["Month"] / 12.0)
    company_df["Funds_Raised_Log"] = np.log1p(company_df["Funds_Raised"].clip(lower=0))

    X = company_df[COMPANY_FEATURES]
    y = company_df["Layoff_Flag"]

    num_cols = [
        "workforce_impacted %",
        "Funds_Raised",
        "Funds_Raised_Log",
        "Year",
        "Quarter",
        "Month_Sin",
        "Month_Cos",
    ]
    cat_cols = ["Industry", "Stage", "Country"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    candidates: dict[str, Pipeline] = {
        "logistic": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        solver="saga",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=450,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        candidates["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=450,
                        learning_rate=0.05,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                ),
            ]
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    selected_name, selected_pipeline, cv_roc_auc = _evaluate_candidates(
        "Company",
        candidates,
        X_train=X_train,
        y_train=y_train,
    )

    calibration_model = _fit_calibrated_model(selected_pipeline, X_train, y_train)
    val_proba = calibration_model.predict_proba(X_val)[:, 1]
    thresholds = learn_risk_thresholds(
        y_true=y_val,
        y_proba=val_proba,
        target_precision=TARGET_BUCKET_PRECISION,
    )

    final_model = _fit_calibrated_model(selected_pipeline, X_train_full, y_train_full)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    _print_metrics("Company", y_test, test_pred, test_proba, thresholds=thresholds)

    company_scores = final_model.predict_proba(X)[:, 1]
    company_df["Company_Risk_Score"] = company_scores
    company_df["Company_Risk_Level"] = [
        score_to_risk_label(score, thresholds["low_max"], thresholds["high_min"])
        for score in company_scores
    ]

    return TrainOutput(
        model=final_model,
        scored_df=company_df,
        thresholds=thresholds,
        selected_model=selected_name,
        cv_roc_auc=cv_roc_auc,
    )


def train_employee_model() -> TrainOutput:
    emp_df = pd.read_csv("Employee-levelData.csv")

    emp_df["Layoff_Risk"] = emp_df["LeaveOrNot"].astype(int)
    emp_df["Skills"] = emp_df["Skills"].fillna("").astype(str).apply(_normalize_skills)
    emp_df["Skill_Count"] = emp_df["Skills"].apply(
        lambda s: 0 if s == "none" else len([token for token in str(s).split(",") if token.strip()])
    )
    emp_df["Experience_Age_Ratio"] = (
        emp_df["ExperienceInCurrentDomain"].astype(float)
        / emp_df["Age"].replace(0, np.nan).astype(float)
    ).fillna(0.0)

    X = emp_df[EMPLOYEE_FEATURES]
    y = emp_df["Layoff_Risk"]

    num_cols = [
        "JoiningYear",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
        "Performance_Score",
        "Skill_Count",
        "Experience_Age_Ratio",
    ]
    cat_cols = ["Education", "City", "Gender", "EverBenched", "Skills"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    candidates: dict[str, Pipeline] = {
        "logistic": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        solver="saga",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }
    if HAS_XGBOOST and XGBClassifier is not None:
        candidates["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=420,
                        learning_rate=0.06,
                        max_depth=5,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_alpha=0.05,
                        reg_lambda=1.8,
                        random_state=RANDOM_STATE,
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                ),
            ]
        )

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    selected_name, selected_pipeline, cv_roc_auc = _evaluate_candidates(
        "Employee",
        candidates,
        X_train=X_train,
        y_train=y_train,
    )

    calibration_model = _fit_calibrated_model(selected_pipeline, X_train, y_train)
    val_proba = calibration_model.predict_proba(X_val)[:, 1]
    thresholds = learn_risk_thresholds(
        y_true=y_val,
        y_proba=val_proba,
        target_precision=TARGET_BUCKET_PRECISION,
    )

    final_model = _fit_calibrated_model(selected_pipeline, X_train_full, y_train_full)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    _print_metrics("Employee", y_test, test_pred, test_proba, thresholds=thresholds)

    employee_scores = final_model.predict_proba(X)[:, 1]
    emp_df["Employee_Risk_Score"] = employee_scores
    emp_df["Employee_Risk_Level"] = [
        score_to_risk_label(score, thresholds["low_max"], thresholds["high_min"])
        for score in employee_scores
    ]

    return TrainOutput(
        model=final_model,
        scored_df=emp_df,
        thresholds=thresholds,
        selected_model=selected_name,
        cv_roc_auc=cv_roc_auc,
    )


def main() -> None:
    company_output = train_company_model()
    employee_output = train_employee_model()

    joblib.dump(company_output.model, "company_model.pkl")
    joblib.dump(employee_output.model, "employee_model.pkl")

    company_output.scored_df.to_csv("company_scored.csv", index=False)
    employee_output.scored_df.to_csv("employee_scored.csv", index=False)

    risk_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_bucket_precision": TARGET_BUCKET_PRECISION,
        "company": {
            "selected_model": company_output.selected_model,
            "cv_roc_auc": round(company_output.cv_roc_auc, 6),
            **company_output.thresholds,
        },
        "employee": {
            "selected_model": employee_output.selected_model,
            "cv_roc_auc": round(employee_output.cv_roc_auc, 6),
            **employee_output.thresholds,
        },
    }
    RISK_CONFIG_PATH.write_text(json.dumps(risk_config, indent=2), encoding="utf-8")

    print("\nModels and scored datasets saved successfully.")
    print(f"Risk threshold config saved: {RISK_CONFIG_PATH.resolve()}")


if __name__ == "__main__":
    main()
