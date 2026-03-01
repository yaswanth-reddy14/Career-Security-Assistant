# -*- coding: utf-8 -*-

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

RANDOM_STATE = 42
COMPANY_FEATURES = [
    "Industry",
    "workforce_impacted %",
    "Funds_Raised",
    "Stage",
    "Country",
    "Year",
    "Quarter",
    "Month_Sin",
    "Month_Cos",
]


def _safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None


def _print_metrics(name: str, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> None:
    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_true, y_pred, zero_division=0), 4))
    print("F1:", round(f1_score(y_true, y_pred, zero_division=0), 4))
    auc = _safe_roc_auc(y_true, y_proba)
    if auc is not None:
        print("ROC-AUC:", round(auc, 4))


def train_company_model() -> tuple[Pipeline, pd.DataFrame]:
    company_df = pd.read_csv("company-level.csv")

    company_df["Laid_Off_Count"] = company_df["Laid_Off_Count"].fillna(0)
    company_df["Layoff_Flag"] = (company_df["Laid_Off_Count"] > 0).astype(int)

    company_df["Date"] = pd.to_datetime(company_df["Date"], dayfirst=True, errors="coerce")
    company_df["Year"] = company_df["Date"].dt.year
    company_df["Month"] = company_df["Date"].dt.month
    company_df["Quarter"] = company_df["Date"].dt.quarter
    company_df["Month_Sin"] = np.sin(2 * np.pi * company_df["Month"] / 12.0)
    company_df["Month_Cos"] = np.cos(2 * np.pi * company_df["Month"] / 12.0)

    X = company_df[COMPANY_FEATURES]
    y = company_df["Layoff_Flag"]

    num_cols = ["workforce_impacted %", "Funds_Raised", "Year", "Quarter", "Month_Sin", "Month_Cos"]
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

    model = Pipeline(
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
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    _print_metrics("Company Level", y_test, y_pred, y_proba)

    company_df["Company_Risk_Score"] = model.predict_proba(X)[:, 1]

    return model, company_df


def train_employee_model() -> tuple[Pipeline, pd.DataFrame]:
    emp_df = pd.read_csv("Employee-levelData.csv")

    emp_df["Layoff_Risk"] = emp_df["LeaveOrNot"].astype(int)
    emp_df["Skill_Count"] = emp_df["Skills"].fillna("").astype(str).apply(lambda s: 0 if not s.strip() else len([t for t in s.split(",") if t.strip()]))

    features = [
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
        "Skills",
    ]

    X = emp_df[features]
    y = emp_df["Layoff_Risk"]

    num_cols = [
        "JoiningYear",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
        "Performance_Score",
        "Skill_Count",
    ]
    cat_cols = ["Education", "City", "Gender", "EverBenched"]
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
                cat_cols + ["Skills"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    _print_metrics("Employee Level", y_test, y_pred, y_proba)

    emp_df["Employee_Risk_Score"] = model.predict_proba(X)[:, 1]

    return model, emp_df


def main() -> None:
    company_model, company_scored_df = train_company_model()
    employee_model, employee_scored_df = train_employee_model()

    joblib.dump(company_model, "company_model.pkl")
    joblib.dump(employee_model, "employee_model.pkl")

    company_scored_df.to_csv("company_scored.csv", index=False)
    employee_scored_df.to_csv("employee_scored.csv", index=False)

    print("\nModels and scored datasets saved successfully.")


if __name__ == "__main__":
    main()
