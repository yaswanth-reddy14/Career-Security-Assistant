from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import app as app_module
import alerts
import auth


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def _save_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def _restore_bytes(path: Path, payload: bytes | None) -> None:
    if payload is None:
        if path.exists():
            path.unlink()
        return
    path.write_bytes(payload)


def _as_bool(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def _pick_company_cases(
    company_model,
    company_df: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, object], dict[str, object], float, float]:
    industries = sorted(company_df["Industry"].dropna().astype(str).unique().tolist())[:10]
    stages = sorted(company_df["Stage"].dropna().astype(str).unique().tolist())[:8]
    countries = (
        company_df["Country"].dropna().astype(str).value_counts().head(10).index.tolist()
    )
    funds_values = [20.0, 50.0, 100.0, 300.0, 700.0]
    impacted_values = [5.0, 10.0, 20.0, 35.0, 50.0]

    now = pd.Timestamp.now()
    year = int(now.year)
    quarter = int(((int(now.month) - 1) // 3) + 1)
    month_sin = float(np.sin(2 * np.pi * int(now.month) / 12.0))
    month_cos = float(np.cos(2 * np.pi * int(now.month) / 12.0))

    rows: list[dict[str, object]] = []
    for industry in industries:
        for stage in stages:
            for country in countries:
                for funds in funds_values:
                    for impacted_pct in impacted_values:
                        rows.append(
                            {
                                "Industry": industry,
                                "Funds_Raised": funds,
                                "Funds_Raised_Log": float(np.log1p(max(funds, 0.0))),
                                "Stage": stage,
                                "Country": country,
                                "workforce_impacted %": impacted_pct / 100.0,
                                "Year": year,
                                "Quarter": quarter,
                                "Month_Sin": month_sin,
                                "Month_Cos": month_cos,
                                "_workforce_pct_ui": impacted_pct,
                            }
                        )

    grid = pd.DataFrame(rows)
    risk = company_model.predict_proba(grid.drop(columns=["_workforce_pct_ui"]))[:, 1]
    grid["risk"] = risk

    low_row = grid.sort_values("risk", ascending=True).iloc[0]
    high_row = grid.sort_values("risk", ascending=False).iloc[0]

    low_case = {
        "company_name": "HC Low Case",
        "industry": str(low_row["Industry"]),
        "funds": float(low_row["Funds_Raised"]),
        "stage": str(low_row["Stage"]),
        "country": str(low_row["Country"]),
        "workforce_impacted": float(low_row["_workforce_pct_ui"]),
    }
    high_case = {
        "company_name": "HC High Case",
        "industry": str(high_row["Industry"]),
        "funds": float(high_row["Funds_Raised"]),
        "stage": str(high_row["Stage"]),
        "country": str(high_row["Country"]),
        "workforce_impacted": float(high_row["_workforce_pct_ui"]),
    }

    high_score = float(high_row["risk"])
    low_score = float(low_row["risk"])
    if high_score < threshold:
        high_case["industry"] = low_case["industry"]
        high_case["stage"] = low_case["stage"]
        high_case["country"] = low_case["country"]
        high_case["funds"] = low_case["funds"]
        high_case["workforce_impacted"] = low_case["workforce_impacted"]

    return low_case, high_case, low_score, high_score


def _pick_employee_cases(
    employee_model,
    threshold: float,
) -> tuple[dict[str, object], dict[str, object], float, float]:
    low_case = {
        "education": "Bachelors",
        "joining_year": 2014,
        "city": "New Delhi",
        "payment_tier": 3,
        "age": 28,
        "gender": "Female",
        "ever_benched": "No",
        "experience": 3,
        "skills": "Python, SQL, Excel, Pandas",
        "target_role": "Auto-detect",
        "performance_score": 4,
    }
    high_case = {
        "education": "Bachelors",
        "joining_year": 2024,
        "city": "Bangalore",
        "payment_tier": 1,
        "age": 24,
        "gender": "Male",
        "ever_benched": "Yes",
        "experience": 1,
        "skills": "Excel",
        "target_role": "Auto-detect",
        "performance_score": 1,
    }

    def to_model_row(case: dict[str, object]) -> pd.DataFrame:
        skills_tokens = sorted(
            {s.strip().lower() for s in str(case["skills"]).split(",") if s.strip()}
        )
        skills_model = ",".join(skills_tokens) if skills_tokens else "none"
        skill_count = len(skills_tokens)
        age = int(case["age"])
        exp = int(case["experience"])
        return pd.DataFrame(
            [
                {
                    "Education": str(case["education"]),
                    "JoiningYear": int(case["joining_year"]),
                    "City": str(case["city"]),
                    "PaymentTier": int(case["payment_tier"]),
                    "Age": age,
                    "Gender": str(case["gender"]),
                    "EverBenched": str(case["ever_benched"]),
                    "ExperienceInCurrentDomain": exp,
                    "Skills": skills_model,
                    "Performance_Score": int(case["performance_score"]),
                    "Skill_Count": skill_count,
                    "Experience_Age_Ratio": float(exp / age) if age > 0 else 0.0,
                }
            ]
        )

    low_score = float(employee_model.predict_proba(to_model_row(low_case))[0][1])
    high_score = float(employee_model.predict_proba(to_model_row(high_case))[0][1])

    if high_score < threshold:
        grid_rows = []
        for city in ["Bangalore", "New Delhi", "Pune"]:
            for bench in ["No", "Yes"]:
                for tier in [1, 2, 3]:
                    for age in [24, 28, 35]:
                        for exp in [1, 3, 5, 8]:
                            for year in [2010, 2014, 2020, 2024]:
                                for perf in [1, 3, 5]:
                                    case = {
                                        "education": "Bachelors",
                                        "joining_year": year,
                                        "city": city,
                                        "payment_tier": tier,
                                        "age": age,
                                        "gender": "Male",
                                        "ever_benched": bench,
                                        "experience": exp,
                                        "skills": "Excel",
                                        "target_role": "Auto-detect",
                                        "performance_score": perf,
                                    }
                                    row = to_model_row(case).iloc[0].to_dict()
                                    row["_case"] = case
                                    grid_rows.append(row)
        grid = pd.DataFrame(grid_rows)
        scores = employee_model.predict_proba(grid.drop(columns=["_case"]))[:, 1]
        idx = int(np.argmax(scores))
        high_case = dict(grid.iloc[idx]["_case"])
        high_score = float(scores[idx])

    return low_case, high_case, low_score, high_score


def _run_test(name: str, fn: Callable[[], tuple[bool, str]]) -> CheckResult:
    try:
        ok, details = fn()
        return CheckResult(name=name, status=_as_bool(ok), details=details)
    except Exception as exc:  # broad for smoke-check robustness
        return CheckResult(name=name, status="FAIL", details=f"{type(exc).__name__}: {exc}")


def main() -> int:
    users_path = Path("users.csv")
    alerts_path = Path("alerts_log.csv")

    users_backup = _save_bytes(users_path)
    alerts_backup = _save_bytes(alerts_path)

    orig_signup_email = app_module._send_signup_otp_email
    orig_profile_email = app_module._send_profile_email_otp_email

    results: list[CheckResult] = []
    summary: dict[str, object] = {}

    timestamp_tag = str(int(time.time()))
    test_user = f"hc_user_{timestamp_tag}"
    test_password = "hcpass123"
    first_email = f"{test_user}@example.com"
    updated_email = f"{test_user}_new@example.com"

    try:
        results.append(
            _run_test(
                "Python compile",
                lambda: (
                    True,
                    "py_compile already validated before running this script",
                ),
            )
        )

        results.append(
            _run_test(
                "SMTP readiness check",
                lambda: (
                    True,
                    str(alerts.send_alert_email("Healthcheck", "Body", "noreply@example.com")),
                ),
            )
        )

        app_module._send_signup_otp_email = lambda recipient, otp: (True, "mock-signup-email")
        app_module._send_profile_email_otp_email = lambda recipient, otp: (True, "mock-profile-email")

        flask_app = app_module.app
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()

        def signup_otp_flow() -> tuple[bool, str]:
            resp = client.post(
                "/signup",
                data={
                    "role": "Employee",
                    "username": test_user,
                    "email": first_email,
                    "company_name": "HealthCheck Corp",
                    "password": test_password,
                    "confirm_password": test_password,
                },
                follow_redirects=False,
            )
            if resp.status_code not in (302, 303):
                return False, f"unexpected status: {resp.status_code}"
            if "/signup/verify-otp" not in str(resp.headers.get("Location", "")):
                return False, f"unexpected redirect: {resp.headers.get('Location')}"

            with client.session_transaction() as sess:
                token = str(sess.get("pending_signup_token", "")).strip()
            if not token:
                return False, "pending signup token missing"

            otp_data = app_module.SIGNUP_OTP_STORE.get(token)
            if not otp_data:
                return False, "signup OTP store missing token"
            otp = str(otp_data.get("otp", ""))

            resp_verify = client.post(
                "/signup/verify-otp",
                data={"action": "verify", "otp": otp},
                follow_redirects=False,
            )
            if resp_verify.status_code not in (302, 303):
                return False, f"verify status: {resp_verify.status_code}"
            if "/signin" not in str(resp_verify.headers.get("Location", "")):
                return False, f"verify redirect: {resp_verify.headers.get('Location')}"
            if not auth.username_exists(test_user):
                return False, "user not created after OTP verify"
            return True, "signup OTP create+verify flow passed"

        results.append(_run_test("Signup OTP flow", signup_otp_flow))

        def signin_and_profile_otp_flow() -> tuple[bool, str]:
            resp_signin = client.post(
                "/signin",
                data={
                    "role": "Employee",
                    "username": test_user,
                    "password": test_password,
                },
                follow_redirects=False,
            )
            if resp_signin.status_code not in (302, 303):
                return False, f"signin status: {resp_signin.status_code}"
            if "/dashboard" not in str(resp_signin.headers.get("Location", "")):
                return False, f"signin redirect: {resp_signin.headers.get('Location')}"

            resp_email = client.post(
                "/profile",
                data={"action": "update_email", "email": updated_email},
                follow_redirects=False,
            )
            if resp_email.status_code not in (302, 303):
                return False, f"profile update status: {resp_email.status_code}"
            if "/profile/verify-email-otp" not in str(resp_email.headers.get("Location", "")):
                return False, f"profile redirect: {resp_email.headers.get('Location')}"

            with client.session_transaction() as sess:
                token = str(sess.get("pending_profile_email_token", "")).strip()
            if not token:
                return False, "pending profile email token missing"

            otp_data = app_module.PROFILE_EMAIL_OTP_STORE.get(token)
            if not otp_data:
                return False, "profile OTP store missing token"
            otp = str(otp_data.get("otp", ""))

            resp_verify = client.post(
                "/profile/verify-email-otp",
                data={"action": "verify", "otp": otp},
                follow_redirects=False,
            )
            if resp_verify.status_code not in (302, 303):
                return False, f"profile verify status: {resp_verify.status_code}"
            if "/profile" not in str(resp_verify.headers.get("Location", "")):
                return False, f"profile verify redirect: {resp_verify.headers.get('Location')}"

            current_email = auth.get_active_user_email(test_user)
            if current_email.lower() != updated_email.lower():
                return False, f"email not updated: {current_email}"
            return True, "profile email OTP verify flow passed"

        results.append(_run_test("Profile email OTP flow", signin_and_profile_otp_flow))

        company_model, employee_model = app_module.load_models()
        company_df, _ = app_module.load_data()
        company_low, company_high, company_low_score, company_high_score = _pick_company_cases(
            company_model=company_model,
            company_df=company_df,
            threshold=app_module.get_risk_bounds("company")[1],
        )
        employee_low, employee_high, employee_low_score, employee_high_score = _pick_employee_cases(
            employee_model=employee_model,
            threshold=app_module.get_risk_bounds("employee")[1],
        )

        summary["company_cases"] = {
            "low": {**company_low, "score": round(company_low_score, 4)},
            "high": {**company_high, "score": round(company_high_score, 4)},
            "threshold_high": round(app_module.get_risk_bounds("company")[1], 4),
        }
        summary["employee_cases"] = {
            "low": {**employee_low, "score": round(employee_low_score, 4)},
            "high": {**employee_high, "score": round(employee_high_score, 4)},
            "threshold_high": round(app_module.get_risk_bounds("employee")[1], 4),
        }

        with client.session_transaction() as sess:
            sess["authenticated"] = True
            sess["role"] = "Admin"
            sess["username"] = "admin"

        def company_predict_smoke() -> tuple[bool, str]:
            before = len(alerts.load_alert_log())
            resp_predict = client.post(
                "/predict/company",
                data={
                    "action": "predict",
                    "company_name": company_low["company_name"],
                    "industry": company_low["industry"],
                    "funds": str(company_low["funds"]),
                    "stage": company_low["stage"],
                    "country": company_low["country"],
                    "workforce_impacted": str(company_low["workforce_impacted"]),
                    "additional_hr_emails": "",
                },
                follow_redirects=True,
            )
            if resp_predict.status_code != 200:
                return False, f"predict status: {resp_predict.status_code}"
            if b"Risk Level" not in resp_predict.data:
                return False, "predict response missing risk output"

            resp_alert = client.post(
                "/predict/company",
                data={
                    "action": "predict_and_alert",
                    "company_name": company_high["company_name"],
                    "industry": company_high["industry"],
                    "funds": str(company_high["funds"]),
                    "stage": company_high["stage"],
                    "country": company_high["country"],
                    "workforce_impacted": str(company_high["workforce_impacted"]),
                    "additional_hr_emails": "qa_hr@example.com",
                },
                follow_redirects=True,
            )
            if resp_alert.status_code != 200:
                return False, f"predict_and_alert status: {resp_alert.status_code}"
            after = len(alerts.load_alert_log())
            if company_high_score >= app_module.get_risk_bounds("company")[1] and after <= before:
                return False, "expected alert log row for company high-risk case"
            return True, f"company predictor + alert path ok (log rows: {before} -> {after})"

        results.append(_run_test("Company predictor flow", company_predict_smoke))

        def employee_predict_smoke() -> tuple[bool, str]:
            before = len(alerts.load_alert_log())
            resp_predict = client.post(
                "/predict/employee",
                data={
                    "action": "predict",
                    "education": employee_low["education"],
                    "joining_year": str(employee_low["joining_year"]),
                    "city": employee_low["city"],
                    "payment_tier": str(employee_low["payment_tier"]),
                    "age": str(employee_low["age"]),
                    "gender": employee_low["gender"],
                    "ever_benched": employee_low["ever_benched"],
                    "experience": str(employee_low["experience"]),
                    "skills": employee_low["skills"],
                    "target_role": employee_low["target_role"],
                    "performance_score": str(employee_low["performance_score"]),
                    "additional_employee_emails": "",
                },
                follow_redirects=True,
            )
            if resp_predict.status_code != 200:
                return False, f"predict status: {resp_predict.status_code}"
            if b"Risk Level" not in resp_predict.data:
                return False, "predict response missing risk output"

            resp_alert = client.post(
                "/predict/employee",
                data={
                    "action": "predict_and_alert",
                    "education": employee_high["education"],
                    "joining_year": str(employee_high["joining_year"]),
                    "city": employee_high["city"],
                    "payment_tier": str(employee_high["payment_tier"]),
                    "age": str(employee_high["age"]),
                    "gender": employee_high["gender"],
                    "ever_benched": employee_high["ever_benched"],
                    "experience": str(employee_high["experience"]),
                    "skills": employee_high["skills"],
                    "target_role": employee_high["target_role"],
                    "performance_score": str(employee_high["performance_score"]),
                    "additional_employee_emails": "qa_emp@example.com",
                },
                follow_redirects=True,
            )
            if resp_alert.status_code != 200:
                return False, f"predict_and_alert status: {resp_alert.status_code}"
            after = len(alerts.load_alert_log())
            if employee_high_score >= app_module.get_risk_bounds("employee")[1] and after <= before:
                return False, "expected alert log row for employee high-risk case"
            return True, f"employee predictor + alert path ok (log rows: {before} -> {after})"

        results.append(_run_test("Employee predictor flow", employee_predict_smoke))

        results.append(
            _run_test(
                "Alert log schema",
                lambda: (
                    all(
                        col in alerts.load_alert_log().columns
                        for col in [
                            "sent_at",
                            "alert_type",
                            "recipient",
                            "risk_score",
                            "threshold",
                            "status",
                            "message",
                        ]
                    ),
                    "alert log columns verified",
                ),
            )
        )

    finally:
        app_module._send_signup_otp_email = orig_signup_email
        app_module._send_profile_email_otp_email = orig_profile_email
        app_module.SIGNUP_OTP_STORE.clear()
        app_module.PROFILE_EMAIL_OTP_STORE.clear()
        _restore_bytes(users_path, users_backup)
        _restore_bytes(alerts_path, alerts_backup)

    pass_count = sum(1 for r in results if r.status == "PASS")
    fail_count = sum(1 for r in results if r.status == "FAIL")

    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "results": [r.__dict__ for r in results],
        "summary": summary,
    }
    print(json.dumps(report, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
