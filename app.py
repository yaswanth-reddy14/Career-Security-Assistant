from __future__ import annotations

import difflib
import json
import os
import secrets
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, session, url_for

from alerts import send_alert_email, trigger_alert
from auth import (
    authenticate_user,
    create_user,
    ensure_default_admin,
    get_active_user_email,
    get_active_user_profile,
    is_valid_email,
    list_active_emails_by_role,
    update_active_user_email,
    update_active_user_password,
    username_exists,
)

load_dotenv(override=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

RISK_CONFIG_PATH = Path("risk_thresholds.json")
DEFAULT_RISK_BOUNDS = {"low_max": 0.40, "high_min": 0.70}
BALANCED_THRESHOLD_HIGH_SHARE = 0.45
BALANCED_LOW_QUANTILE = 0.20
BALANCED_HIGH_QUANTILE = 0.80
SCORED_FILE_MAP: dict[str, tuple[str, str]] = {
    "company": ("company_scored.csv", "Company_Risk_Score"),
    "employee": ("employee_scored.csv", "Employee_Risk_Score"),
}


ROLE_SKILL_MAP: dict[str, list[str]] = {
    "Data Analyst": ["sql", "python", "power bi", "tableau", "excel", "statistics"],
    "Data Scientist": ["python", "machine learning", "deep learning", "sql", "pandas", "numpy"],
    "Backend Developer": ["python", "java", "sql", "api", "microservices", "docker"],
    "Frontend Developer": ["javascript", "react", "html", "css", "typescript", "ui/ux"],
    "Full Stack Developer": ["javascript", "react", "node.js", "sql", "api", "docker"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "ci/cd", "linux", "terraform"],
}

EXTRA_COMPANY_INDUSTRIES: list[str] = [
    "Information Technology (IT)",
    "Consulting",
    "Business Process Services (BPS)",
    "Software",
    "FinTech",
    "Healthcare",
    "Pharmaceuticals",
    "Manufacturing",
    "Telecommunications",
    "E-commerce",
    "Banking",
    "Insurance",
    "Education",
    "Logistics",
    "Energy",
    "Automotive",
    "Media",
    "Travel",
    "Real Estate",
]

SIGNUP_OTP_EXPIRY_MINUTES = max(1, int(os.getenv("SIGNUP_OTP_EXPIRY_MINUTES", "10")))
SIGNUP_OTP_MAX_ATTEMPTS = max(1, int(os.getenv("SIGNUP_OTP_MAX_ATTEMPTS", "5")))
SIGNUP_OTP_STORE: dict[str, dict[str, object]] = {}
PROFILE_EMAIL_OTP_EXPIRY_MINUTES = max(1, int(os.getenv("PROFILE_EMAIL_OTP_EXPIRY_MINUTES", "10")))
PROFILE_EMAIL_OTP_MAX_ATTEMPTS = max(1, int(os.getenv("PROFILE_EMAIL_OTP_MAX_ATTEMPTS", "5")))
PROFILE_EMAIL_OTP_STORE: dict[str, dict[str, object]] = {}


def _cleanup_signup_otp_store() -> None:
    now = datetime.now()
    expired_tokens = [
        token
        for token, data in SIGNUP_OTP_STORE.items()
        if not isinstance(data.get("expires_at"), datetime) or now > data["expires_at"]
    ]
    for token in expired_tokens:
        SIGNUP_OTP_STORE.pop(token, None)


def _generate_signup_otp() -> str:
    return f"{secrets.randbelow(1000000):06d}"


def _send_signup_otp_email(recipient: str, otp: str) -> tuple[bool, str]:
    subject = "Career Security Assistant - Signup OTP Verification"
    body = (
        "Use this OTP to complete your signup.\n\n"
        f"OTP: {otp}\n"
        f"Expires in: {SIGNUP_OTP_EXPIRY_MINUTES} minute(s)\n\n"
        "If you did not request this, ignore this email."
    )
    return send_alert_email(subject=subject, body=body, recipient=recipient)


def _start_signup_otp_session(
    role: str,
    username: str,
    email: str,
    company_name: str,
    password: str,
) -> tuple[bool, str, str | None]:
    _cleanup_signup_otp_store()
    token = secrets.token_urlsafe(24)
    otp = _generate_signup_otp()
    expires_at = datetime.now() + timedelta(minutes=SIGNUP_OTP_EXPIRY_MINUTES)

    SIGNUP_OTP_STORE[token] = {
        "role": role,
        "username": username,
        "email": email,
        "company_name": company_name,
        "password": password,
        "otp": otp,
        "expires_at": expires_at,
        "attempts": 0,
    }

    ok, msg = _send_signup_otp_email(recipient=email, otp=otp)
    if not ok:
        SIGNUP_OTP_STORE.pop(token, None)
        return False, f"Unable to send OTP email: {msg}", None
    return True, "OTP sent to your email. Enter it to finish signup.", token


def _cleanup_profile_email_otp_store() -> None:
    now = datetime.now()
    expired_tokens = [
        token
        for token, data in PROFILE_EMAIL_OTP_STORE.items()
        if not isinstance(data.get("expires_at"), datetime) or now > data["expires_at"]
    ]
    for token in expired_tokens:
        PROFILE_EMAIL_OTP_STORE.pop(token, None)


def _send_profile_email_otp_email(recipient: str, otp: str) -> tuple[bool, str]:
    subject = "Career Security Assistant - Email Change OTP Verification"
    body = (
        "Use this OTP to confirm your new profile email.\n\n"
        f"OTP: {otp}\n"
        f"Expires in: {PROFILE_EMAIL_OTP_EXPIRY_MINUTES} minute(s)\n\n"
        "If you did not request this, ignore this email."
    )
    return send_alert_email(subject=subject, body=body, recipient=recipient)


def _start_profile_email_otp_session(username: str, new_email: str) -> tuple[bool, str, str | None]:
    _cleanup_profile_email_otp_store()
    token = secrets.token_urlsafe(24)
    otp = _generate_signup_otp()
    expires_at = datetime.now() + timedelta(minutes=PROFILE_EMAIL_OTP_EXPIRY_MINUTES)

    PROFILE_EMAIL_OTP_STORE[token] = {
        "username": username,
        "new_email": new_email,
        "otp": otp,
        "expires_at": expires_at,
        "attempts": 0,
    }

    ok, msg = _send_profile_email_otp_email(recipient=new_email, otp=otp)
    if not ok:
        PROFILE_EMAIL_OTP_STORE.pop(token, None)
        return False, f"Unable to send OTP email: {msg}", None

    return True, "OTP sent to your new email. Verify to complete email update.", token


@lru_cache(maxsize=1)
def load_risk_config() -> dict[str, object]:
    if not RISK_CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(RISK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=2)
def load_balanced_risk_bounds(model_key: str) -> tuple[float, float] | None:
    score_file, score_col = SCORED_FILE_MAP.get(model_key, ("", ""))
    if not score_file or not score_col:
        return None

    score_path = Path(score_file)
    if not score_path.exists():
        return None

    try:
        scores = pd.read_csv(score_path, usecols=[score_col])[score_col].dropna().astype(float)
    except Exception:
        return None

    if scores.empty:
        return None

    low_max = float(scores.quantile(BALANCED_LOW_QUANTILE))
    high_min = float(scores.quantile(BALANCED_HIGH_QUANTILE))
    if not (0.0 <= low_max < high_min <= 1.0):
        return None
    return low_max, high_min


def get_risk_bounds(model_key: str) -> tuple[float, float]:
    cfg = load_risk_config()
    entry = cfg.get(model_key, {}) if isinstance(cfg, dict) else {}
    low_raw = entry.get("low_max", DEFAULT_RISK_BOUNDS["low_max"]) if isinstance(entry, dict) else DEFAULT_RISK_BOUNDS["low_max"]
    high_raw = entry.get("high_min", DEFAULT_RISK_BOUNDS["high_min"]) if isinstance(entry, dict) else DEFAULT_RISK_BOUNDS["high_min"]

    try:
        low_max = float(low_raw)
    except (TypeError, ValueError):
        low_max = DEFAULT_RISK_BOUNDS["low_max"]
    try:
        high_min = float(high_raw)
    except (TypeError, ValueError):
        high_min = DEFAULT_RISK_BOUNDS["high_min"]

    if not (0.0 <= low_max < high_min <= 1.0):
        return float(DEFAULT_RISK_BOUNDS["low_max"]), float(DEFAULT_RISK_BOUNDS["high_min"])

    # If configured high-risk coverage is too broad, rebalance using score quantiles.
    high_share_raw = entry.get("high_bucket_share") if isinstance(entry, dict) else None
    try:
        high_share = float(high_share_raw)
    except (TypeError, ValueError):
        high_share = None

    if high_share is not None and high_share > BALANCED_THRESHOLD_HIGH_SHARE:
        balanced = load_balanced_risk_bounds(model_key)
        if balanced is not None:
            return balanced

    return low_max, high_min


def risk_label(score: float, model_key: str = "company") -> str:
    low_max, high_min = get_risk_bounds(model_key)
    if score >= high_min:
        return "High"
    if score <= low_max:
        return "Low"
    return "Medium"


def suggest_industry_for_company(company_name: str, company_df: pd.DataFrame) -> tuple[str | None, str | None]:
    name = str(company_name).strip().lower()
    if not name or "Company" not in company_df.columns or "Industry" not in company_df.columns:
        return None, None

    pairs = company_df[["Company", "Industry"]].dropna().copy()
    if pairs.empty:
        return None, None

    pairs["Company_norm"] = pairs["Company"].astype(str).str.strip().str.lower()
    exact = pairs[pairs["Company_norm"] == name]
    if not exact.empty:
        return str(exact["Industry"].mode().iloc[0]), str(exact["Company"].iloc[0])

    company_names = pairs["Company_norm"].unique().tolist()
    close = difflib.get_close_matches(name, company_names, n=1, cutoff=0.75)
    if not close:
        return None, None

    matched_norm = close[0]
    matched = pairs[pairs["Company_norm"] == matched_norm]
    if matched.empty:
        return None, None

    return str(matched["Industry"].mode().iloc[0]), str(matched["Company"].iloc[0])


def company_cause_breakdown(
    funds: float,
    workforce_impacted_pct: float,
    stage: str,
    month: int,
    industry: str,
    risk: float,
    threshold: float,
) -> list[dict[str, object]]:
    stage_text = str(stage).lower()
    industry_text = str(industry).lower()
    return [
        {
            "cause": "Workforce impact pressure",
            "triggered": workforce_impacted_pct >= 20.0,
            "details": f"Observed workforce impacted: {workforce_impacted_pct:.1f}%. Trigger: >= 20.0%.",
        },
        {
            "cause": "Funding runway pressure",
            "triggered": funds <= 50.0,
            "details": f"Observed funds raised: {funds:.1f}M USD. Trigger: <= 50.0M USD.",
        },
        {
            "cause": "Early-stage volatility",
            "triggered": any(k in stage_text for k in ["seed", "pre-seed", "series a"]),
            "details": f"Observed company stage: '{stage}'. Trigger: seed/pre-seed/series A.",
        },
        {
            "cause": "Seasonal restructuring period",
            "triggered": month in (1, 2, 3, 10, 11, 12),
            "details": f"Observed month: {int(month)}. Trigger: Jan-Mar or Oct-Dec.",
        },
        {
            "cause": "Industry volatility signal",
            "triggered": any(k in industry_text for k in ["other", "consumer", "retail"]),
            "details": f"Observed industry: '{industry}'. Trigger includes other/consumer/retail.",
        },
        {
            "cause": "Aggregate model risk signal",
            "triggered": risk >= threshold,
            "details": f"Predicted risk score: {risk:.3f}. Trigger: score >= {threshold:.3f}.",
        },
    ]


def employee_cause_breakdown(
    performance_score: int,
    ever_benched: str,
    experience_years: int,
    payment_tier: int,
    skill_count: int,
    joining_year: int,
    risk: float,
    threshold: float,
) -> list[dict[str, object]]:
    return [
        {
            "cause": "Performance pressure",
            "triggered": int(performance_score) <= 2,
            "details": f"Observed performance score: {int(performance_score)}. Trigger: <= 2.",
        },
        {
            "cause": "Bench history signal",
            "triggered": str(ever_benched).strip().lower() == "yes",
            "details": f"Observed ever benched: '{ever_benched}'. Trigger: yes.",
        },
        {
            "cause": "Low domain experience",
            "triggered": int(experience_years) <= 2,
            "details": f"Observed domain experience: {int(experience_years)} years. Trigger: <= 2 years.",
        },
        {
            "cause": "Compensation-tier exposure",
            "triggered": int(payment_tier) <= 1,
            "details": f"Observed payment tier: {int(payment_tier)}. Trigger: <= 1.",
        },
        {
            "cause": "Limited skill breadth",
            "triggered": int(skill_count) <= 2,
            "details": f"Observed skill count: {int(skill_count)}. Trigger: <= 2.",
        },
        {
            "cause": "Recent joining exposure",
            "triggered": int(joining_year) >= 2023,
            "details": f"Observed joining year: {int(joining_year)}. Trigger: >= 2023.",
        },
        {
            "cause": "Aggregate model risk signal",
            "triggered": risk >= threshold,
            "details": f"Predicted risk score: {risk:.3f}. Trigger: score >= {threshold:.3f}.",
        },
    ]


def _normalize_skills(skills_text: str) -> set[str]:
    tokens = [s.strip().lower() for s in str(skills_text).split(",")]
    return {s for s in tokens if s}


def infer_role_from_skills(skills_text: str) -> str:
    current = _normalize_skills(skills_text)
    best_role = "Data Analyst"
    best_overlap = -1
    for role, role_skills in ROLE_SKILL_MAP.items():
        overlap = len(current.intersection({s.lower() for s in role_skills}))
        if overlap > best_overlap:
            best_overlap = overlap
            best_role = role
    return best_role


def recommend_skill_gaps(skills_text: str, role: str) -> list[str]:
    current = _normalize_skills(skills_text)
    required = [s.lower() for s in ROLE_SKILL_MAP.get(role, [])]
    missing = [s for s in required if s not in current]
    return missing[:3]


def _parse_email_list(raw: str) -> list[str]:
    emails = [item.strip() for item in str(raw).split(",") if item.strip()]
    return list(dict.fromkeys(emails))


def _flash_alert_delivery_summary(alert_label: str, sent: int, total: int, failed: list[str]) -> None:
    failed_count = len(failed)
    if total <= 0:
        flash(f"{alert_label} alert not sent because no recipients were provided.", "warning")
        return

    if sent == total:
        flash(f"{alert_label} alert sent to {sent}/{total} recipient(s).", "success")
    elif sent == 0:
        flash(f"{alert_label} alert failed for all {total} recipient(s).", "error")
    else:
        flash(
            f"{alert_label} alert partially sent: {sent}/{total} delivered, {failed_count} failed.",
            "warning",
        )

    for item in failed[:3]:
        flash(item, "warning")
    if failed_count > 3:
        flash(f"...and {failed_count - 3} more failed recipient(s).", "warning")


def _report_filename(prefix: str, name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
    cleaned = "_".join([token for token in cleaned.split("_") if token])
    if not cleaned:
        cleaned = "report"
    return f"{prefix}_{cleaned}.txt"


def build_company_report_text(
    company_name: str,
    generated_at: pd.Timestamp,
    risk: float,
    threshold: float,
    inputs: dict[str, object],
    causes: list[dict[str, object]],
) -> str:
    lines = [
        "Career Security Assistant - Company Layoff Risk Report",
        f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary",
        f"- Company: {company_name}",
        f"- Predicted risk score: {risk:.3f}",
        f"- Alert threshold: {threshold:.3f}",
        f"- Risk level: {risk_label(risk, 'company')}",
        "",
        "Input Data",
    ]
    for key, value in inputs.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Cause Analysis")
    for idx, cause in enumerate(causes, start=1):
        status = "Contributing" if bool(cause.get("triggered", False)) else "Not Contributing"
        lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')} [{status}]")
        lines.append(f"   {cause.get('details', '')}")

    return "\n".join(lines)


def build_employee_report_text(
    username: str,
    generated_at: pd.Timestamp,
    risk: float,
    threshold: float,
    inputs: dict[str, object],
    causes: list[dict[str, object]],
    recommended_role: str,
    recommended_skills: list[str],
) -> str:
    lines = [
        "Career Security Assistant - Employee Layoff Risk Report",
        f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Summary",
        f"- User: {username}",
        f"- Predicted risk score: {risk:.3f}",
        f"- Alert threshold: {threshold:.3f}",
        f"- Risk level: {risk_label(risk, 'employee')}",
        "",
        "Input Data",
    ]
    for key, value in inputs.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Cause Analysis")
    for idx, cause in enumerate(causes, start=1):
        status = "Contributing" if bool(cause.get("triggered", False)) else "Not Contributing"
        lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')} [{status}]")
        lines.append(f"   {cause.get('details', '')}")

    lines.append("")
    lines.append("Skill Guidance")
    lines.append(f"- Recommended role track: {recommended_role}")
    if recommended_skills:
        lines.append("- Top recommended skills: " + ", ".join([s.title() for s in recommended_skills]))
    else:
        lines.append("- Skill profile already aligns with selected role.")

    return "\n".join(lines)


def _is_authenticated() -> bool:
    return bool(session.get("authenticated", False))


def _role() -> str:
    return str(session.get("role", "")).strip()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not _is_authenticated():
            flash("Please sign in first.", "warning")
            return redirect(url_for("signin"))
        return view_func(*args, **kwargs)

    return wrapper


def roles_required(*roles: str):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if _role() not in roles:
                flash("You do not have access to this page.", "error")
                return redirect(url_for("dashboard"))
            return view_func(*args, **kwargs)

        return wrapper

    return decorator


@app.context_processor
def inject_auth_context():
    return {
        "is_authenticated": _is_authenticated(),
        "current_role": _role(),
        "current_username": str(session.get("username", "")).strip(),
    }


@lru_cache(maxsize=1)
def load_models():
    company_path = Path("company_model.pkl")
    employee_path = Path("employee_model.pkl")
    if not company_path.exists() or not employee_path.exists():
        missing = []
        if not company_path.exists():
            missing.append("company_model.pkl")
        if not employee_path.exists():
            missing.append("employee_model.pkl")
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(missing)
            + ". Run `python fyp.py` to train/generate model artifacts."
        )
    return joblib.load(company_path), joblib.load(employee_path)


@lru_cache(maxsize=1)
def load_data():
    return pd.read_csv("company-level.csv"), pd.read_csv("Employee-levelData.csv")


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    ensure_default_admin()
    if _is_authenticated():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        role = str(request.form.get("role", "")).strip()
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))

        ok, msg, user = authenticate_user(username, password, role=role)
        if ok and user is not None:
            session["authenticated"] = True
            session["role"] = str(user.get("role", role))
            session["username"] = str(user.get("username", username))
            flash("Signed in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash(msg, "error")

    return render_template("signin.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if _is_authenticated():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        role = str(request.form.get("role", "")).strip()
        username = str(request.form.get("username", "")).strip()
        email = str(request.form.get("email", "")).strip()
        company_name = str(request.form.get("company_name", "")).strip()
        password = str(request.form.get("password", ""))
        confirm_password = str(request.form.get("confirm_password", ""))

        if role not in {"HR", "Employee"}:
            flash("Invalid role", "error")
            return render_template("signup.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters", "error")
            return render_template("signup.html")
        if username_exists(username):
            flash("Username already exists", "error")
            return render_template("signup.html")
        if not email:
            flash("Email is required", "error")
            return render_template("signup.html")
        if not is_valid_email(email):
            flash("Please enter a valid email address", "error")
            return render_template("signup.html")
        if len(company_name) < 2:
            flash("Company name is required", "error")
            return render_template("signup.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters", "error")
            return render_template("signup.html")
        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html")

        ok, msg, token = _start_signup_otp_session(
            role=role,
            username=username,
            email=email,
            company_name=company_name,
            password=password,
        )
        if ok and token is not None:
            session["pending_signup_token"] = token
            flash(msg, "success")
            return redirect(url_for("signup_verify_otp"))
        flash(msg, "error")

    return render_template("signup.html")


@app.route("/signup/verify-otp", methods=["GET", "POST"])
def signup_verify_otp():
    if _is_authenticated():
        return redirect(url_for("dashboard"))

    _cleanup_signup_otp_store()
    token = str(session.get("pending_signup_token", "")).strip()
    pending = SIGNUP_OTP_STORE.get(token)
    if not token or pending is None:
        session.pop("pending_signup_token", None)
        flash("Signup session expired. Please start signup again.", "warning")
        return redirect(url_for("signup"))

    if request.method == "POST":
        action = str(request.form.get("action", "verify")).strip()

        if action == "resend":
            otp = _generate_signup_otp()
            pending["otp"] = otp
            pending["attempts"] = 0
            pending["expires_at"] = datetime.now() + timedelta(minutes=SIGNUP_OTP_EXPIRY_MINUTES)

            ok, msg = _send_signup_otp_email(recipient=str(pending["email"]), otp=otp)
            if ok:
                flash("A new OTP has been sent to your email.", "success")
            else:
                flash(f"Unable to resend OTP email: {msg}", "error")
            return redirect(url_for("signup_verify_otp"))

        entered_otp = str(request.form.get("otp", "")).strip()
        expires_at = pending.get("expires_at")

        if not entered_otp.isdigit() or len(entered_otp) != 6:
            flash("Enter a valid 6-digit OTP.", "error")
            return redirect(url_for("signup_verify_otp"))
        if not isinstance(expires_at, datetime) or datetime.now() > expires_at:
            flash("OTP expired. Please click Resend OTP.", "warning")
            return redirect(url_for("signup_verify_otp"))

        attempts = int(pending.get("attempts", 0)) + 1
        pending["attempts"] = attempts
        if attempts > SIGNUP_OTP_MAX_ATTEMPTS:
            SIGNUP_OTP_STORE.pop(token, None)
            session.pop("pending_signup_token", None)
            flash("Too many incorrect OTP attempts. Please sign up again.", "error")
            return redirect(url_for("signup"))

        if entered_otp != str(pending.get("otp", "")):
            remaining = max(0, SIGNUP_OTP_MAX_ATTEMPTS - attempts)
            flash(f"Invalid OTP. {remaining} attempt(s) left.", "error")
            return redirect(url_for("signup_verify_otp"))

        username = str(pending.get("username", "")).strip()
        if username_exists(username):
            SIGNUP_OTP_STORE.pop(token, None)
            session.pop("pending_signup_token", None)
            flash("Username already exists. Please sign up with a different username.", "error")
            return redirect(url_for("signup"))

        ok, msg = create_user(
            username=username,
            password=str(pending.get("password", "")),
            role=str(pending.get("role", "")),
            email=str(pending.get("email", "")),
            company_name=str(pending.get("company_name", "")),
        )

        SIGNUP_OTP_STORE.pop(token, None)
        session.pop("pending_signup_token", None)

        if ok:
            flash("Account created successfully. Please sign in.", "success")
            return redirect(url_for("signin"))

        flash(msg, "error")
        return redirect(url_for("signup"))

    expires_at = pending.get("expires_at")
    seconds_left = 0
    if isinstance(expires_at, datetime):
        seconds_left = max(0, int((expires_at - datetime.now()).total_seconds()))
    minutes_left = max(1, (seconds_left + 59) // 60)

    return render_template(
        "verify_signup_otp.html",
        email=str(pending.get("email", "")),
        minutes_left=minutes_left,
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("landing"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    username = str(session.get("username", "")).strip()
    if not username:
        flash("Session is missing user details. Please sign in again.", "error")
        return redirect(url_for("signin"))

    if request.method == "POST":
        action = str(request.form.get("action", "")).strip()

        if action == "update_email":
            new_email = str(request.form.get("email", "")).strip()
            if not new_email:
                flash("Email is required", "error")
            elif not is_valid_email(new_email):
                flash("Please enter a valid email address", "error")
            else:
                profile_data = get_active_user_profile(username)
                if profile_data is None:
                    flash("Unable to load profile details.", "error")
                else:
                    current_email = str(profile_data.get("email", "")).strip().lower()
                    if new_email.lower() == current_email:
                        flash("New email is the same as current email.", "warning")
                    else:
                        ok, msg, token = _start_profile_email_otp_session(
                            username=username,
                            new_email=new_email,
                        )
                        if ok and token is not None:
                            session["pending_profile_email_token"] = token
                            flash(msg, "success")
                            return redirect(url_for("profile_verify_email_otp"))
                        flash(msg, "error")
        elif action == "update_password":
            current_password = str(request.form.get("current_password", ""))
            new_password = str(request.form.get("new_password", ""))
            confirm_password = str(request.form.get("confirm_password", ""))

            if new_password != confirm_password:
                flash("New password and confirm password do not match.", "error")
            else:
                ok, msg = update_active_user_password(
                    username=username,
                    current_password=current_password,
                    new_password=new_password,
                )
                flash(msg, "success" if ok else "error")
        else:
            flash("Invalid profile action.", "error")

        return redirect(url_for("profile"))

    profile_data = get_active_user_profile(username)
    if profile_data is None:
        flash("Unable to load profile details.", "error")
        return redirect(url_for("dashboard"))

    return render_template("profile.html", profile=profile_data)


@app.route("/profile/verify-email-otp", methods=["GET", "POST"])
@login_required
def profile_verify_email_otp():
    _cleanup_profile_email_otp_store()

    username = str(session.get("username", "")).strip()
    token = str(session.get("pending_profile_email_token", "")).strip()
    pending = PROFILE_EMAIL_OTP_STORE.get(token)

    if not username:
        session.pop("pending_profile_email_token", None)
        flash("Session is missing user details. Please sign in again.", "error")
        return redirect(url_for("signin"))

    if not token or pending is None or str(pending.get("username", "")).strip() != username:
        session.pop("pending_profile_email_token", None)
        flash("Email verification session expired. Please try again from profile.", "warning")
        return redirect(url_for("profile"))

    if request.method == "POST":
        action = str(request.form.get("action", "verify")).strip()

        if action == "cancel":
            PROFILE_EMAIL_OTP_STORE.pop(token, None)
            session.pop("pending_profile_email_token", None)
            flash("Email update verification canceled.", "warning")
            return redirect(url_for("profile"))

        if action == "resend":
            otp = _generate_signup_otp()
            pending["otp"] = otp
            pending["attempts"] = 0
            pending["expires_at"] = datetime.now() + timedelta(minutes=PROFILE_EMAIL_OTP_EXPIRY_MINUTES)

            ok, msg = _send_profile_email_otp_email(recipient=str(pending.get("new_email", "")), otp=otp)
            if ok:
                flash("A new OTP has been sent to your new email.", "success")
            else:
                flash(f"Unable to resend OTP email: {msg}", "error")
            return redirect(url_for("profile_verify_email_otp"))

        entered_otp = str(request.form.get("otp", "")).strip()
        expires_at = pending.get("expires_at")

        if not entered_otp.isdigit() or len(entered_otp) != 6:
            flash("Enter a valid 6-digit OTP.", "error")
            return redirect(url_for("profile_verify_email_otp"))
        if not isinstance(expires_at, datetime) or datetime.now() > expires_at:
            flash("OTP expired. Please click Resend OTP.", "warning")
            return redirect(url_for("profile_verify_email_otp"))

        attempts = int(pending.get("attempts", 0)) + 1
        pending["attempts"] = attempts
        if attempts > PROFILE_EMAIL_OTP_MAX_ATTEMPTS:
            PROFILE_EMAIL_OTP_STORE.pop(token, None)
            session.pop("pending_profile_email_token", None)
            flash("Too many incorrect OTP attempts. Please retry email update from profile.", "error")
            return redirect(url_for("profile"))

        if entered_otp != str(pending.get("otp", "")):
            remaining = max(0, PROFILE_EMAIL_OTP_MAX_ATTEMPTS - attempts)
            flash(f"Invalid OTP. {remaining} attempt(s) left.", "error")
            return redirect(url_for("profile_verify_email_otp"))

        ok, msg = update_active_user_email(
            username=username,
            email=str(pending.get("new_email", "")).strip(),
        )
        PROFILE_EMAIL_OTP_STORE.pop(token, None)
        session.pop("pending_profile_email_token", None)
        flash(msg, "success" if ok else "error")
        return redirect(url_for("profile"))

    expires_at = pending.get("expires_at")
    seconds_left = 0
    if isinstance(expires_at, datetime):
        seconds_left = max(0, int((expires_at - datetime.now()).total_seconds()))
    minutes_left = max(1, (seconds_left + 59) // 60)

    return render_template(
        "verify_profile_email_otp.html",
        email=str(pending.get("new_email", "")),
        minutes_left=minutes_left,
    )


@app.route("/predict/company", methods=["GET", "POST"])
@login_required
@roles_required("Admin", "HR")
def company_predictor():
    company_model, _ = load_models()
    company_df, _ = load_data()

    dataset_industries = company_df["Industry"].dropna().astype(str).unique().tolist()
    industry_options = sorted(set(dataset_industries + EXTRA_COMPANY_INDUSTRIES))
    stage_options = sorted(company_df["Stage"].dropna().astype(str).unique().tolist())
    country_options = sorted(company_df["Country"].dropna().astype(str).unique().tolist())

    result = None
    suggested = None
    matched_company = None

    form_values = {
        "company_name": str(request.form.get("company_name", "Demo Company")),
        "industry": str(request.form.get("industry", industry_options[0] if industry_options else "")),
        "funds": str(request.form.get("funds", "100")),
        "stage": str(request.form.get("stage", stage_options[0] if stage_options else "")),
        "country": str(request.form.get("country", country_options[0] if country_options else "")),
        "workforce_impacted": str(request.form.get("workforce_impacted", "10")),
        "additional_hr_emails": str(request.form.get("additional_hr_emails", "")),
    }

    if request.method == "POST":
        try:
            action = str(request.form.get("action", "predict"))
            send_alert_requested = action == "predict_and_alert"
            company_name = form_values["company_name"].strip()
            industry = form_values["industry"].strip()
            funds = float(form_values["funds"])
            stage = form_values["stage"].strip()
            country = form_values["country"].strip()
            workforce_impacted = float(form_values["workforce_impacted"])

            suggested, matched_company = suggest_industry_for_company(company_name, company_df)
            if suggested and suggested in industry_options and not request.form.get("industry"):
                industry = suggested

            now = pd.Timestamp.now()
            funds_non_negative = max(funds, 0.0)
            input_data = pd.DataFrame(
                [
                    {
                        "Industry": industry,
                        "Funds_Raised": funds,
                        "Funds_Raised_Log": float(np.log1p(funds_non_negative)),
                        "Stage": stage,
                        "Country": country,
                        "workforce_impacted %": workforce_impacted / 100.0,
                        "Year": int(now.year),
                        "Quarter": int(((int(now.month) - 1) // 3) + 1),
                        "Month_Sin": float(np.sin(2 * np.pi * int(now.month) / 12.0)),
                        "Month_Cos": float(np.cos(2 * np.pi * int(now.month) / 12.0)),
                    }
                ]
            )

            _, threshold = get_risk_bounds("company")
            risk = float(company_model.predict_proba(input_data)[0][1])
            causes = company_cause_breakdown(
                funds=funds,
                workforce_impacted_pct=workforce_impacted,
                stage=stage,
                month=int(now.month),
                industry=industry,
                risk=risk,
                threshold=threshold,
            )

            result = {
                "company_name": company_name,
                "risk_score": round(risk, 3),
                "risk_level": risk_label(risk, "company"),
                "causes": causes,
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if send_alert_requested:
                if risk < threshold:
                    flash(
                        f"Alert not sent because risk {risk:.3f} is below threshold {threshold:.3f}.",
                        "warning",
                    )
                else:
                    auto_hr_recipients = list_active_emails_by_role("HR")
                    manual_recipients = _parse_email_list(form_values["additional_hr_emails"])
                    recipients = list(dict.fromkeys(auto_hr_recipients + manual_recipients))

                    if not recipients:
                        flash(
                            "No HR recipient email available. Add HR user emails or provide additional HR email(s).",
                            "warning",
                        )
                    else:
                        company_report_text = build_company_report_text(
                            company_name=company_name,
                            generated_at=now,
                            risk=risk,
                            threshold=threshold,
                            inputs={
                                "Industry": industry,
                                "Funds Raised (USD millions)": f"{funds:.1f}",
                                "Company Stage": stage,
                                "Country": country,
                                "Workforce impacted %": f"{workforce_impacted:.1f}",
                                "Year": int(now.year),
                                "Quarter": int(((int(now.month) - 1) // 3) + 1),
                                "Month": int(now.month),
                            },
                            causes=causes,
                        )
                        report_filename = _report_filename("company_risk_report", company_name)

                        sent = 0
                        failed: list[str] = []
                        for recipient in recipients:
                            ok, msg = trigger_alert(
                                alert_type="COMPANY",
                                subject=f"High Company Layoff Risk Alert: {company_name}",
                                body=company_report_text,
                                recipient=recipient,
                                risk_score=risk,
                                threshold=threshold,
                                report_text=company_report_text,
                                report_filename=report_filename,
                                cooldown_minutes=0,
                            )
                            if ok:
                                sent += 1
                            else:
                                failed.append(f"{recipient}: {msg}")

                        _flash_alert_delivery_summary(
                            alert_label="Company",
                            sent=sent,
                            total=len(recipients),
                            failed=failed,
                        )
        except Exception as exc:
            flash(f"Unable to run company prediction: {exc}", "error")

    return render_template(
        "company_predictor.html",
        industry_options=industry_options,
        stage_options=stage_options,
        country_options=country_options,
        form_values=form_values,
        result=result,
        suggested=suggested,
        matched_company=matched_company,
    )


@app.route("/predict/employee", methods=["GET", "POST"])
@login_required
@roles_required("Admin", "HR", "Employee")
def employee_predictor():
    _, employee_model = load_models()
    _, emp_df = load_data()

    edu_options = sorted(emp_df["Education"].dropna().astype(str).unique().tolist())
    city_options = sorted(emp_df["City"].dropna().astype(str).unique().tolist())
    gender_options = sorted(emp_df["Gender"].dropna().astype(str).unique().tolist())
    bench_options = sorted(emp_df["EverBenched"].dropna().astype(str).unique().tolist())
    role_options = ["Auto-detect"] + list(ROLE_SKILL_MAP.keys())

    form_values = {
        "education": str(request.form.get("education", edu_options[0] if edu_options else "")),
        "joining_year": str(request.form.get("joining_year", "2019")),
        "city": str(request.form.get("city", city_options[0] if city_options else "")),
        "payment_tier": str(request.form.get("payment_tier", "2")),
        "age": str(request.form.get("age", "28")),
        "gender": str(request.form.get("gender", gender_options[0] if gender_options else "")),
        "ever_benched": str(request.form.get("ever_benched", bench_options[0] if bench_options else "")),
        "experience": str(request.form.get("experience", "3")),
        "skills": str(request.form.get("skills", "Python, SQL")),
        "target_role": str(request.form.get("target_role", "Auto-detect")),
        "performance_score": str(request.form.get("performance_score", "3")),
        "additional_employee_emails": str(request.form.get("additional_employee_emails", "")),
    }

    result = None
    if request.method == "POST":
        try:
            action = str(request.form.get("action", "predict"))
            send_alert_requested = action == "predict_and_alert"
            education = form_values["education"].strip()
            joining_year = int(form_values["joining_year"])
            city = form_values["city"].strip()
            payment_tier = int(form_values["payment_tier"])
            age = int(form_values["age"])
            gender = form_values["gender"].strip()
            ever_benched = form_values["ever_benched"].strip()
            experience = int(form_values["experience"])
            skills = form_values["skills"].strip()
            target_role = form_values["target_role"].strip()
            performance_score = int(form_values["performance_score"])

            normalized_skill_tokens = sorted(_normalize_skills(skills))
            model_skills = ",".join(normalized_skill_tokens) if normalized_skill_tokens else "none"
            skill_count = len(normalized_skill_tokens)
            experience_age_ratio = float(experience / age) if age > 0 else 0.0
            input_data = pd.DataFrame(
                [
                    {
                        "Education": education,
                        "JoiningYear": joining_year,
                        "City": city,
                        "PaymentTier": payment_tier,
                        "Age": age,
                        "Gender": gender,
                        "EverBenched": ever_benched,
                        "ExperienceInCurrentDomain": experience,
                        "Skills": model_skills,
                        "Performance_Score": performance_score,
                        "Skill_Count": skill_count,
                        "Experience_Age_Ratio": experience_age_ratio,
                    }
                ]
            )

            _, threshold = get_risk_bounds("employee")
            risk = float(employee_model.predict_proba(input_data)[0][1])
            selected_role = infer_role_from_skills(skills) if target_role == "Auto-detect" else target_role
            recommended_skills = recommend_skill_gaps(skills, selected_role)
            causes = employee_cause_breakdown(
                performance_score=performance_score,
                ever_benched=ever_benched,
                experience_years=experience,
                payment_tier=payment_tier,
                skill_count=skill_count,
                joining_year=joining_year,
                risk=risk,
                threshold=threshold,
            )

            result = {
                "risk_score": round(risk, 3),
                "risk_level": risk_label(risk, "employee"),
                "recommended_role": selected_role,
                "recommended_skills": [skill.title() for skill in recommended_skills],
                "causes": causes,
            }

            if send_alert_requested:
                if risk < threshold:
                    flash(
                        f"Alert not sent because risk {risk:.3f} is below threshold {threshold:.3f}.",
                        "warning",
                    )
                else:
                    current_role = _role()
                    current_username = str(session.get("username", "")).strip()
                    auto_recipients: list[str] = []
                    if current_role == "Employee":
                        email = get_active_user_email(current_username)
                        if email:
                            auto_recipients = [email]
                    else:
                        auto_recipients = list_active_emails_by_role("Employee")

                    manual_recipients = _parse_email_list(form_values["additional_employee_emails"])
                    recipients = list(dict.fromkeys(auto_recipients + manual_recipients))

                    if not recipients:
                        if current_role == "Employee":
                            flash(
                                "No employee email found in your account. Add one in signup/user data or enter it manually.",
                                "warning",
                            )
                        else:
                            flash(
                                "No employee recipient email available. Add employee user emails or provide additional email(s).",
                                "warning",
                            )
                    else:
                        now = pd.Timestamp.now()
                        employee_report_text = build_employee_report_text(
                            username=current_username if current_username else "Employee",
                            generated_at=now,
                            risk=risk,
                            threshold=threshold,
                            inputs={
                                "Education": education,
                                "Joining Year": joining_year,
                                "City": city,
                                "Payment Tier": payment_tier,
                                "Age": age,
                                "Gender": gender,
                                "Ever Benched": ever_benched,
                                "Experience in Current Domain": experience,
                                "Skills": skills,
                                "Performance Score": performance_score,
                                "Skill Count": skill_count,
                            },
                            causes=causes,
                            recommended_role=selected_role,
                            recommended_skills=recommended_skills,
                        )
                        report_filename = _report_filename(
                            "employee_risk_report",
                            current_username if current_username else "employee",
                        )

                        sent = 0
                        failed: list[str] = []
                        for recipient in recipients:
                            ok, msg = trigger_alert(
                                alert_type="EMPLOYEE",
                                subject="Personal Layoff Probability Alert",
                                body=employee_report_text,
                                recipient=recipient,
                                risk_score=risk,
                                threshold=threshold,
                                report_text=employee_report_text,
                                report_filename=report_filename,
                                cooldown_minutes=0,
                            )
                            if ok:
                                sent += 1
                            else:
                                failed.append(f"{recipient}: {msg}")

                        _flash_alert_delivery_summary(
                            alert_label="Employee",
                            sent=sent,
                            total=len(recipients),
                            failed=failed,
                        )
        except Exception as exc:
            flash(f"Unable to run employee prediction: {exc}", "error")

    return render_template(
        "employee_predictor.html",
        edu_options=edu_options,
        city_options=city_options,
        gender_options=gender_options,
        bench_options=bench_options,
        role_options=role_options,
        form_values=form_values,
        result=result,
    )


if __name__ == "__main__":
    ensure_default_admin()
    app.run(host="0.0.0.0", port=5000, debug=True)
