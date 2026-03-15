from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(override=True)


import difflib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from alerts import get_storage_mode, load_alert_log, trigger_alert
from auth import (
    authenticate_user,
    create_user,
    ensure_default_admin,
    get_active_user_email,
    get_user_storage_mode,
    list_active_emails_by_role,
)

st.set_page_config(page_title="Career Security Assistant", layout="wide")
st.title("Career Security Assistant")
st.caption("Secure Workforce Layoff Risk Prediction System")

RISK_CONFIG_PATH = Path("risk_thresholds.json")
DEFAULT_RISK_BOUNDS = {"low_max": 0.40, "high_min": 0.70}
BALANCED_THRESHOLD_HIGH_SHARE = 0.45
BALANCED_LOW_QUANTILE = 0.20
BALANCED_HIGH_QUANTILE = 0.80
SCORED_FILE_MAP: dict[str, tuple[str, str]] = {
    "company": ("company_scored.csv", "Company_Risk_Score"),
    "employee": ("employee_scored.csv", "Employee_Risk_Score"),
}


@st.cache_data
def load_risk_config() -> dict[str, object]:
    if not RISK_CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(RISK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


@st.cache_data
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


def render_risk_level(level: str) -> None:
    if level == "Low":
        st.success("Risk Level: Low")
    elif level == "Medium":
        st.warning("Risk Level: Medium")
    else:
        st.error("Risk Level: High")


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


def company_high_risk_reasons(
    funds: float,
    workforce_impacted_pct: float,
    stage: str,
    month: int,
    industry: str,
) -> list[str]:
    reasons: list[str] = []
    stage_text = str(stage).lower()
    industry_text = str(industry).lower()

    if workforce_impacted_pct >= 20:
        reasons.append("High workforce impact percentage suggests active downsizing pressure.")
    if funds <= 50:
        reasons.append("Lower funds raised can reduce runway and increase layoff likelihood.")
    if any(k in stage_text for k in ["seed", "pre-seed", "series a"]):
        reasons.append("Early growth stage companies often face higher financial volatility.")
    if month in (1, 2, 3, 10, 11, 12):
        reasons.append("Current month is in a period where restructuring cycles are commonly observed.")
    if any(k in industry_text for k in ["other", "consumer", "retail"]):
        reasons.append("Selected industry segment historically shows higher workforce variability.")

    if not reasons:
        reasons.append("Multiple combined model signals indicate elevated company-level risk.")
    return reasons


def employee_high_risk_reasons(
    performance_score: int,
    ever_benched: str,
    experience_years: int,
    payment_tier: int,
    skill_count: int,
    joining_year: int,
) -> list[str]:
    reasons: list[str] = []

    if performance_score <= 2:
        reasons.append("Low performance score is strongly associated with higher layoff risk.")
    if str(ever_benched).strip().lower() == "yes":
        reasons.append("Prior bench time indicates lower recent project allocation stability.")
    if experience_years <= 2:
        reasons.append("Lower domain experience can increase vulnerability during optimization cycles.")
    if payment_tier <= 1:
        reasons.append("Current payment tier may reflect roles more exposed to cost-cutting decisions.")
    if skill_count <= 2:
        reasons.append("Limited skill breadth can reduce redeployment opportunities.")
    if joining_year >= 2023:
        reasons.append("Recent joiners may face higher risk during workforce restructuring.")

    if not reasons:
        reasons.append("Multiple combined model signals indicate elevated employee-level risk.")
    return reasons


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
            "details": f"Observed workforce impacted: {workforce_impacted_pct:.1f}%. Trigger condition: >= 20.0%.",
        },
        {
            "cause": "Funding runway pressure",
            "triggered": funds <= 50.0,
            "details": f"Observed funds raised: {funds:.1f}M USD. Trigger condition: <= 50.0M USD.",
        },
        {
            "cause": "Early-stage volatility",
            "triggered": any(k in stage_text for k in ["seed", "pre-seed", "series a"]),
            "details": f"Observed company stage: '{stage}'. Trigger condition: seed/pre-seed/series A.",
        },
        {
            "cause": "Seasonal restructuring period",
            "triggered": month in (1, 2, 3, 10, 11, 12),
            "details": f"Observed month: {int(month)}. Trigger condition: Jan-Mar or Oct-Dec.",
        },
        {
            "cause": "Industry volatility signal",
            "triggered": any(k in industry_text for k in ["other", "consumer", "retail"]),
            "details": f"Observed industry: '{industry}'. Trigger condition: includes other/consumer/retail.",
        },
        {
            "cause": "Aggregate model risk signal",
            "triggered": risk >= threshold,
            "details": f"Predicted risk score: {risk:.3f}. Trigger condition: score >= {threshold:.3f}.",
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
            "details": f"Observed performance score: {int(performance_score)}. Trigger condition: <= 2.",
        },
        {
            "cause": "Bench history signal",
            "triggered": str(ever_benched).strip().lower() == "yes",
            "details": f"Observed ever benched: '{ever_benched}'. Trigger condition: yes.",
        },
        {
            "cause": "Low domain experience",
            "triggered": int(experience_years) <= 2,
            "details": f"Observed domain experience: {int(experience_years)} years. Trigger condition: <= 2 years.",
        },
        {
            "cause": "Compensation-tier exposure",
            "triggered": int(payment_tier) <= 1,
            "details": f"Observed payment tier: {int(payment_tier)}. Trigger condition: <= 1.",
        },
        {
            "cause": "Limited skill breadth",
            "triggered": int(skill_count) <= 2,
            "details": f"Observed skill count: {int(skill_count)}. Trigger condition: <= 2.",
        },
        {
            "cause": "Recent joining exposure",
            "triggered": int(joining_year) >= 2023,
            "details": f"Observed joining year: {int(joining_year)}. Trigger condition: >= 2023.",
        },
        {
            "cause": "Aggregate model risk signal",
            "triggered": risk >= threshold,
            "details": f"Predicted risk score: {risk:.3f}. Trigger condition: score >= {threshold:.3f}.",
        },
    ]


def _split_causes(causes: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contributing = [item for item in causes if bool(item.get("triggered", False))]
    non_contributing = [item for item in causes if not bool(item.get("triggered", False))]
    return contributing, non_contributing


def _triggered_cause_descriptions(causes: list[dict[str, object]]) -> list[str]:
    triggered = [str(item["details"]) for item in causes if bool(item.get("triggered", False))]
    if triggered:
        return triggered
    return ["No single rule crossed threshold; combined model pattern still contributes to risk."]


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

    contributing, non_contributing = _split_causes(causes)
    lines.append("")
    lines.append("Contributing Causes")
    if contributing:
        for idx, cause in enumerate(contributing, start=1):
            lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')}")
            lines.append(f"   {cause.get('details', '')}")
    else:
        lines.append("1. No individual rule-based cause was triggered.")

    lines.append("")
    lines.append("Other Evaluated Factors (Not Contributing)")
    if non_contributing:
        for idx, cause in enumerate(non_contributing, start=1):
            lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')}")
            lines.append(f"   {cause.get('details', '')}")
    else:
        lines.append("1. All evaluated factors were marked as contributing.")

    lines.append("")
    lines.append("Note")
    lines.append("This report combines rule-based cause checks and model-level risk output.")
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

    contributing, non_contributing = _split_causes(causes)
    lines.append("")
    lines.append("Contributing Causes")
    if contributing:
        for idx, cause in enumerate(contributing, start=1):
            lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')}")
            lines.append(f"   {cause.get('details', '')}")
    else:
        lines.append("1. No individual rule-based cause was triggered.")

    lines.append("")
    lines.append("Other Evaluated Factors (Not Contributing)")
    if non_contributing:
        for idx, cause in enumerate(non_contributing, start=1):
            lines.append(f"{idx}. {cause.get('cause', 'Unknown cause')}")
            lines.append(f"   {cause.get('details', '')}")
    else:
        lines.append("1. All evaluated factors were marked as contributing.")

    lines.append("")
    lines.append("Skill Guidance")
    lines.append(f"- Recommended role track: {recommended_role}")
    if recommended_skills:
        lines.append("- Top recommended skills: " + ", ".join([s.title() for s in recommended_skills]))
    else:
        lines.append("- Skill profile already aligns with selected role.")

    lines.append("")
    lines.append("Note")
    lines.append("This report combines rule-based cause checks and model-level risk output.")
    return "\n".join(lines)


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


def init_auth_state() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = ""
    if "username" not in st.session_state:
        st.session_state.username = ""


def login_ui() -> None:
    init_auth_state()
    ensure_default_admin()

    st.sidebar.header("Account")
    mode = st.sidebar.radio("Action", ["Sign In", "Sign Up"])

    user_store_mode, user_store_msg = get_user_storage_mode()
    st.sidebar.caption(f"User store: {user_store_mode.upper()}")
    st.sidebar.caption(user_store_msg)

    if mode == "Sign In":
        role = st.sidebar.selectbox("Role", ["Admin", "HR", "Employee"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Sign In"):
            ok, msg, user = authenticate_user(username, password, role=role)
            if ok and user is not None:
                st.session_state.authenticated = True
                st.session_state.role = str(user.get("role", role))
                st.session_state.username = str(user.get("username", username))
                st.sidebar.success("Authenticated")
                st.rerun()
            else:
                st.sidebar.error(msg)

    else:
        role = st.sidebar.selectbox("Register As", ["HR", "Employee"])
        username = st.sidebar.text_input("New Username")
        email = st.sidebar.text_input("Email (optional)")
        password = st.sidebar.text_input("New Password", type="password")
        confirm_password = st.sidebar.text_input("Confirm Password", type="password")

        if st.sidebar.button("Create Account"):
            if password != confirm_password:
                st.sidebar.error("Passwords do not match")
            else:
                ok, msg = create_user(username=username, password=password, role=role, email=email)
                if ok:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(msg)

    if not st.session_state.authenticated:
        st.info("Please sign in from the sidebar to continue.")
        st.stop()

    st.sidebar.success(f"Signed in as {st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.role = ""
        st.session_state.username = ""
        st.rerun()


@st.cache_resource
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


@st.cache_data
def load_data():
    return pd.read_csv("company-level.csv"), pd.read_csv("Employee-levelData.csv")


def render_company_predictor(
    company_model,
    company_df: pd.DataFrame,
    threshold: float,
    cooldown_minutes: int,
    auto_alert: bool,
) -> None:
    st.subheader("Company-Level Predictor")

    dataset_industries = company_df["Industry"].dropna().astype(str).unique().tolist()
    industry_options = sorted(set(dataset_industries + EXTRA_COMPANY_INDUSTRIES))
    stage_options = sorted(company_df["Stage"].dropna().astype(str).unique().tolist())
    country_options = sorted(company_df["Country"].dropna().astype(str).unique().tolist())

    with st.form("company_form"):
        company_name = st.text_input("Company Name", value="Demo Company")
        suggested_industry, matched_company = suggest_industry_for_company(company_name, company_df)
        default_industry = suggested_industry if suggested_industry in industry_options else industry_options[0]
        industry_options_with_custom = industry_options + ["Custom Industry"]
        default_idx = (
            industry_options_with_custom.index(default_industry)
            if default_industry in industry_options_with_custom
            else 0
        )
        selected_industry = st.selectbox("Industry", industry_options_with_custom, index=default_idx)
        custom_industry = ""
        if selected_industry == "Custom Industry":
            custom_industry = st.text_input("Enter Custom Industry")
        industry = custom_industry.strip() if selected_industry == "Custom Industry" and custom_industry.strip() else selected_industry
        if suggested_industry:
            st.caption(f"Auto-suggested from '{matched_company}': {suggested_industry} (you can change it)")
        else:
            st.caption("No company-industry match found. Please choose industry manually.")
        funds = st.number_input("Funds Raised (USD millions)", min_value=0.0, value=100.0)
        stage = st.selectbox("Company Stage", stage_options)
        country = st.selectbox("Country", country_options)
        workforce_impacted = st.slider("Workforce impacted %", 0.0, 100.0, 10.0, 0.1)
        now = pd.Timestamp.now()
        st.caption(f"Real-time timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        additional_hr_emails_text = st.text_input("Additional HR alert emails (comma-separated, optional)")

        c1, c2 = st.columns(2)
        with c1:
            predict_company = st.form_submit_button("Predict Company Risk")
        with c2:
            predict_company_and_alert = st.form_submit_button("Predict + Send Alert")

    if not (predict_company or predict_company_and_alert):
        return

    input_data = pd.DataFrame(
        [
            {
                "Industry": industry,
                "Funds_Raised": funds,
                "Funds_Raised_Log": float(np.log1p(max(float(funds), 0.0))),
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

    risk = float(company_model.predict_proba(input_data)[0][1])
    risk_level = risk_label(risk, "company")
    st.write(f"Company: {company_name}")
    st.metric("Company Risk Score", f"{risk:.3f}")
    render_risk_level(risk_level)
    st.progress(risk)

    company_causes = company_cause_breakdown(
        funds=funds,
        workforce_impacted_pct=float(workforce_impacted),
        stage=stage,
        month=int(now.month),
        industry=industry,
        risk=risk,
        threshold=threshold,
    )
    st.markdown("**Layoff Cause Analysis (All Causes)**")
    company_cause_df = pd.DataFrame(
        [
            {
                "Cause": str(item.get("cause", "")),
                "Status": "Contributing" if bool(item.get("triggered", False)) else "Not Contributing",
                "Details": str(item.get("details", "")),
            }
            for item in company_causes
        ]
    )
    st.dataframe(company_cause_df, use_container_width=True)

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
            "Workforce impacted %": f"{float(workforce_impacted):.1f}",
            "Year": int(now.year),
            "Quarter": int(((int(now.month) - 1) // 3) + 1),
            "Month": int(now.month),
        },
        causes=company_causes,
    )
    company_report_filename = _report_filename("company_risk_report", company_name)
    st.download_button(
        label="Download Full Company Report",
        data=company_report_text.encode("utf-8"),
        file_name=company_report_filename,
        mime="text/plain",
        key=f"download_company_report_{company_report_filename}",
    )

    if risk >= threshold:
        st.error("High company risk detected.")
        reasons = _triggered_cause_descriptions(company_causes)
        st.markdown("**Triggered high-risk causes:**")
        for idx, reason in enumerate(reasons, start=1):
            st.write(f"{idx}. {reason}")
        hr_recipients = list_active_emails_by_role("HR")
        additional_recipients = _parse_email_list(additional_hr_emails_text)
        recipients = list(dict.fromkeys(hr_recipients + additional_recipients))

        if hr_recipients:
            st.caption(f"Auto HR recipients found: {len(hr_recipients)}")

        if not recipients:
            st.warning("No HR recipient email available. Add HR user emails or provide additional HR email(s).")
            return

        should_alert = predict_company_and_alert or auto_alert
        if should_alert:
            effective_cooldown = 0 if predict_company_and_alert else cooldown_minutes
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
                    report_filename=company_report_filename,
                    cooldown_minutes=effective_cooldown,
                )
                if ok:
                    sent += 1
                else:
                    failed.append(f"{recipient}: {msg}")

            if sent:
                st.success(f"Company alert sent to {sent}/{len(recipients)} recipient(s).")
            for item in failed:
                st.warning(item)


def render_employee_predictor(
    employee_model,
    emp_df: pd.DataFrame,
    threshold: float,
    cooldown_minutes: int,
    auto_alert: bool,
) -> None:
    st.subheader("Employee-Level Predictor")

    edu_options = sorted(emp_df["Education"].dropna().astype(str).unique().tolist())
    city_options = sorted(emp_df["City"].dropna().astype(str).unique().tolist())
    gender_options = sorted(emp_df["Gender"].dropna().astype(str).unique().tolist())
    bench_options = sorted(emp_df["EverBenched"].dropna().astype(str).unique().tolist())

    with st.form("employee_form"):
        education = st.selectbox("Education", edu_options)
        joining_year = st.number_input("Joining Year", min_value=2000, max_value=2035, value=2019)
        city = st.selectbox("City", city_options)
        payment_tier = st.number_input("Payment Tier", min_value=1, max_value=5, value=2)
        age = st.number_input("Age", min_value=18, max_value=70, value=28)
        gender = st.selectbox("Gender", gender_options)
        ever_benched = st.selectbox("Ever Benched", bench_options)
        exp = st.number_input("Experience in Current Domain", min_value=0, max_value=30, value=3)
        skills = st.text_input("Skills (comma-separated)", "Python, SQL")
        role_options = ["Auto-detect"] + list(ROLE_SKILL_MAP.keys())
        target_role = st.selectbox("Target Job Role", role_options)
        perf = st.number_input("Performance Score", min_value=1, max_value=5, value=3)
        employee_email_text = st.text_input("Additional employee alert emails (comma-separated, optional)")

        c1, c2 = st.columns(2)
        with c1:
            predict_employee = st.form_submit_button("Predict Employee Risk")
        with c2:
            predict_employee_and_alert = st.form_submit_button("Predict + Send Alert")

    if not (predict_employee or predict_employee_and_alert):
        return

    normalized_skill_tokens = sorted(_normalize_skills(skills))
    model_skills = ",".join(normalized_skill_tokens) if normalized_skill_tokens else "none"
    skill_count = len(normalized_skill_tokens)
    experience_age_ratio = float(int(exp) / int(age)) if int(age) > 0 else 0.0
    input_data = pd.DataFrame(
        [
            {
                "Education": education,
                "JoiningYear": int(joining_year),
                "City": city,
                "PaymentTier": int(payment_tier),
                "Age": int(age),
                "Gender": gender,
                "EverBenched": ever_benched,
                "ExperienceInCurrentDomain": int(exp),
                "Skills": model_skills,
                "Performance_Score": int(perf),
                "Skill_Count": int(skill_count),
                "Experience_Age_Ratio": float(experience_age_ratio),
            }
        ]
    )

    risk = float(employee_model.predict_proba(input_data)[0][1])
    risk_level = risk_label(risk, "employee")
    st.metric("Employee Risk Score", f"{risk:.3f}")
    render_risk_level(risk_level)
    st.progress(risk)

    selected_role = infer_role_from_skills(skills) if target_role == "Auto-detect" else target_role
    recommended_skills = recommend_skill_gaps(skills, selected_role)
    employee_causes = employee_cause_breakdown(
        performance_score=int(perf),
        ever_benched=ever_benched,
        experience_years=int(exp),
        payment_tier=int(payment_tier),
        skill_count=int(skill_count),
        joining_year=int(joining_year),
        risk=risk,
        threshold=threshold,
    )
    st.markdown("**Layoff Cause Analysis (All Causes)**")
    employee_cause_df = pd.DataFrame(
        [
            {
                "Cause": str(item.get("cause", "")),
                "Status": "Contributing" if bool(item.get("triggered", False)) else "Not Contributing",
                "Details": str(item.get("details", "")),
            }
            for item in employee_causes
        ]
    )
    st.dataframe(employee_cause_df, use_container_width=True)

    employee_username = str(st.session_state.get("username", "Employee"))
    employee_report_text = build_employee_report_text(
        username=employee_username,
        generated_at=pd.Timestamp.now(),
        risk=risk,
        threshold=threshold,
        inputs={
            "Education": education,
            "Joining Year": int(joining_year),
            "City": city,
            "Payment Tier": int(payment_tier),
            "Age": int(age),
            "Gender": gender,
            "Ever Benched": ever_benched,
            "Experience in Current Domain": int(exp),
            "Skills": skills,
            "Performance Score": int(perf),
            "Skill Count": int(skill_count),
        },
        causes=employee_causes,
        recommended_role=selected_role,
        recommended_skills=recommended_skills,
    )
    employee_report_filename = _report_filename("employee_risk_report", employee_username)
    st.download_button(
        label="Download Full Employee Report",
        data=employee_report_text.encode("utf-8"),
        file_name=employee_report_filename,
        mime="text/plain",
        key=f"download_employee_report_{employee_report_filename}",
    )

    if risk >= threshold:
        st.error("High employee risk detected.")
        reasons = _triggered_cause_descriptions(employee_causes)
        st.markdown("**Triggered high-risk causes:**")
        for idx, reason in enumerate(reasons, start=1):
            st.write(f"{idx}. {reason}")

        st.markdown("**Skill Gap Recommender**")
        st.write(f"Recommended role track: {selected_role}")
        if recommended_skills:
            st.write("Top 3 skills to learn:")
            for idx, skill in enumerate(recommended_skills, start=1):
                st.write(f"{idx}. {skill.title()}")
        else:
            st.write("Great profile match for this role. Focus on advanced projects/certifications.")
        current_role = str(st.session_state.get("role", ""))
        current_username = str(st.session_state.get("username", ""))
        auto_recipients: list[str] = []
        if current_role == "Employee":
            auto_employee_email = get_active_user_email(current_username)
            if auto_employee_email:
                auto_recipients = [auto_employee_email]
                st.caption(f"Auto employee recipient from your account: {auto_employee_email}")
        else:
            auto_recipients = list_active_emails_by_role("Employee")
            if auto_recipients:
                st.caption(f"Auto employee recipients found: {len(auto_recipients)}")

        manual_recipients = _parse_email_list(employee_email_text)
        recipients = list(dict.fromkeys(auto_recipients + manual_recipients))

        if not recipients:
            if current_role == "Employee":
                st.warning("No employee email found. Add your email in account data or provide one manually.")
            else:
                st.warning("No employee recipient email available. Add employee user emails or provide additional email(s).")
            return

        should_alert = predict_employee_and_alert or auto_alert
        if should_alert:
            effective_cooldown = 0 if predict_employee_and_alert else cooldown_minutes
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
                    report_filename=employee_report_filename,
                    cooldown_minutes=effective_cooldown,
                )
                if ok:
                    sent += 1
                else:
                    failed.append(f"{recipient}: {msg}")

            if sent:
                st.success(f"Employee alert sent to {sent}/{len(recipients)} recipient(s).")
            for item in failed:
                st.warning(item)


def render_analytics(company_model, employee_model, company_df: pd.DataFrame, emp_df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Risk Analytics")

    company_plot_df = company_df.copy()
    company_plot_df["Laid_Off_Count"] = company_plot_df["Laid_Off_Count"].fillna(0)
    company_plot_df["Date"] = pd.to_datetime(company_plot_df["Date"], dayfirst=True, errors="coerce")
    company_plot_df["Year"] = company_plot_df["Date"].dt.year
    company_plot_df["Month"] = company_plot_df["Date"].dt.month
    company_plot_df["Quarter"] = company_plot_df["Date"].dt.quarter
    month_angle = 2 * np.pi * company_plot_df["Month"] / 12.0
    company_plot_df["Month_Sin"] = np.sin(month_angle)
    company_plot_df["Month_Cos"] = np.cos(month_angle)
    company_plot_df["Funds_Raised_Log"] = np.log1p(company_plot_df["Funds_Raised"].clip(lower=0))

    company_features = [
        "Industry",
        "Funds_Raised",
        "Funds_Raised_Log",
        "Stage",
        "Country",
        "workforce_impacted %",
        "Year",
        "Quarter",
        "Month_Sin",
        "Month_Cos",
    ]

    company_plot_df = company_plot_df.dropna(subset=["Year", "Month"])
    company_plot_df["Predicted_Risk"] = company_model.predict_proba(company_plot_df[company_features])[:, 1]

    trend = company_plot_df[["Date", "Predicted_Risk"]].dropna().sort_values("Date").set_index("Date")
    st.write("Company risk trend over time")
    st.line_chart(trend)

    country_risk = (
        company_plot_df.groupby("Country", as_index=False)["Predicted_Risk"]
        .mean()
        .sort_values("Predicted_Risk", ascending=False)
    )
    st.write("Average company risk by country")
    st.bar_chart(country_risk.set_index("Country"))

    emp_plot_df = emp_df.copy()
    emp_plot_df["Skill_Count"] = (
        emp_plot_df["Skills"]
        .fillna("")
        .astype(str)
        .apply(lambda s: sorted(_normalize_skills(s)))
    )
    emp_plot_df["Skills"] = emp_plot_df["Skill_Count"].apply(lambda tokens: ",".join(tokens) if tokens else "none")
    emp_plot_df["Skill_Count"] = emp_plot_df["Skill_Count"].apply(len)
    emp_plot_df["Experience_Age_Ratio"] = (
        emp_plot_df["ExperienceInCurrentDomain"].astype(float)
        / emp_plot_df["Age"].replace(0, np.nan).astype(float)
    ).fillna(0.0)
    emp_features = [
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
    emp_plot_df["Predicted_Risk"] = employee_model.predict_proba(emp_plot_df[emp_features])[:, 1]

    city_risk = (
        emp_plot_df.groupby("City", as_index=False)["Predicted_Risk"]
        .mean()
        .sort_values("Predicted_Risk", ascending=False)
    )
    st.write("Average employee risk by city")
    st.bar_chart(city_risk.set_index("City"))

    heatmap_df = (
        emp_plot_df.groupby(["City", "PaymentTier"])["Predicted_Risk"]
        .mean()
        .unstack(fill_value=0)
        .sort_index()
    )
    st.write("Employee risk heatmap (City vs Payment Tier)")
    try:
        st.dataframe(heatmap_df.style.background_gradient(cmap="Reds").format("{:.3f}"))
    except Exception:
        # Fallback when optional styling dependencies (e.g., matplotlib) are unavailable.
        st.dataframe(heatmap_df.round(3), use_container_width=True)


def render_alert_history() -> None:
    st.divider()
    st.subheader("Alert History")
    log_df = load_alert_log()
    if log_df.empty:
        st.info("No alerts logged yet.")
        return

    log_df = log_df.sort_values("sent_at", ascending=False)
    st.dataframe(log_df, use_container_width=True)


login_ui()
role = st.session_state.role

company_model, employee_model = load_models()
company_df, emp_df = load_data()

st.sidebar.header("Alert Settings")
_, company_default_high = get_risk_bounds("company")
_, employee_default_high = get_risk_bounds("employee")
company_alert_threshold = st.sidebar.slider("Company alert threshold", 0.0, 1.0, float(company_default_high), 0.01)
employee_alert_threshold = st.sidebar.slider("Employee alert threshold", 0.0, 1.0, float(employee_default_high), 0.01)
auto_alert = st.sidebar.checkbox("Auto-send alerts on high risk", value=True)
cooldown_minutes = st.sidebar.slider("Alert cooldown (minutes)", min_value=5, max_value=240, value=60, step=5)

storage_mode, storage_msg = get_storage_mode()
st.sidebar.caption(f"Alert storage: {storage_mode.upper()}")
st.sidebar.caption(storage_msg)

if role in ("Admin", "HR"):
    col1, col2 = st.columns(2)
    with col1:
        render_company_predictor(
            company_model,
            company_df,
            threshold=company_alert_threshold,
            cooldown_minutes=cooldown_minutes,
            auto_alert=auto_alert,
        )
    with col2:
        render_employee_predictor(
            employee_model,
            emp_df,
            threshold=employee_alert_threshold,
            cooldown_minutes=cooldown_minutes,
            auto_alert=auto_alert,
        )
    render_analytics(company_model, employee_model, company_df, emp_df)
    render_alert_history()
elif role == "Employee":
    render_employee_predictor(
        employee_model,
        emp_df,
        threshold=employee_alert_threshold,
        cooldown_minutes=cooldown_minutes,
        auto_alert=auto_alert,
    )
    render_alert_history()

