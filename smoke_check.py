from __future__ import annotations

import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

USERS_PATH = Path("users.csv")
ALERTS_PATH = Path("alerts_log.csv")


def backup_file(path: Path) -> bytes | None:
    if path.exists():
        return path.read_bytes()
    return None


def restore_file(path: Path, backup: bytes | None) -> None:
    if backup is None:
        if path.exists():
            path.unlink()
        return
    path.write_bytes(backup)


def record(results: list[dict[str, Any]], name: str, ok: bool, detail: str = "") -> None:
    results.append({"name": name, "ok": ok, "detail": detail})


def main() -> int:
    # Keep smoke test isolated from external DB side effects.
    os.environ["MONGODB_URI"] = ""
    os.environ["MONGODB_TIMEOUT_MS"] = "1000"
    socket.setdefaulttimeout(12)

    # Import after env override so auth/alerts use local CSV mode.
    import alerts
    import app as app_module
    import auth

    # Re-apply after dotenv override in imported modules.
    os.environ["MONGODB_URI"] = ""
    os.environ["MONGODB_TIMEOUT_MS"] = "1000"

    results: list[dict[str, Any]] = []
    users_backup = backup_file(USERS_PATH)
    alerts_backup = backup_file(ALERTS_PATH)

    try:
        auth.ensure_default_admin()
        admin_user = os.getenv("ADMIN_USERNAME", "admin").strip()
        admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")

        client_admin = app_module.app.test_client()

        # Basic route checks
        for route in ["/", "/signin", "/signup"]:
            resp = client_admin.get(route)
            record(results, f"GET {route}", resp.status_code == 200, f"status={resp.status_code}")

        # Admin login check
        resp = client_admin.post(
            "/signin",
            data={"role": "Admin", "username": admin_user, "password": admin_pass},
            follow_redirects=True,
        )
        is_dashboard = getattr(resp, "request", None) is not None and getattr(resp.request, "path", "") == "/dashboard"
        record(results, "Admin login", resp.status_code == 200 and is_dashboard, f"status={resp.status_code}")

        # Predictor checks
        company_df, emp_df = app_module.load_data()

        company_payload = {
            "action": "predict",
            "company_name": "SmokeTest Company",
            "industry": str(company_df["Industry"].dropna().astype(str).iloc[0]),
            "funds": "300",
            "stage": str(company_df["Stage"].dropna().astype(str).iloc[0]),
            "country": str(company_df["Country"].dropna().astype(str).iloc[0]),
            "workforce_impacted": "10",
            "additional_hr_emails": "",
        }
        resp = client_admin.post("/predict/company", data=company_payload, follow_redirects=True)
        record(
            results,
            "Company prediction route",
            resp.status_code == 200 and (b"Risk Score" in resp.data or b"Risk Level" in resp.data),
            f"status={resp.status_code}",
        )

        employee_payload = {
            "action": "predict",
            "education": str(emp_df["Education"].dropna().astype(str).iloc[0]),
            "joining_year": "2018",
            "city": str(emp_df["City"].dropna().astype(str).iloc[0]),
            "payment_tier": "2",
            "age": "28",
            "gender": str(emp_df["Gender"].dropna().astype(str).iloc[0]),
            "ever_benched": str(emp_df["EverBenched"].dropna().astype(str).iloc[0]),
            "experience": "4",
            "skills": "Python, SQL, Excel, Pandas",
            "target_role": "Auto-detect",
            "performance_score": "4",
            "additional_employee_emails": "",
        }
        resp = client_admin.post("/predict/employee", data=employee_payload, follow_redirects=True)
        record(
            results,
            "Employee prediction route",
            resp.status_code == 200 and (b"Risk Score" in resp.data or b"Risk Level" in resp.data),
            f"status={resp.status_code}",
        )

        # Signup OTP flow (mock email sender)
        original_signup_sender = app_module._send_signup_otp_email

        def mock_signup_sender(recipient: str, otp: str):
            return True, f"mock-otp-sent:{recipient}:{otp}"

        app_module._send_signup_otp_email = mock_signup_sender
        unique = datetime.now().strftime("%Y%m%d%H%M%S")
        test_username = f"smoke_user_{unique}"
        test_email = f"{test_username}@example.com"
        test_password = "smoke12345"

        # Signup requires unauthenticated session -> use fresh client.
        client_public = app_module.app.test_client()
        resp = client_public.post(
            "/signup",
            data={
                "role": "Employee",
                "username": test_username,
                "email": test_email,
                "company_name": "SmokeCo",
                "password": test_password,
                "confirm_password": test_password,
            },
            follow_redirects=False,
        )
        signup_redirect_ok = resp.status_code in (301, 302) and "/signup/verify-otp" in resp.headers.get("Location", "")
        record(results, "Signup start + OTP session", signup_redirect_ok, f"status={resp.status_code}")

        with client_public.session_transaction() as sess:
            token = str(sess.get("pending_signup_token", "")).strip()
        otp_ok = token in app_module.SIGNUP_OTP_STORE
        record(results, "Signup OTP token created", otp_ok, f"token_exists={otp_ok}")

        otp_value = str(app_module.SIGNUP_OTP_STORE[token]["otp"]) if otp_ok else ""
        resp = client_public.post(
            "/signup/verify-otp",
            data={"action": "verify", "otp": otp_value},
            follow_redirects=True,
        )
        signup_verified = resp.status_code == 200 and auth.username_exists(test_username)
        record(results, "Signup OTP verify + user creation", signup_verified, f"status={resp.status_code}")
        app_module._send_signup_otp_email = original_signup_sender

        # Sign in as new user
        client_user = app_module.app.test_client()
        resp = client_user.post(
            "/signin",
            data={"role": "Employee", "username": test_username, "password": test_password},
            follow_redirects=True,
        )
        signed_in_user = resp.status_code == 200 and getattr(resp.request, "path", "") == "/dashboard"
        record(results, "New user sign in", signed_in_user, f"status={resp.status_code}")

        # Profile email OTP flow (mock email sender)
        original_profile_sender = app_module._send_profile_email_otp_email

        def mock_profile_sender(recipient: str, otp: str):
            return True, f"mock-profile-otp:{recipient}:{otp}"

        app_module._send_profile_email_otp_email = mock_profile_sender
        new_email = f"{test_username}.new@example.com"

        resp = client_user.post(
            "/profile",
            data={"action": "update_email", "email": new_email},
            follow_redirects=False,
        )
        profile_redirect_ok = resp.status_code in (301, 302) and "/profile/verify-email-otp" in resp.headers.get("Location", "")
        record(results, "Profile email OTP start", profile_redirect_ok, f"status={resp.status_code}")

        with client_user.session_transaction() as sess:
            ptoken = str(sess.get("pending_profile_email_token", "")).strip()
        potp_ok = ptoken in app_module.PROFILE_EMAIL_OTP_STORE
        record(results, "Profile OTP token created", potp_ok, f"token_exists={potp_ok}")

        potp_value = str(app_module.PROFILE_EMAIL_OTP_STORE[ptoken]["otp"]) if potp_ok else ""
        resp = client_user.post(
            "/profile/verify-email-otp",
            data={"action": "verify", "otp": potp_value},
            follow_redirects=True,
        )
        profile_data = auth.get_active_user_profile(test_username) or {}
        profile_email_updated = str(profile_data.get("email", "")).strip().lower() == new_email.lower()
        record(
            results,
            "Profile OTP verify + email update",
            resp.status_code == 200 and profile_email_updated,
            f"status={resp.status_code}",
        )
        app_module._send_profile_email_otp_email = original_profile_sender

        # Real SMTP email check (no mocking here)
        smtp_recipient = os.getenv("SMTP_SENDER", "").strip() or "smoke@example.com"
        ok, msg = alerts.send_alert_email(
            subject="Smoke Test Email",
            body="Smoke test for email pipeline.",
            recipient=smtp_recipient,
        )
        record(results, "SMTP real email send", ok, msg)

        # Alert trigger + log append check
        ok2, msg2 = alerts.trigger_alert(
            alert_type="SMOKE",
            subject="Smoke Alert Trigger",
            body="Smoke alert body",
            recipient=smtp_recipient,
            risk_score=0.91,
            threshold=0.80,
            report_text="Smoke test report",
            report_filename="smoke_report.txt",
            cooldown_minutes=0,
        )
        log_df = alerts.load_alert_log()
        log_ok = not log_df.empty and (log_df["alert_type"].astype(str) == "SMOKE").any()
        record(results, "Alert trigger writes log", log_ok, f"trigger_ok={ok2}; msg={msg2}")

        # Print summary
        print("SMOKE TEST SUMMARY")
        print("-" * 80)
        passed = 0
        for item in results:
            status = "PASS" if item["ok"] else "FAIL"
            if item["ok"]:
                passed += 1
            print(f"[{status}] {item['name']} :: {item['detail']}")
        print("-" * 80)
        print(f"TOTAL: {passed}/{len(results)} passed")

        return 0 if passed == len(results) else 1

    finally:
        restore_file(USERS_PATH, users_backup)
        restore_file(ALERTS_PATH, alerts_backup)


if __name__ == "__main__":
    raise SystemExit(main())
