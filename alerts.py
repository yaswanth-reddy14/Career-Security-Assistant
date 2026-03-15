from __future__ import annotations

import os
import ssl
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
load_dotenv(override=True)


try:
    from pymongo import DESCENDING, MongoClient
except ImportError:  # Optional dependency for MongoDB mode.
    MongoClient = None
    DESCENDING = -1
try:
    import certifi
except ImportError:
    certifi = None

ALERT_LOG_PATH = Path("alerts_log.csv")


def _smtp_setting(name: str) -> str:
    return os.getenv(name, "").strip()


def _empty_alert_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sent_at",
            "alert_type",
            "recipient",
            "risk_score",
            "threshold",
            "status",
            "message",
        ]
    )


def _mongo_collection_with_status():
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        return None, "MONGODB_URI not set; using CSV storage"
    if MongoClient is None:
        return None, "pymongo not installed; using CSV storage"

    try:
        db_name = os.getenv("MONGODB_DB", "career_security")
        coll_name = os.getenv("MONGODB_ALERTS_COLLECTION", "alerts")
        mongo_kwargs = {
            "serverSelectionTimeoutMS": int(os.getenv("MONGODB_TIMEOUT_MS", "8000")),
            "connectTimeoutMS": int(os.getenv("MONGODB_TIMEOUT_MS", "8000")),
            "socketTimeoutMS": int(os.getenv("MONGODB_TIMEOUT_MS", "8000")),
            "retryWrites": True,
            "tls": True,
        }
        if certifi is not None:
            mongo_kwargs["tlsCAFile"] = certifi.where()

        client = MongoClient(uri, **mongo_kwargs)
        client.admin.command("ping")
        return client[db_name][coll_name], "MongoDB storage active"
    except Exception as exc:
        return None, f"Mongo unavailable ({exc}); using CSV storage"


def get_storage_mode() -> tuple[str, str]:
    collection, msg = _mongo_collection_with_status()
    return ("mongodb", msg) if collection is not None else ("csv", msg)


def _get_mongo_collection():
    collection, _ = _mongo_collection_with_status()
    return collection


def send_alert_email(
    subject: str,
    body: str,
    recipient: str,
    attachments: list[tuple[str, bytes | str, str]] | None = None,
) -> tuple[bool, str]:
    """Send alert email using SMTP env configuration.

    Required env vars:
    SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER
    """
    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_SENDER"]
    settings = {key: _smtp_setting(key) for key in required}
    missing = [key for key, value in settings.items() if not value]

    if missing:
        return False, f"Missing SMTP settings: {', '.join(missing)}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings["SMTP_SENDER"]
    msg["To"] = recipient
    msg.set_content(body)

    if attachments:
        for filename, content, mime_type in attachments:
            if isinstance(content, str):
                payload = content.encode("utf-8")
            else:
                payload = content

            if "/" in str(mime_type):
                maintype, subtype = str(mime_type).split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"

            msg.add_attachment(payload, maintype=maintype, subtype=subtype, filename=str(filename))

    smtp_host = settings["SMTP_HOST"]
    smtp_port = int(settings["SMTP_PORT"])
    smtp_user = settings["SMTP_USERNAME"]
    smtp_password = settings["SMTP_PASSWORD"]
    is_gmail = "gmail.com" in smtp_host.lower()

    if is_gmail:
        # Gmail app passwords are commonly copied with spaces.
        smtp_password = smtp_password.replace(" ", "")

    try:
        if smtp_port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
    except smtplib.SMTPAuthenticationError as exc:
        if is_gmail:
            return (
                False,
                "Email authentication failed for Gmail (535). Use a 16-character "
                "Google App Password (not your regular Gmail password), enable "
                "2-Step Verification, and keep SMTP_USERNAME/SMTP_SENDER as the full Gmail address.",
            )
        return False, f"SMTP authentication failed: {exc}"
    except Exception as exc:
        return False, f"Email sending failed: {exc}"

    return True, "Alert sent successfully"


def load_alert_log(log_path: Path | str = ALERT_LOG_PATH) -> pd.DataFrame:
    collection = _get_mongo_collection()

    if collection is not None:
        docs = list(
            collection.find(
                {},
                {
                    "_id": 0,
                    "sent_at": 1,
                    "alert_type": 1,
                    "recipient": 1,
                    "risk_score": 1,
                    "threshold": 1,
                    "status": 1,
                    "message": 1,
                },
            ).sort("sent_at", DESCENDING)
        )
        if not docs:
            return _empty_alert_df()

        df = pd.DataFrame(docs)
        df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
        return df

    path = Path(log_path)
    if not path.exists():
        return _empty_alert_df()

    df = pd.read_csv(path)
    if "sent_at" in df.columns:
        df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    return df


def append_alert_log(
    alert_type: str,
    recipient: str,
    risk_score: float,
    threshold: float,
    status: str,
    message: str,
    log_path: Path | str = ALERT_LOG_PATH,
) -> None:
    row = {
        "sent_at": datetime.now().isoformat(timespec="seconds"),
        "alert_type": str(alert_type),
        "recipient": str(recipient),
        "risk_score": round(float(risk_score), 4),
        "threshold": round(float(threshold), 4),
        "status": str(status),
        "message": str(message),
    }

    collection = _get_mongo_collection()
    if collection is not None:
        collection.insert_one(row)
        return

    path = Path(log_path)
    row_df = pd.DataFrame([row])
    if path.exists():
        row_df.to_csv(path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(path, index=False)


def can_send_alert(
    alert_type: str,
    recipient: str,
    cooldown_minutes: int = 60,
    log_path: Path | str = ALERT_LOG_PATH,
) -> tuple[bool, str]:
    df = load_alert_log(log_path)
    if df.empty:
        return True, "No previous alerts for cooldown check"

    mask = (
        (df["alert_type"].astype(str) == str(alert_type))
        & (df["recipient"].astype(str) == str(recipient))
        & (df["status"].astype(str) == "SUCCESS")
    )
    recent = df.loc[mask].sort_values("sent_at", ascending=False)

    if recent.empty:
        return True, "No previous successful alerts"

    last_sent = recent.iloc[0]["sent_at"]
    if pd.isna(last_sent):
        return True, "Invalid previous timestamp"

    next_allowed = last_sent.to_pydatetime() + timedelta(minutes=int(cooldown_minutes))
    now = datetime.now()

    if now < next_allowed:
        mins_left = int((next_allowed - now).total_seconds() // 60) + 1
        return False, f"Cooldown active. Try again in ~{mins_left} minute(s)."

    return True, "Cooldown passed"


def trigger_alert(
    alert_type: str,
    subject: str,
    body: str,
    recipient: str,
    risk_score: float,
    threshold: float,
    report_text: str | None = None,
    report_filename: str | None = None,
    cooldown_minutes: int = 60,
    log_path: Path | str = ALERT_LOG_PATH,
) -> tuple[bool, str]:
    allowed, reason = can_send_alert(
        alert_type=alert_type,
        recipient=recipient,
        cooldown_minutes=cooldown_minutes,
        log_path=log_path,
    )
    if not allowed:
        append_alert_log(
            alert_type=alert_type,
            recipient=recipient,
            risk_score=risk_score,
            threshold=threshold,
            status="SKIPPED",
            message=reason,
            log_path=log_path,
        )
        return False, reason

    email_body = report_text if report_text else body
    attachments = None
    if report_text:
        attachments = [
            (
                report_filename or f"{str(alert_type).lower()}_risk_report.txt",
                report_text.encode("utf-8"),
                "text/plain",
            )
        ]

    ok, msg = send_alert_email(
        subject=subject,
        body=email_body,
        recipient=recipient,
        attachments=attachments,
    )
    append_alert_log(
        alert_type=alert_type,
        recipient=recipient,
        risk_score=risk_score,
        threshold=threshold,
        status="SUCCESS" if ok else "FAILED",
        message=msg,
        log_path=log_path,
    )
    return ok, msg

