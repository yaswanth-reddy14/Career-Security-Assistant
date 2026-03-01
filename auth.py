from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv
load_dotenv(override=True)

try:
    from pymongo import MongoClient
except ImportError:  # Optional dependency
    MongoClient = None
try:
    import certifi
except ImportError:
    certifi = None

USERS_CSV_PATH = Path("users.csv")


def _hash_password(password: str, salt_hex: str | None = None) -> str:
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    digest = hashlib.sha256((salt_hex + password).encode("utf-8")).hexdigest()
    return f"{salt_hex}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt_hex, known_digest = stored_hash.split("$", 1)
    candidate = hashlib.sha256((salt_hex + password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate, known_digest)


def _mongo_collection_with_status():
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        return None, "MONGODB_URI not set; using CSV user store"
    if MongoClient is None:
        return None, "pymongo not installed; using CSV user store"

    try:
        db_name = os.getenv("MONGODB_DB", "career_security")
        coll_name = os.getenv("MONGODB_USERS_COLLECTION", "users")
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
        return client[db_name][coll_name], "MongoDB user store active"
    except Exception as exc:
        return None, f"Mongo unavailable ({exc}); using CSV user store"


def get_user_storage_mode() -> tuple[str, str]:
    collection, msg = _mongo_collection_with_status()
    return ("mongodb", msg) if collection is not None else ("csv", msg)


def _load_users_csv(path: Path = USERS_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["username", "username_lower", "password_hash", "role", "email", "created_at", "is_active"])
    df = pd.read_csv(path)
    if "username_lower" not in df.columns and "username" in df.columns:
        df["username_lower"] = df["username"].astype(str).str.lower()
    if "is_active" not in df.columns:
        df["is_active"] = True
    return df


def _save_users_csv(df: pd.DataFrame, path: Path = USERS_CSV_PATH) -> None:
    df.to_csv(path, index=False)


def create_user(username: str, password: str, role: str, email: str = "") -> tuple[bool, str]:
    username = username.strip()
    role = role.strip()
    email = email.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if role not in {"HR", "Employee", "Admin"}:
        return False, "Invalid role"

    username_lower = username.lower()
    now = datetime.now().isoformat(timespec="seconds")
    password_hash = _hash_password(password)

    collection, _ = _mongo_collection_with_status()
    if collection is not None:
        existing = collection.find_one({"username_lower": username_lower})
        if existing:
            return False, "Username already exists"

        collection.insert_one(
            {
                "username": username,
                "username_lower": username_lower,
                "password_hash": password_hash,
                "role": role,
                "email": email,
                "created_at": now,
                "is_active": True,
            }
        )
        return True, "User registered successfully"

    df = _load_users_csv()
    if not df[df["username_lower"].astype(str) == username_lower].empty:
        return False, "Username already exists"

    new_row = pd.DataFrame(
        [
            {
                "username": username,
                "username_lower": username_lower,
                "password_hash": password_hash,
                "role": role,
                "email": email,
                "created_at": now,
                "is_active": True,
            }
        ]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    _save_users_csv(df)
    return True, "User registered successfully"


def authenticate_user(username: str, password: str, role: str | None = None) -> tuple[bool, str, dict | None]:
    username = username.strip()
    username_lower = username.lower()

    collection, _ = _mongo_collection_with_status()
    if collection is not None:
        user = collection.find_one({"username_lower": username_lower, "is_active": True}, {"_id": 0})
        if not user:
            return False, "User not found", None
        if role and str(user.get("role", "")) != role:
            return False, "Role mismatch", None
        if not _verify_password(password, str(user.get("password_hash", ""))):
            return False, "Invalid password", None
        return True, "Authenticated", user

    df = _load_users_csv()
    rec = df[(df["username_lower"].astype(str) == username_lower) & (df["is_active"].astype(str) != "False")]
    if rec.empty:
        return False, "User not found", None

    user = rec.iloc[0].to_dict()
    if role and str(user.get("role", "")) != role:
        return False, "Role mismatch", None
    if not _verify_password(password, str(user.get("password_hash", ""))):
        return False, "Invalid password", None
    return True, "Authenticated", user


def ensure_default_admin() -> None:
    admin_user = os.getenv("ADMIN_USERNAME", "admin").strip()
    admin_pass = os.getenv("ADMIN_PASSWORD", "admin123")
    ok, _, user = authenticate_user(admin_user, admin_pass, role="Admin")
    if ok and user:
        return

    # If username exists with different password, keep existing user.
    exists, msg, _ = authenticate_user(admin_user, admin_pass, role=None)
    if exists:
        return
    if msg == "Invalid password":
        return

    create_user(admin_user, admin_pass, "Admin", email="")


def _normalize_email(value: object) -> str:
    email = str(value).strip()
    if not email or email.lower() == "nan":
        return ""
    return email


def list_active_emails_by_role(role: str) -> list[str]:
    """Return unique active user emails for a given role."""
    role = role.strip()
    if not role:
        return []

    collection, _ = _mongo_collection_with_status()
    emails: list[str] = []

    if collection is not None:
        cursor = collection.find(
            {"role": role, "is_active": True},
            {"_id": 0, "email": 1},
        )
        for doc in cursor:
            email = _normalize_email(doc.get("email", ""))
            if email:
                emails.append(email)
    else:
        df = _load_users_csv()
        if df.empty:
            return []
        role_mask = df["role"].astype(str) == role
        active_mask = df["is_active"].astype(str).str.lower() != "false"
        filtered = df.loc[role_mask & active_mask, ["email"]].copy()
        for value in filtered["email"].tolist():
            email = _normalize_email(value)
            if email:
                emails.append(email)

    # Preserve first-seen order while removing duplicates.
    return list(dict.fromkeys(emails))


def get_active_user_email(username: str) -> str:
    """Return the active user's email, or empty string when unavailable."""
    username_lower = str(username).strip().lower()
    if not username_lower:
        return ""

    collection, _ = _mongo_collection_with_status()
    if collection is not None:
        user = collection.find_one(
            {"username_lower": username_lower, "is_active": True},
            {"_id": 0, "email": 1},
        )
        if not user:
            return ""
        return _normalize_email(user.get("email", ""))

    df = _load_users_csv()
    if df.empty:
        return ""

    rec = df[
        (df["username_lower"].astype(str) == username_lower)
        & (df["is_active"].astype(str).str.lower() != "false")
    ]
    if rec.empty:
        return ""
    return _normalize_email(rec.iloc[0].get("email", ""))

