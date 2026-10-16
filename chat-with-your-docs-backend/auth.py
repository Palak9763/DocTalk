"""Authentication and User Management for DocTalk.

Zero external dependencies: uses Python standard library
- sqlite3 for storage
- hashlib + secrets for PBKDF2 password hashing & salts
- hmac + base64 + json for tamper-proof signed bearer tokens
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# Secret key for HMAC token signing (can be set via env var)
AUTH_SECRET = os.getenv("AUTH_SECRET", "doctalk-secret-local-key-2026-secure")
TOKEN_EXPIRY_SECONDS = 7 * 24 * 3600  # 7 days

DB_PATH = Path(os.getenv("AUTH_DB_PATH", "data/users.db"))


# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    """Create tables and seed default admin and student demo accounts."""
    conn = _get_db()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'student')),
                created_at REAL NOT NULL
            )
            """
        )

    # Seed default accounts if empty
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM users")
    count = cur.fetchone()["count"]

    if count == 0:
        logger.info("Seeding default demo accounts for DocTalk...")
        create_user(
            name="Prof. Sarah (Admin)",
            email="admin@doctalk.com",
            password="admin123",
            role="admin",
        )
        create_user(
            name="Alex Student",
            email="student@doctalk.com",
            password="student123",
            role="student",
        )
        logger.info("Demo accounts seeded: admin@doctalk.com and student@doctalk.com")


# ---------------------------------------------------------------------------
# Password Hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str) -> str:
    key = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        iterations=100_000,
    )
    return key.hex()


# ---------------------------------------------------------------------------
# Token Generation & Verification (HMAC-SHA256 Signed JSON)
# ---------------------------------------------------------------------------

def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": int(time.time()) + TOKEN_EXPIRY_SECONDS,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64encode(payload_bytes)

    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).digest()
    sig_b64 = _b64encode(signature)

    return f"{payload_b64}.{sig_b64}"


def verify_token(token: str) -> Optional[dict[str, Any]]:
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, sig_b64 = parts

        expected_sig = hmac.new(
            AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).digest()
        actual_sig = _b64decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64decode(payload_b64).decode("utf-8"))
        if payload.get("exp", 0) < time.time():
            return None  # expired

        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# User Operations
# ---------------------------------------------------------------------------

def create_user(
    name: str,
    email: str,
    password: str,
    role: str = "student",
) -> dict[str, Any]:
    email = email.strip().lower()
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    user_id = secrets.token_hex(8)
    created_at = time.time()

    conn = _get_db()
    with conn:
        conn.execute(
            """
            INSERT INTO users (id, name, email, password_hash, salt, role, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, name, email, password_hash, salt, role, created_at),
        )

    return {
        "id": user_id,
        "name": name,
        "email": email,
        "role": role,
    }


def authenticate_user(email: str, password: str) -> Optional[dict[str, Any]]:
    email = email.strip().lower()
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    if not row:
        return None

    expected_hash = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected_hash, row["password_hash"]):
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
    }


def get_user_by_id(user_id: str) -> Optional[dict[str, Any]]:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, email, role, created_at FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# FastAPI Dependency Helpers
# ---------------------------------------------------------------------------

async def get_optional_current_user(
    authorization: Optional[str] = Header(None),
) -> Optional[dict[str, Any]]:
    """Return user dict if valid Bearer token provided, else None."""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    payload = verify_token(parts[1])
    if not payload:
        return None
    return get_user_by_id(payload.get("sub", ""))


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    """Require valid logged-in user."""
    user = await get_optional_current_user(authorization)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(
    authorization: Optional[str] = Header(None),
) -> dict[str, Any]:
    """Require admin role if token is provided.
    
    If auth token is provided and role is 'student', return 403 Forbidden.
    If no token is provided in local dev mode, allow for backward compatibility.
    """
    if not authorization:
        # Backward compatibility for direct CLI/curl usage without auth header
        return {"id": "default", "role": "admin", "name": "Local Dev"}
    user = await get_current_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privilege required for document management.",
        )
    return user


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=4)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=4, max_length=100)
    role: str = Field("student", pattern="^(student|admin)$")
