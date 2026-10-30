"""Authentication, Authorization, and User Management for Chat With Your Docs.

Implements:
- Argon2id password hashing via argon2-cffi with constant-time dummy verification for unknown emails.
- PyJWT HS256 access tokens with claims: sub (user_id), role, iat, exp, jti.
- Parameterized SQLite storage in WAL mode (users table).
- In-memory rate limiting for login attempts (IP + email) and signups (IP).
- Pydantic validation for email, name, and password rules.
- FastAPI dependencies: get_current_user and require_admin.
- Auth endpoints router for signup, login, me, change-password, logout, and admin user listing.
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, ValidationInfo, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)

AUTH_DB_PATH = Path(os.getenv("AUTH_DB_PATH", str(ROOT / "data" / "app.db")))
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "480"))
MIN_PASSWORD_LENGTH = int(os.getenv("MIN_PASSWORD_LENGTH", "8"))
LOGIN_MAX_FAILURES = int(os.getenv("LOGIN_MAX_FAILURES", "5"))
LOGIN_LOCK_MINUTES = int(os.getenv("LOGIN_LOCK_MINUTES", "15"))

# JWT Secret setup: if unset, generate 64 random bytes once and persist in data/.jwt_secret
_SECRET_FILE = ROOT / "data" / ".jwt_secret"

def _resolve_jwt_secret() -> str:
    env_secret = os.getenv("JWT_SECRET")
    if env_secret and env_secret.strip():
        return env_secret.strip()

    if _SECRET_FILE.is_file():
        try:
            persisted = _SECRET_FILE.read_text(encoding="utf-8").strip()
            if persisted:
                return persisted
        except Exception as exc:
            logger.warning("Failed to read %s: %s", _SECRET_FILE, exc)

    new_secret = secrets.token_hex(64)
    try:
        _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SECRET_FILE.write_text(new_secret, encoding="utf-8")
        try:
            os.chmod(str(_SECRET_FILE), 0o600)
        except Exception:
            pass  # Windows may not support POSIX file mode changes
        logger.warning(
            "JWT_SECRET is not set in environment. Generated random 64-byte secret and saved to %s. "
            "Please set JWT_SECRET explicitly for production!",
            _SECRET_FILE,
        )
    except Exception as exc:
        logger.error("Could not persist JWT_SECRET to %s: %s", _SECRET_FILE, exc)

    return new_secret

JWT_SECRET = _resolve_jwt_secret()

def get_jwt_secret() -> str:
    """Return the active JWT secret (respects runtime environment overrides)."""
    return os.getenv("JWT_SECRET") or JWT_SECRET

def get_admin_emails() -> set[str]:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}

# ---------------------------------------------------------------------------
# Database Management
# ---------------------------------------------------------------------------
def get_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a thread-safe SQLite connection in WAL mode."""
    target = Path(db_path or AUTH_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_auth_db(db_path: Path | str | None = None) -> None:
    """Create the users table if missing."""
    conn = get_db(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT,
                last_login TEXT
            );
            """
        )
    conn.close()

# ---------------------------------------------------------------------------
# Password Hashing & Verification (argon2-cffi)
# ---------------------------------------------------------------------------
ph = PasswordHasher()
# Pre-computed dummy argon2 hash for constant-time comparison when email is unknown
_DUMMY_HASH = ph.hash("doctalk-dummy-password-for-timing-consistency-999")

def hash_password(password: str) -> str:
    """Hash plaintext password with argon2id."""
    return ph.hash(password)

def verify_password(password_hash: str, password: str) -> bool:
    """Verify password against argon2 hash. Returns True if valid, False on mismatch."""
    try:
        return ph.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:
        return False

def dummy_verify(password: str) -> None:
    """Perform dummy hash verification to preserve constant-time response for unknown emails."""
    try:
        ph.verify(_DUMMY_HASH, password)
    except Exception:
        pass

def needs_rehash(password_hash: str) -> bool:
    """Check if argon2 hash parameters need upgrading."""
    try:
        return ph.check_needs_rehash(password_hash)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Clock & Rate Limiting
# ---------------------------------------------------------------------------
_clock: Callable[[], float] = time.time

def set_clock(fn: Callable[[], float]) -> None:
    """Injectable clock for unit tests to verify rate limits without sleeping."""
    global _clock
    _clock = fn

def reset_clock() -> None:
    """Reset clock to system time.time."""
    global _clock
    _clock = time.time

_login_lock = threading.Lock()
# Key: (ip, email_lower) -> list of failure timestamps
_login_failures: dict[tuple[str, str], list[float]] = {}

_signup_lock = threading.Lock()
# Key: ip -> list of signup timestamps
_signup_attempts: dict[str, list[float]] = {}

def get_client_ip(request: Request) -> str:
    """Extract client IP from request, taking proxy headers into account."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def check_login_rate_limit(ip: str, email: str) -> None:
    """Raise 429 with Retry-After if too many failed login attempts."""
    now = _clock()
    lock_window = LOGIN_LOCK_MINUTES * 60
    key = (ip, email.strip().lower())

    with _login_lock:
        failures = _login_failures.get(key, [])
        # Prune failures older than lock window
        valid_failures = [t for t in failures if (now - t) < lock_window]
        _login_failures[key] = valid_failures

        if len(valid_failures) >= LOGIN_MAX_FAILURES:
            oldest = valid_failures[0]
            retry_after = int(oldest + lock_window - now)
            if retry_after > 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed login attempts. Please try again later.",
                    headers={"Retry-After": str(max(1, retry_after))},
                )

def record_login_failure(ip: str, email: str) -> None:
    now = _clock()
    key = (ip, email.strip().lower())
    with _login_lock:
        if key not in _login_failures:
            _login_failures[key] = []
        _login_failures[key].append(now)

def clear_login_failures(ip: str, email: str) -> None:
    key = (ip, email.strip().lower())
    with _login_lock:
        _login_failures.pop(key, None)

def check_and_record_signup_rate_limit(ip: str) -> None:
    """Simple per-IP rate limit: max 10 signups per hour."""
    now = _clock()
    window = 3600  # 1 hour
    max_signups = 10

    with _signup_lock:
        attempts = [t for t in _signup_attempts.get(ip, []) if (now - t) < window]
        if len(attempts) >= max_signups:
            oldest = attempts[0]
            retry_after = int(oldest + window - now)
            if retry_after > 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many signup requests. Please try again later.",
                    headers={"Retry-After": str(max(1, retry_after))},
                )
        attempts.append(now)
        _signup_attempts[ip] = attempts

def reset_rate_limiters() -> None:
    """Clear in-memory rate limiting state (useful for test isolation)."""
    with _login_lock:
        _login_failures.clear()
    with _signup_lock:
        _signup_attempts.clear()

# ---------------------------------------------------------------------------
# JWT Token Generation & Verification
# ---------------------------------------------------------------------------
def create_access_token(user_id: str, role: str) -> str:
    """Generate PyJWT HS256 access token with claims sub, role, iat, exp, jti."""
    now = int(_clock())
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + ACCESS_TOKEN_MINUTES * 60,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm="HS256")

def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate PyJWT access token.

    Rejects expired or tampered tokens with 401 and WWW-Authenticate: Bearer.
    """
    try:
        payload = jwt.decode(
            token,
            get_jwt_secret(),
            algorithms=["HS256"],
            options={"require": ["exp", "sub"]},
        )
        return payload
    except (jwt.PyJWTError, Exception) as exc:
        logger.debug("JWT decode failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ---------------------------------------------------------------------------
# User DB Operations (Supabase with SQLite Fallback)
# ---------------------------------------------------------------------------
try:
    from supabase_backend import (
        is_supabase_configured,
        sb_create_user,
        sb_get_user_by_email,
        sb_get_user_by_id,
        sb_update_last_login,
        sb_update_user_password,
    )
except ImportError:
    def is_supabase_configured() -> bool: return False
    def sb_get_user_by_id(u: str): return None
    def sb_get_user_by_email(e: str): return None
    def sb_create_user(d: dict): return False
    def sb_update_user_password(u: str, h: str): return False
    def sb_update_last_login(u: str, t: str): pass

def get_user_by_id(user_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    if is_supabase_configured():
        sb_user = sb_get_user_by_id(user_id)
        if sb_user:
            return sb_user
    conn = get_db(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, name, password_hash, role, created_at, last_login FROM users WHERE id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_user_by_email(email: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    email_clean = email.strip().lower()
    if is_supabase_configured():
        sb_user = sb_get_user_by_email(email_clean)
        if sb_user:
            return sb_user
    conn = get_db(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, email, name, password_hash, role, created_at, last_login FROM users WHERE email = ?",
            (email_clean,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def create_user(
    name: str,
    email: str,
    password: str,
    role: str = "user",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    user_id = uuid.uuid4().hex
    email_clean = email.strip().lower()
    pw_hash = hash_password(password)
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "id": user_id,
        "email": email_clean,
        "name": name.strip(),
        "password_hash": pw_hash,
        "role": role,
        "created_at": now_iso,
        "last_login": None,
    }

    if is_supabase_configured():
        ok = sb_create_user(record)
        if ok:
            return {
                "id": user_id,
                "email": email_clean,
                "name": name.strip(),
                "role": role,
                "created_at": now_iso,
                "last_login": None,
            }

    conn = get_db(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO users (id, email, name, password_hash, role, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?, NULL);
                """,
                (user_id, email_clean, name.strip(), pw_hash, role, now_iso),
            )
        return {
            "id": user_id,
            "email": email_clean,
            "name": name.strip(),
            "role": role,
            "created_at": now_iso,
            "last_login": None,
        }
    finally:
        conn.close()

def update_user_password(user_id: str, new_password_hash: str, db_path: Path | str | None = None) -> None:
    if is_supabase_configured():
        sb_update_user_password(user_id, new_password_hash)
    conn = get_db(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_password_hash, user_id),
            )
    finally:
        conn.close()

def update_last_login(user_id: str, db_path: Path | str | None = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    if is_supabase_configured():
        sb_update_last_login(user_id, now_iso)
    conn = get_db(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (now_iso, user_id),
            )
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
def _validate_password_complexity(v: str, email_candidate: str | None = None) -> str:
    if len(v) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(v) > 128:
        raise ValueError("Password cannot exceed 128 characters.")
    if v.isdigit():
        raise ValueError("Password cannot be entirely numeric.")
    if email_candidate and v.strip().lower() == email_candidate.strip().lower():
        raise ValueError("Password cannot be equal to email address.")
    return v
class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    email: EmailStr
    password: str = Field(...)

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str) -> str:
        trimmed = v.strip()
        if not (1 <= len(trimmed) <= 60):
            raise ValueError("Name must be between 1 and 60 characters.")
        return trimmed

    @field_validator("password")
    @classmethod
    def check_password(cls, v: str, info: ValidationInfo) -> str:
        email = info.data.get("email") if info.data else None
        return _validate_password_complexity(v, str(email) if email else None)
class LoginRequest(BaseModel):
    email: EmailStr
    password: str
class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------
bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict[str, Any]:
    """Validate Bearer token and return current active user record (no password_hash)."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account was removed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
    }

async def require_admin(
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Ensure authenticated user has the 'admin' role, otherwise 403."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return user

# ---------------------------------------------------------------------------
# Auth API Router
# ---------------------------------------------------------------------------
router = APIRouter(tags=["Authentication"])

@router.post(
    "/api/auth/signup",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def api_signup(req: SignupRequest, request: Request) -> dict[str, Any]:
    client_ip = get_client_ip(request)
    check_and_record_signup_rate_limit(client_ip)

    email_clean = req.email.strip().lower()
    existing = get_user_by_email(email_clean)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    role = "admin" if email_clean in get_admin_emails() else "user"
    user = create_user(
        name=req.name,
        email=email_clean,
        password=req.password,
        role=role,
    )
    token = create_access_token(user["id"], user["role"])

    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
        },
    }

@router.post(
    "/api/auth/login",
    status_code=status.HTTP_200_OK,
    summary="Log in with email and password",
)
async def api_login(req: LoginRequest, request: Request) -> dict[str, Any]:
    client_ip = get_client_ip(request)
    email_clean = req.email.strip().lower()

    check_login_rate_limit(client_ip, email_clean)

    user = get_user_by_email(email_clean)
    if not user:
        dummy_verify(req.password)
        record_login_failure(client_ip, email_clean)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    if not verify_password(user["password_hash"], req.password):
        record_login_failure(client_ip, email_clean)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    # Rehash password if argon2 parameters updated
    if needs_rehash(user["password_hash"]):
        new_hash = hash_password(req.password)
        update_user_password(user["id"], new_hash)

    # Login succeeded: clear failures and record login timestamp
    clear_login_failures(client_ip, email_clean)
    update_last_login(user["id"])

    token = create_access_token(user["id"], user["role"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
        },
    }

@router.get(
    "/api/auth/me",
    status_code=status.HTTP_200_OK,
    summary="Get profile of the currently signed-in user",
)
async def api_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"],
    }

@router.post(
    "/api/auth/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Change user password",
)
async def api_change_password(
    req: ChangePasswordRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    full_user = get_user_by_id(user["id"])
    if not full_user or not verify_password(full_user["password_hash"], req.current_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password.",
        )

    # Validate new password rules against user email
    try:
        _validate_password_complexity(req.new_password, user["email"])
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )

    new_hash = hash_password(req.new_password)
    update_user_password(user["id"], new_hash)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post(
    "/api/auth/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out active session (client discards token)",
)
async def api_logout() -> Response:
    """Stateless logout endpoint: clients discard the bearer token."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)
