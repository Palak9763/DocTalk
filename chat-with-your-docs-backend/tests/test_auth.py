"""Authentication and Per-User Data Isolation Tests.
Tests cover:
- Signup, login, me, and state flow.
- Duplicate email handling and case insensitivity.
- Password complexity and email validation (422 responses).
- Constant-time identical 401 response for bad password vs unknown user.
- In-memory rate limiting with injectable clock (429 Retry-After).
- Token security (missing, garbage, expired, wrong secret, alg 'none').
- Argon2 password hashing and absence of plaintext in app.db.
- Per-user data isolation across uploads, state, deletion, and chat retrieval.
- Admin role elevation via ADMIN_EMAILS and /api/admin/users authorization.
- Change-password lifecycle.
- CORS preflight with Authorization header.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator
import jwt
import pytest
from fastapi.testclient import TestClient
from tests.fakes import FakeClient, install_fake_reranker, restore_reranker
@pytest.fixture()
def auth_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Isolate DB, library, index, and JWT secrets in temporary directories."""
    import auth
    from rag import config as cfg
    import server as srv

    lib = tmp_path / "library"
    idx = tmp_path / "index"
    db_file = tmp_path / "app.db"
    jwt_secret = "test-jwt-secret-very-secure-and-long-enough-for-testing-purposes-64b"

    lib.mkdir(parents=True, exist_ok=True)
    idx.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "LIBRARY_DIR", lib)
    monkeypatch.setattr(cfg, "INDEX_DIR", idx)
    monkeypatch.setattr(auth, "AUTH_DB_PATH", db_file)
    monkeypatch.setattr(auth, "JWT_SECRET", jwt_secret)
    monkeypatch.setenv("JWT_SECRET", jwt_secret)
    auth.init_auth_db(db_file)
    auth.reset_rate_limiters()
    auth.reset_clock()
    srv.registry.reset()

    return {
        "lib": lib,
        "idx": idx,
        "db": db_file,
        "secret": jwt_secret,
    }

@pytest.fixture()
def fake_llm(monkeypatch: pytest.MonkeyPatch, auth_env: dict[str, Any]) -> FakeClient:
    from rag import config as cfg, llm
    fc = FakeClient()
    monkeypatch.setattr(llm, "_client", fc)
    monkeypatch.setattr(cfg, "INDEX_DIR", auth_env["idx"])
    llm.reset_embedding_cache()
    return fc

@pytest.fixture()
def client(auth_env: dict[str, Any], fake_llm: FakeClient) -> Iterator[TestClient]:
    install_fake_reranker()
    import server as srv
    with TestClient(srv.app, raise_server_exceptions=False) as tc:
        yield tc
    restore_reranker()

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestAuthLifecycle:
    def test_signup_login_me_state_roundtrip(self, client: TestClient) -> None:
        """Signup then login then /me round trip; token works for /api/state."""
        # 1. Signup
        signup_res = client.post(
            "/api/auth/signup",
            json={
                "name": "Asha Roy",
                "email": "asha@example.com",
                "password": "correct-horse-9",
            },
        )
        assert signup_res.status_code == 201
        data = signup_res.json()
        assert "token" in data
        assert data["user"]["email"] == "asha@example.com"
        assert data["user"]["role"] == "user"
        assert "password_hash" not in data["user"]

        token = data["token"]
        # 2. Login
        login_res = client.post(
            "/api/auth/login",
            json={
                "email": "asha@example.com",
                "password": "correct-horse-9",
            },
        )
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert "token" in login_data
        assert login_data["user"]["email"] == "asha@example.com"

        # 3. GET /api/auth/me
        me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        me_data = me_res.json()
        assert me_data["name"] == "Asha Roy"
        assert me_data["email"] == "asha@example.com"
        assert me_data["role"] == "user"
        assert "created_at" in me_data
        assert "password_hash" not in me_data

        # 4. Token works for /api/state
        state_res = client.get("/api/state", headers={"Authorization": f"Bearer {token}"})
        assert state_res.status_code == 200
        state_data = state_res.json()
        assert state_data["ready"] is False
        assert state_data["files"] == []

    def test_logout_endpoint(self, client: TestClient) -> None:
        """POST /api/auth/logout returns 204."""
        res = client.post("/api/auth/logout")
        assert res.status_code == 204

class TestValidationAndDuplicates:
    def test_duplicate_email_and_case_insensitivity(self, client: TestClient) -> None:
        """Duplicate email (also different case "A@x.com" vs "a@x.com") -> 409."""
        r1 = client.post(
            "/api/auth/signup",
            json={"name": "User One", "email": "A@x.com", "password": "password123"},
        )
        assert r1.status_code == 201

        # Duplicate with different case
        r2 = client.post(
            "/api/auth/signup",
            json={"name": "User Two", "email": "a@x.com", "password": "password123"},
        )
        assert r2.status_code == 409
        assert r2.json() == {"detail": "An account with this email already exists."}

    def test_weak_or_invalid_credentials_return_422(self, client: TestClient) -> None:
        """Weak, short, numeric-only, email-equal password or bad email -> 422."""
        # Short password (< 8 chars)
        r_short = client.post(
            "/api/auth/signup",
            json={"name": "Short", "email": "short@example.com", "password": "short"},
        )
        assert r_short.status_code == 422

        # Entirely numeric
        r_num = client.post(
            "/api/auth/signup",
            json={"name": "Num", "email": "num@example.com", "password": "1234567890"},
        )
        assert r_num.status_code == 422

        # Equal to email
        r_equal = client.post(
            "/api/auth/signup",
            json={"name": "Equal", "email": "equal@example.com", "password": "equal@example.com"},
        )
        assert r_equal.status_code == 422

        # Invalid email syntax
        r_email = client.post(
            "/api/auth/signup",
            json={"name": "Bad Email", "email": "not-an-email", "password": "validpassword123"},
        )
        assert r_email.status_code == 422

class TestLoginSecurityAndRateLimiting:
    def test_wrong_password_and_unknown_email_identical_401(self, client: TestClient) -> None:
        """Wrong password and unknown email return identical 401 bodies."""
        client.post(
            "/api/auth/signup",
            json={"name": "Carol", "email": "carol@example.com", "password": "securepassword123"},
        )

        r_wrong_pw = client.post(
            "/api/auth/login",
            json={"email": "carol@example.com", "password": "incorrectpassword"},
        )
        r_unknown_email = client.post(
            "/api/auth/login",
            json={"email": "unknown@example.com", "password": "anypassword123"},
        )

        assert r_wrong_pw.status_code == 401
        assert r_unknown_email.status_code == 401
        assert r_wrong_pw.json() == {"detail": "Incorrect email or password."}
        assert r_unknown_email.json() == {"detail": "Incorrect email or password."}

    def test_rate_limiting_locks_out_after_max_failures(self, client: TestClient) -> None:
        """After LOGIN_MAX_FAILURES wrong attempts the next attempt returns 429 with Retry-After."""
        import auth

        current_time = 100000.0
        auth.set_clock(lambda: current_time)

        client.post(
            "/api/auth/signup",
            json={"name": "Dave", "email": "dave@example.com", "password": "validpassword123"},
        )

        # 5 consecutive failures
        for _ in range(auth.LOGIN_MAX_FAILURES):
            r = client.post(
                "/api/auth/login",
                json={"email": "dave@example.com", "password": "wrong_attempt"},
            )
            assert r.status_code == 401

        # 6th attempt (even with right password) must return 429
        locked_res = client.post(
            "/api/auth/login",
            json={"email": "dave@example.com", "password": "validpassword123"},
        )
        assert locked_res.status_code == 429
        assert "Retry-After" in locked_res.headers
        assert int(locked_res.headers["Retry-After"]) > 0

        # Advance clock past lock window
        current_time += auth.LOGIN_LOCK_MINUTES * 60 + 5

        # Login now succeeds and resets failure count
        ok_res = client.post(
            "/api/auth/login",
            json={"email": "dave@example.com", "password": "validpassword123"},
        )
        assert ok_res.status_code == 200
class TestTokenValidation:
    def test_protected_endpoints_reject_invalid_tokens(self, client: TestClient) -> None:
        """Protected endpoints return 401 without token, garbage token, expired, wrong secret, alg 'none'."""
        # 1. No token
        r_none = client.get("/api/state")
        assert r_none.status_code == 401
        assert "Bearer" in r_none.headers.get("WWW-Authenticate", "")

        # 2. Garbage token
        r_garbage = client.get("/api/state", headers={"Authorization": "Bearer not-a-jwt"})
        assert r_garbage.status_code == 401
        assert "Bearer" in r_garbage.headers.get("WWW-Authenticate", "")

        # 3. Expired token
        expired_token = jwt.encode(
            {"sub": "user123", "role": "user", "exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
            "test-jwt-secret-very-secure-and-long-enough-for-testing-purposes-64b",
            algorithm="HS256",
        )
        r_expired = client.get("/api/state", headers={"Authorization": f"Bearer {expired_token}"})
        assert r_expired.status_code == 401

        # 4. Token signed with wrong secret
        wrong_secret_token = jwt.encode(
            {"sub": "user123", "role": "user", "exp": int(time.time()) + 3600, "iat": int(time.time())},
            "completely-different-wrong-secret-key-12345",
            algorithm="HS256",
        )
        r_wrong_secret = client.get("/api/state", headers={"Authorization": f"Bearer {wrong_secret_token}"})
        assert r_wrong_secret.status_code == 401

        # 5. Alg 'none' token
        none_token = jwt.encode(
            {"sub": "user123", "role": "user", "exp": int(time.time()) + 3600, "iat": int(time.time())},
            key="",
            algorithm="none",
        )
        r_none_alg = client.get("/api/state", headers={"Authorization": f"Bearer {none_token}"})
        assert r_none_alg.status_code == 401
class TestDatabaseSecurity:
    def test_argon2_hash_and_no_plaintext_in_db(self, client: TestClient, auth_env: dict[str, Any]) -> None:
        """password_hash in the DB starts with "$argon2" and plaintext password appears nowhere in DB file."""
        secret_password = "super-secret-unique-passphrase-999"
        client.post(
            "/api/auth/signup",
            json={"name": "Eve", "email": "eve@example.com", "password": secret_password},
        )

        db_path = auth_env["db"]
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE email = 'eve@example.com'")
        row = cur.fetchone()
        conn.close()

        assert row is not None
        pw_hash = row["password_hash"]
        assert pw_hash.startswith("$argon2")

        # Verify plaintext does not appear in binary database file
        raw_db_bytes = db_path.read_bytes()
        assert secret_password.encode("utf-8") not in raw_db_bytes
class TestPerUserIsolation:
    def test_user_data_isolation(self, client: TestClient) -> None:
        """User A uploads file; User B's state shows none; B deleting A's file -> 404; B cannot retrieve A's text."""
        # 1. Signup User A and User B
        res_a = client.post(
            "/api/auth/signup",
            json={"name": "Alice", "email": "alice@example.com", "password": "passwordA123"},
        )
        token_a = res_a.json()["token"]

        res_b = client.post(
            "/api/auth/signup",
            json={"name": "Bob", "email": "bob@example.com", "password": "passwordB123"},
        )
        token_b = res_b.json()["token"]

        # 2. User A uploads a markdown document
        doc_content_a = b"# Quantum Computing\n\nQubits leverage quantum superposition and entanglement."
        upload_a = client.post(
            "/api/upload",
            files=[("files", ("quantum.md", doc_content_a, "text/markdown"))],
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert upload_a.status_code == 200
        assert upload_a.json()["ready"] is True

        # 3. User B checks state: should have NO files
        state_b = client.get("/api/state", headers={"Authorization": f"Bearer {token_b}"})
        assert state_b.status_code == 200
        assert state_b.json()["files"] == []
        assert state_b.json()["ready"] is False

        # 4. User B attempts to delete User A's file: must return 404
        del_b = client.delete("/api/files/quantum.md", headers={"Authorization": f"Bearer {token_b}"})
        assert del_b.status_code == 404

        # 5. User B uploads their own document with same filename
        doc_content_b = b"# Biology Notes\n\nPhotosynthesis converts light into chemical energy."
        upload_b = client.post(
            "/api/upload",
            files=[("files", ("quantum.md", doc_content_b, "text/markdown"))],
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert upload_b.status_code == 200

        # 6. User B chats about Quantum: retrieved sources must NOT contain User A's text
        chat_b = client.post(
            "/api/chat",
            json={"question": "What is quantum superposition?", "mode": "fast"},
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert chat_b.status_code == 200
        lines = [json.loads(line) for line in chat_b.content.decode().splitlines() if line.strip()]
        sources_events = [e for e in lines if e.get("type") == "sources"]
        assert len(sources_events) > 0
        sources = sources_events[0].get("sources", [])
        for s in sources:
            assert "superposition and entanglement" not in s.get("text", "")

        # 7. User A still has their quantum document intact
        state_a = client.get("/api/state", headers={"Authorization": f"Bearer {token_a}"})
        assert state_a.status_code == 200
        assert any(f["name"] == "quantum.md" for f in state_a.json()["files"])
class TestAdminPrivileges:
    def test_admin_emails_and_admin_users_endpoint(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Admin email in ADMIN_EMAILS becomes admin; GET /api/admin/users is 403 for user and 200 for admin."""
        monkeypatch.setenv("ADMIN_EMAILS", "boss@doctalk.com,director@doctalk.com")

        # Signup admin
        res_admin = client.post(
            "/api/auth/signup",
            json={"name": "The Boss", "email": "boss@doctalk.com", "password": "adminpassword123"},
        )
        assert res_admin.status_code == 201
        assert res_admin.json()["user"]["role"] == "admin"
        token_admin = res_admin.json()["token"]

        # Signup normal user
        res_user = client.post(
            "/api/auth/signup",
            json={"name": "Regular User", "email": "user@doctalk.com", "password": "userpassword123"},
        )
        assert res_user.status_code == 201
        assert res_user.json()["user"]["role"] == "user"
        token_user = res_user.json()["token"]

        # Normal user calling /api/admin/users -> 403
        r_forbidden = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token_user}"})
        assert r_forbidden.status_code == 403

        # Admin calling /api/admin/users -> 200 with user list and file counts
        r_admin = client.get("/api/admin/users", headers={"Authorization": f"Bearer {token_admin}"})
        assert r_admin.status_code == 200
        user_list = r_admin.json()
        assert len(user_list) >= 2
        for u in user_list:
            assert "password_hash" not in u
            assert "file_count" in u
class TestPasswordChange:
    def test_change_password_flow(self, client: TestClient) -> None:
        """change-password: wrong current password -> 400; success -> old password fails, new works."""
        res = client.post(
            "/api/auth/signup",
            json={"name": "George", "email": "george@example.com", "password": "initial_password_1"},
        )
        token = res.json()["token"]

        # Wrong current password -> 400
        r_wrong = client.post(
            "/api/auth/change-password",
            json={"current_password": "wrong_current_pass", "new_password": "new_password_2"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r_wrong.status_code == 400
        assert r_wrong.json() == {"detail": "Incorrect current password."}

        # Success -> 204
        r_ok = client.post(
            "/api/auth/change-password",
            json={"current_password": "initial_password_1", "new_password": "new_password_2"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r_ok.status_code == 204

        # Old password login fails
        r_old_login = client.post(
            "/api/auth/login",
            json={"email": "george@example.com", "password": "initial_password_1"},
        )
        assert r_old_login.status_code == 401

        # New password login succeeds
        r_new_login = client.post(
            "/api/auth/login",
            json={"email": "george@example.com", "password": "new_password_2"},
        )
        assert r_new_login.status_code == 200
class TestCORS:
    def test_cors_preflight_allows_authorization(self, client: TestClient) -> None:
        """CORS preflight still allows the Authorization header from http://localhost:5173."""
        r = client.options(
            "/api/state",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
        allowed_headers = r.headers.get("access-control-allow-headers", "").lower()
        assert "*" in allowed_headers or "authorization" in allowed_headers
