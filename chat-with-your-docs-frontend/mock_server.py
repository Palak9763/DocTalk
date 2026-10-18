#!/usr/bin/env python3
"""
mock_server.py — Dev-only mock backend for Chat With Your Docs with Auth & User Isolation.
Standard library only. Runs on port 8000:
    python mock_server.py

Implements:
- POST /api/auth/signup (201 with token and user; 409 duplicate; 422 invalid)
- POST /api/auth/login (200 with token and user; 401 generic; 429 on special password 'ratelimit1')
- GET  /api/auth/me (200 with user profile; 401 if missing/invalid token)
- POST /api/auth/change-password (204 on success; 400 wrong current password; 422 invalid)
- POST /api/auth/logout (204)
- GET  /api/admin/users (200 user list for admins; 403 for normal users)
- Per-user file and state isolation across all /api/* endpoints
- Token expiration trigger: password 'expired' triggers 401 on subsequent requests
"""

import datetime
import json
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

# ─── In-memory Storage ────────────────────────────────────────────────────────
_lock = threading.Lock()

# Users: email -> {id, name, email, password, role, created_at, last_login}
_users: dict[str, dict] = {
    "admin@doctalk.com": {
        "id": "user_admin_001",
        "name": "Sarah Admin",
        "email": "admin@doctalk.com",
        "password": "adminpassword123",
        "role": "admin",
        "created_at": "2026-01-15T10:00:00Z",
        "last_login": "2026-10-18T08:30:00Z",
    },
    "student@doctalk.com": {
        "id": "user_student_002",
        "name": "Alex Student",
        "email": "student@doctalk.com",
        "password": "studentpassword123",
        "role": "user",
        "created_at": "2026-03-20T14:15:00Z",
        "last_login": "2026-10-17T18:00:00Z",
    },
}

# Tokens: token_str -> user_id
_tokens: dict[str, str] = {}
_expired_on_next: set[str] = set()

# Per-user files: user_id -> {filename: {name, size, chunks, pages, type}}
_user_files: dict[str, dict[str, dict]] = {
    "user_admin_001": {
        "admin_syllabus.md": {
            "name": "admin_syllabus.md",
            "size": 1540,
            "chunks": 3,
            "pages": None,
            "type": "md",
        }
    },
    "user_student_002": {},
}

# ─── CORS helper ─────────────────────────────────────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization",
}


def _get_user_for_token(auth_header: str | None) -> tuple[dict | None, str | None]:
    """Validate Bearer token. Returns (user_dict, error_message)."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None, "Authentication credentials were not provided."
    token = auth_header[7:].strip()
    with _lock:
        if token in _expired_on_next:
            _expired_on_next.remove(token)
            _tokens.pop(token, None)
            return None, "Your session has expired. Please sign in again."
        user_id = _tokens.get(token)
        if not user_id:
            return None, "Invalid or expired token."
        for u in _users.values():
            if u["id"] == user_id:
                return u, None
    return None, "User not found."


def _make_state(user_id: str) -> dict:
    with _lock:
        files = list(_user_files.get(user_id, {}).values())
    return {
        "files": files,
        "ready": len(files) > 0,
        "error": None,
        "pipeline": "mock",
        "modes": ["fast", "balanced", "thorough"],
        "models": {
            "llm": "qwen2.5:3b-instruct",
            "embedding": "nomic-embed-text",
            "reranker": "bge-reranker-v2-m3",
        },
        "has_samples": True,
    }


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")

    # ── Send helpers ──────────────────────────────────────────────────────────
    def _send_json(self, code: int, data: dict, extra_headers: dict | None = None):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_no_content(self, code: int = 204):
        self.send_response(code)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def _send_ndjson_stream(self, events):
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        for delay, event in events:
            time.sleep(delay)
            line = (json.dumps(event) + "\n").encode("utf-8")
            try:
                self.wfile.write(f"{len(line):x}\r\n".encode())
                self.wfile.write(line)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except BrokenPipeError:
                break

        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

    # ── CORS preflight ────────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/":
            self._send_json(200, {
                "name": "Chat With Your Docs Mock API",
                "docs": "/docs",
                "state": "/api/state",
            })
            return

        if path == "/api/auth/me":
            user, err = _get_user_for_token(self.headers.get("Authorization"))
            if not user:
                self._send_json(401, {"detail": err or "Unauthorized"})
                return
            self._send_json(200, {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
                "created_at": user["created_at"],
            })
            return

        if path == "/api/admin/users":
            user, err = _get_user_for_token(self.headers.get("Authorization"))
            if not user:
                self._send_json(401, {"detail": err or "Unauthorized"})
                return
            if user["role"] != "admin":
                self._send_json(403, {"detail": "Admin privileges required."})
                return
            with _lock:
                user_list = []
                for u in _users.values():
                    count = len(_user_files.get(u["id"], {}))
                    user_list.append({
                        "id": u["id"],
                        "name": u["name"],
                        "email": u["email"],
                        "role": u["role"],
                        "created_at": u["created_at"],
                        "last_login": u["last_login"],
                        "file_count": count,
                    })
            self._send_json(200, user_list)
            return

        # Protected endpoints
        user, err = _get_user_for_token(self.headers.get("Authorization"))
        if not user:
            self._send_json(401, {"detail": err or "Unauthorized"})
            return

        if path == "/api/state":
            self._send_json(200, _make_state(user["id"]))
        else:
            self._send_json(404, {"detail": "Not found"})

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        # 1. Signup
        if path == "/api/auth/signup":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            name = (body.get("name") or "").strip()
            email = (body.get("email") or "").strip().lower()
            password = body.get("password") or ""

            if not name or len(name) > 60:
                self._send_json(422, {"detail": "Name must be between 1 and 60 characters."})
                return
            if "@" not in email or "." not in email:
                self._send_json(422, {"detail": "Invalid email address format."})
                return
            if len(password) < 8 or len(password) > 128:
                self._send_json(422, {"detail": "Password must be at least 8 characters long."})
                return
            if password.isdigit():
                self._send_json(422, {"detail": "Password cannot be entirely numeric."})
                return
            if password.lower() == email:
                self._send_json(422, {"detail": "Password cannot be equal to email address."})
                return

            with _lock:
                if email in _users:
                    self._send_json(409, {"detail": "An account with this email already exists."})
                    return
                user_id = f"user_{uuid.uuid4().hex[:8]}"
                role = "admin" if ("admin" in email or email == "boss@doctalk.com") else "user"
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                new_user = {
                    "id": user_id,
                    "name": name,
                    "email": email,
                    "password": password,
                    "role": role,
                    "created_at": now,
                    "last_login": None,
                }
                _users[email] = new_user
                _user_files[user_id] = {}
                token = f"mock_token_{uuid.uuid4().hex}"
                _tokens[token] = user_id

            self._send_json(201, {
                "token": token,
                "user": {
                    "id": new_user["id"],
                    "name": new_user["name"],
                    "email": new_user["email"],
                    "role": new_user["role"],
                },
            })
            return

        # 2. Login
        if path == "/api/auth/login":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            email = (body.get("email") or "").strip().lower()
            password = body.get("password") or ""

            # Test trigger: password "ratelimit1" triggers 429
            if password == "ratelimit1":
                self._send_json(429, {"detail": "Too many failed login attempts. Please try again later."}, {"Retry-After": "60"})
                return

            with _lock:
                user = _users.get(email)
                if not user or user["password"] != password:
                    self._send_json(401, {"detail": "Incorrect email or password."})
                    return

                user["last_login"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                token = f"mock_token_{uuid.uuid4().hex}"
                _tokens[token] = user["id"]

                # Test trigger: password "expired" makes the next call return 401
                if password == "expired":
                    _expired_on_next.add(token)

            self._send_json(200, {
                "token": token,
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "role": user["role"],
                },
            })
            return

        # 3. Logout
        if path == "/api/auth/logout":
            auth_header = self.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
                with _lock:
                    _tokens.pop(token, None)
            self._send_no_content(204)
            return

        # Protected endpoints below
        user, err = _get_user_for_token(self.headers.get("Authorization"))
        if not user:
            self._send_json(401, {"detail": err or "Unauthorized"})
            return

        # 4. Change Password
        if path == "/api/auth/change-password":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            curr = body.get("current_password", "")
            new_pw = body.get("new_password", "")

            if curr != user["password"]:
                self._send_json(400, {"detail": "Incorrect current password."})
                return
            if len(new_pw) < 8 or len(new_pw) > 128:
                self._send_json(422, {"detail": "Password must be at least 8 characters long."})
                return
            if new_pw.isdigit():
                self._send_json(422, {"detail": "Password cannot be entirely numeric."})
                return
            if new_pw.lower() == user["email"].lower():
                self._send_json(422, {"detail": "Password cannot be equal to email address."})
                return

            with _lock:
                user["password"] = new_pw
            self._send_no_content(204)
            return

        # 5. Upload files
        if path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b""

            filename = "document.pdf"
            if b'filename="' in raw_body:
                try:
                    filename = raw_body.split(b'filename="')[1].split(b'"')[0].decode(errors="ignore")
                except Exception:
                    filename = "document.pdf"

            ext = filename.split(".")[-1].lower() if "." in filename else "pdf"
            file_meta = {
                "name": filename,
                "size": len(raw_body),
                "chunks": 4,
                "pages": 2 if ext == "pdf" else None,
                "type": ext,
            }

            with _lock:
                if user["id"] not in _user_files:
                    _user_files[user["id"]] = {}
                _user_files[user["id"]][filename] = file_meta

            self._send_json(200, {
                "results": [{"name": filename, "ok": True, "chunks": 4, "error": None}],
                **_make_state(user["id"])
            })
            return

        # 6. Load sample notes
        if path == "/api/samples":
            sample_file = {
                "name": "01_sample_lecture.md",
                "size": 2400,
                "chunks": 5,
                "pages": None,
                "type": "md",
            }
            with _lock:
                if user["id"] not in _user_files:
                    _user_files[user["id"]] = {}
                _user_files[user["id"]]["01_sample_lecture.md"] = sample_file
            self._send_json(200, _make_state(user["id"]))
            return

        # 7. Streaming chat
        if path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            question = body.get("question", "What is this document about?")
            mode = body.get("mode", "balanced")

            with _lock:
                user_doc_count = len(_user_files.get(user["id"], {}))
                first_file_name = next(iter(_user_files.get(user["id"], {}).keys())) if user_doc_count else "notes.md"

            if user_doc_count == 0:
                self._send_json(409, {"detail": "No documents have been uploaded yet."})
                return

            events = [
                (0.04, {"type": "trace", "status": "Resolving query context...", "trace": {"query": question}}),
                (0.06, {"type": "trace", "status": "Searching candidate passages...", "trace": {"retrieved": ["c1", "c2"]}}),
                (0.04, {
                    "type": "sources",
                    "sources": [
                        {
                            "id": "c1",
                            "source": first_file_name,
                            "section": "Overview",
                            "page": 1,
                            "text": f"Grounded content verified from {first_file_name} for {user['name']}.",
                            "score": 0.89,
                            "dense_rank": 1,
                            "sparse_rank": 2,
                            "used": True,
                        }
                    ],
                    "terms": ["document", "content"],
                }),
                (0.04, {"type": "token", "text": f"According to **{first_file_name}**, "}),
                (0.04, {"type": "token", "text": "the key concepts involve "}),
                (0.04, {"type": "token", "text": "verified grounded representations [1]. "}),
                (0.04, {"type": "token", "text": "Everything is isolated per user."}),
                (0.02, {
                    "type": "done",
                    "answer": f"According to **{first_file_name}**, the key concepts involve verified grounded representations [1]. Everything is isolated per user.",
                    "used_retrieval": True,
                    "sources": [
                        {
                            "id": "c1",
                            "source": first_file_name,
                            "section": "Overview",
                            "page": 1,
                            "text": f"Grounded content verified from {first_file_name} for {user['name']}.",
                            "score": 0.89,
                            "dense_rank": 1,
                            "sparse_rank": 2,
                            "used": True,
                        }
                    ],
                    "terms": ["document", "content"],
                    "trace": {"timings_ms": {"search": 120, "generate": 450, "total": 570}},
                }),
            ]
            self._send_ndjson_stream(events)
            return

        self._send_json(404, {"detail": "Not found"})

    # ── DELETE ───────────────────────────────────────────────────────────────
    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        user, err = _get_user_for_token(self.headers.get("Authorization"))
        if not user:
            self._send_json(401, {"detail": err or "Unauthorized"})
            return

        with _lock:
            files_dict = _user_files.setdefault(user["id"], {})

        if path == "/api/files":
            with _lock:
                files_dict.clear()
            self._send_json(200, _make_state(user["id"]))
            return

        if path.startswith("/api/files/"):
            filename = urllib.parse.unquote(path[len("/api/files/"):])
            with _lock:
                if filename not in files_dict:
                    self._send_json(404, {"detail": f"File '{filename}' not found."})
                    return
                del files_dict[filename]
            self._send_json(200, _make_state(user["id"]))
            return

        self._send_json(404, {"detail": "Not found"})


if __name__ == "__main__":
    server_address = ("127.0.0.1", 8000)
    httpd = HTTPServer(server_address, MockHandler)
    print("Mock backend listening on http://127.0.0.1:8000 (Auth & User Isolation enabled)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping mock server.")
        httpd.server_close()
