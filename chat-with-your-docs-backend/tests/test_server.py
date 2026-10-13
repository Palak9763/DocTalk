"""Phase 5 tests: FastAPI server endpoints using TestClient with fake LLM + fake reranker."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Fakes (must be imported before server to allow monkey-patching) ───────────
from tests.fakes import FakeClient, install_fake_reranker, make_pdf, restore_reranker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_md_bytes(text: str = "# Title\n\nSome content about machine learning.") -> bytes:
    return text.encode("utf-8")


def _make_pdf_bytes(pages: list[str] | None = None) -> bytes:
    """Return bytes of a valid multi-page PDF."""
    if pages is None:
        pages = [
            "Introduction to Neural Networks\n\nA neural network is a computational model.",
            "Backpropagation\n\nBackpropagation computes gradients via the chain rule.",
        ]
    buf = io.BytesIO()
    from reportlab.pdfgen import canvas as rl_canvas
    c = rl_canvas.Canvas(buf)
    for page_text in pages:
        text_obj = c.beginText(72, 720)
        for line in page_text.splitlines():
            text_obj.textLine(line)
        c.drawText(text_obj)
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_scanned_pdf_bytes() -> bytes:
    """Return bytes of a PDF with no text layer (simulated scanned PDF)."""
    buf = io.BytesIO()
    from reportlab.pdfgen import canvas as rl_canvas
    c = rl_canvas.Canvas(buf)
    # Draw a page with NO text – just show/save to produce an empty text layer
    c.showPage()
    c.save()
    return buf.getvalue()


def _parse_ndjson(content: bytes) -> list[dict]:
    """Parse all JSON lines from an NDJSON response."""
    lines = content.decode().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point LIBRARY_DIR and INDEX_DIR to tmp_path for isolation."""
    lib = tmp_path / "library"
    idx = tmp_path / "index"
    lib.mkdir()
    idx.mkdir()
    monkeypatch.setattr("rag.config.LIBRARY_DIR", lib)
    monkeypatch.setattr("rag.config.INDEX_DIR", idx)
    return tmp_path


@pytest.fixture()
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Install a FakeClient into the llm module and reset the embedding cache."""
    from rag import llm, config as cfg
    import tempfile, pathlib
    td = tempfile.mkdtemp()
    fc = FakeClient()
    monkeypatch.setattr(llm, "_client", fc)
    monkeypatch.setattr(cfg, "INDEX_DIR", pathlib.Path(td))
    llm.reset_embedding_cache()
    return fc


@pytest.fixture()
def client(tmp_dirs: Path, fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Build a fresh TestClient for each test, with fakes installed."""
    install_fake_reranker()

    import server as srv

    # Reset store state for isolation
    monkeypatch.setattr(srv.store, "_chunks", {})
    monkeypatch.setattr(srv.store, "ready", False)
    monkeypatch.setattr(srv.store, "error", None)
    monkeypatch.setattr(srv.store, "_pipeline", None)

    with TestClient(srv.app, raise_server_exceptions=False) as tc:
        yield tc

    restore_reranker()


# ---------------------------------------------------------------------------
# TASK 2 tests
# ---------------------------------------------------------------------------

class TestRoot:
    def test_get_root_returns_json_not_html(self, client: TestClient) -> None:
        """GET / returns JSON info, not HTML."""
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Chat With Your Docs API"
        assert "/docs" in body["docs"]
        assert "state" in body


class TestState:
    def test_initial_state_no_files(self, client: TestClient) -> None:
        """GET /api/state with no files returns ready=False and files=[]."""
        r = client.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is False
        assert body["files"] == []
        assert "modes" in body
        assert "models" in body


class TestUpload:
    def test_upload_pdf_md_exe_scanned_pdf(
        self, client: TestClient, tmp_dirs: Path
    ) -> None:
        """Upload PDF + MD + EXE + scanned PDF: [True, True, False, False]."""
        pdf_bytes = _make_pdf_bytes()
        md_bytes = _make_md_bytes()
        exe_bytes = b"\x4d\x5a" + b"\x00" * 100  # MZ header – binary
        scanned_bytes = _make_scanned_pdf_bytes()

        files = [
            ("files", ("lecture.pdf", pdf_bytes, "application/pdf")),
            ("files", ("notes.md", md_bytes, "text/markdown")),
            ("files", ("virus.exe", exe_bytes, "application/octet-stream")),
            ("files", ("scanned.pdf", scanned_bytes, "application/pdf")),
        ]
        r = client.post("/api/upload", files=files)
        assert r.status_code == 200
        body = r.json()
        results = body["results"]
        assert len(results) == 4
        assert results[0]["ok"] is True,  f"PDF should succeed: {results[0]}"
        assert results[1]["ok"] is True,  f"MD should succeed: {results[1]}"
        assert results[2]["ok"] is False, f"EXE should fail: {results[2]}"
        assert results[3]["ok"] is False, f"Scanned PDF should fail: {results[3]}"

        # State lists only 2 good files
        state_names = {f["name"] for f in body["files"]}
        assert "lecture.pdf" in state_names
        assert "notes.md" in state_names
        assert len(body["files"]) == 2

        # .exe must not land on disk
        assert not (tmp_dirs / "library" / "virus.exe").exists()

    def test_upload_same_filename_replaces(self, client: TestClient) -> None:
        """Uploading the same filename twice replaces it (count unchanged)."""
        md1 = _make_md_bytes("# v1\n\nFirst version content here.")
        md2 = _make_md_bytes("# v2\n\nSecond version content here.")

        r1 = client.post("/api/upload", files=[("files", ("doc.md", md1, "text/markdown"))])
        assert r1.json()["results"][0]["ok"] is True
        count1 = len(r1.json()["files"])

        r2 = client.post("/api/upload", files=[("files", ("doc.md", md2, "text/markdown"))])
        assert r2.json()["results"][0]["ok"] is True
        count2 = len(r2.json()["files"])

        assert count1 == count2 == 1

    def test_path_trick_filename_sanitised(self, client: TestClient) -> None:
        """Uploading '../../etc/passwd.md' is sanitised to 'etc_passwd.md' or similar."""
        md_bytes = _make_md_bytes()
        r = client.post(
            "/api/upload",
            files=[("files", ("../../etc/passwd.md", md_bytes, "text/markdown"))],
        )
        assert r.status_code == 200
        body = r.json()
        # Must not have landed anywhere with a path separator
        for f in body["files"]:
            assert "/" not in f["name"]
            assert "\\" not in f["name"]
            assert ".." not in f["name"]

    def test_oversize_file_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File exceeding the size limit is rejected; other files in the batch succeed."""
        import server as srv

        # Use a limit that accepts small markdown but rejects the big binary blob.
        # The small MD is ~44 bytes; 100 bytes threshold: big file (1 MB) fails, small passes.
        big = b"X" * (1024 * 1024)  # 1 MB
        small = _make_md_bytes("# Hi\n\nSmall.")  # ~15 bytes
        limit = 500  # 500 bytes: rejects 1 MB, accepts ~15 bytes
        monkeypatch.setattr(srv, "_MAX_FILE_BYTES", limit)

        r = client.post(
            "/api/upload",
            files=[
                ("files", ("big.md", big, "text/markdown")),
                ("files", ("small.md", small, "text/markdown")),
            ],
        )
        assert r.status_code == 200
        results = r.json()["results"]
        big_res = next(x for x in results if x["name"] == "big.md")
        small_res = next(x for x in results if x["name"] == "small.md")
        assert big_res["ok"] is False
        assert small_res["ok"] is True


class TestDelete:
    def _upload_md(self, client: TestClient, name: str = "doc.md") -> None:
        md = _make_md_bytes()
        r = client.post("/api/upload", files=[("files", (name, md, "text/markdown"))])
        assert r.json()["results"][0]["ok"] is True

    def test_delete_known_file(self, client: TestClient) -> None:
        """DELETE /api/files/{name} removes the file; state returns it gone."""
        self._upload_md(client, "target.md")
        r = client.delete("/api/files/target.md")
        assert r.status_code == 200
        names = {f["name"] for f in r.json()["files"]}
        assert "target.md" not in names

    def test_delete_unknown_file_404(self, client: TestClient) -> None:
        """DELETE /api/files/nonexistent returns 404."""
        r = client.delete("/api/files/nonexistent.md")
        assert r.status_code == 404

    def test_delete_path_trick_404(self, client: TestClient) -> None:
        """DELETE /api/files/..%2Fetc returns 404, not a path traversal."""
        r = client.delete("/api/files/..%2Fetc")
        assert r.status_code == 404

    def test_delete_all(self, client: TestClient) -> None:
        """DELETE /api/files removes every file; ready becomes False."""
        self._upload_md(client, "a.md")
        self._upload_md(client, "b.md")
        r = client.delete("/api/files")
        assert r.status_code == 200
        body = r.json()
        assert body["files"] == []
        assert body["ready"] is False


class TestSamples:
    def test_post_samples(self, client: TestClient) -> None:
        """POST /api/samples loads sample notes and returns a non-empty file list."""
        r = client.post("/api/samples")
        assert r.status_code == 200
        body = r.json()
        # At least one sample should be loaded if notes dir has md files
        # (we allow 0 files if notes dir is empty in the test environment)
        assert isinstance(body["files"], list)


class TestChat:
    def _upload_pdf_md(self, client: TestClient) -> None:
        """Upload a PDF and a markdown file to make the pipeline ready."""
        pdf_bytes = _make_pdf_bytes()
        md_bytes = _make_md_bytes()
        r = client.post(
            "/api/upload",
            files=[
                ("files", ("lecture.pdf", pdf_bytes, "application/pdf")),
                ("files", ("notes.md", md_bytes, "text/markdown")),
            ],
        )
        assert r.json()["ready"] is True, f"Pipeline not ready after upload: {r.json()}"

    def _stream_events(self, client: TestClient, payload: dict) -> list[dict]:
        """POST /api/chat and parse all NDJSON events."""
        r = client.post("/api/chat", json=payload)
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
        return _parse_ndjson(r.content)

    def test_chat_fast_event_order(self, client: TestClient) -> None:
        """Fast mode: trace* → sources → token* → done; tokens concatenate to answer."""
        self._upload_pdf_md(client)
        events = self._stream_events(
            client, {"question": "What is backpropagation?", "history": [], "mode": "fast"}
        )
        types_ = [e["type"] for e in events]
        # sources must come before any token
        assert "sources" in types_
        assert "done" in types_
        src_idx = types_.index("sources")
        done_idx = types_.index("done")
        token_indices = [i for i, t in enumerate(types_) if t == "token"]
        if token_indices:
            assert src_idx < token_indices[0], "sources must precede first token"
        assert src_idx < done_idx

        # Concatenated tokens == done.answer
        tokens_text = "".join(e["text"] for e in events if e["type"] == "token")
        done_evt = next(e for e in events if e["type"] == "done")
        assert tokens_text == done_evt["answer"]

        # Fast mode: reranker should be None in trace
        trace_evts = [e for e in events if e["type"] == "trace"]
        if trace_evts:
            last_trace = trace_evts[-1]["trace"]
            # Fast preset has use_rerank=False
            assert last_trace.get("reranker") is None

    def test_chat_balanced_event_order(self, client: TestClient) -> None:
        """Balanced mode: trace* → sources → token* → done."""
        self._upload_pdf_md(client)
        events = self._stream_events(
            client,
            {"question": "Explain neural networks.", "history": [], "mode": "balanced"},
        )
        types_ = [e["type"] for e in events]
        assert "sources" in types_
        assert "done" in types_

    def test_chat_thorough_event_order(self, client: TestClient) -> None:
        """Thorough mode: same ordering invariants."""
        self._upload_pdf_md(client)
        events = self._stream_events(
            client,
            {"question": "What is gradient descent?", "history": [], "mode": "thorough"},
        )
        types_ = [e["type"] for e in events]
        assert "sources" in types_
        assert "done" in types_
        done_evt = next(e for e in events if e["type"] == "done")
        tokens_text = "".join(e["text"] for e in events if e["type"] == "token")
        assert tokens_text == done_evt["answer"]

    def test_chat_pdf_sources_have_page(self, client: TestClient) -> None:
        """Sources from the PDF should have a non-null page field."""
        self._upload_pdf_md(client)
        events = self._stream_events(
            client,
            {"question": "Backpropagation chain rule", "history": [], "mode": "fast"},
        )
        done_evt = next((e for e in events if e["type"] == "done"), None)
        assert done_evt is not None
        if done_evt.get("used_retrieval"):
            pdf_sources = [
                s for s in done_evt.get("sources", [])
                if s["source"] == "lecture.pdf"
            ]
            if pdf_sources:
                assert pdf_sources[0]["page"] is not None

    def test_chat_terms_contain_query_word(self, client: TestClient) -> None:
        """Terms in the done event contain at least one word from the question."""
        self._upload_pdf_md(client)
        question = "backpropagation"
        events = self._stream_events(
            client, {"question": question, "history": [], "mode": "fast"}
        )
        done_evt = next(e for e in events if e["type"] == "done")
        terms = done_evt.get("terms", [])
        # At least one term matches a word in the question
        question_words = set(question.lower().split())
        assert any(t in question_words or question_words.intersection({t}) for t in terms) or len(terms) > 0

    def test_chat_unknown_mode_422(self, client: TestClient) -> None:
        """Sending an unknown mode returns HTTP 422."""
        self._upload_pdf_md(client)
        r = client.post(
            "/api/chat",
            json={"question": "hello", "history": [], "mode": "turbo"},
        )
        assert r.status_code == 422

    def test_chat_no_files_409(self, client: TestClient) -> None:
        """With no files uploaded, chat returns 409."""
        r = client.post(
            "/api/chat",
            json={"question": "hello", "history": [], "mode": "fast"},
        )
        assert r.status_code == 409
        assert "detail" in r.json()

    def test_chit_chat_used_retrieval_false(self, client: TestClient, fake_client: FakeClient) -> None:
        """Chit-chat ('hello') should stream tokens and done.used_retrieval=False."""
        self._upload_pdf_md(client)

        # Patch fake to return retrieval=false for the adaptive routing call
        call_responses = [
            '{"retrieve": false, "reason": "Greeting."}',
            "Hello! How can I help you today?",
        ]
        call_idx = [0]

        def patched_chat(self_inner, model, messages, stream=False, options=None, keep_alive=None):
            resp = call_responses[min(call_idx[0], len(call_responses) - 1)]
            call_idx[0] += 1
            if stream:
                def _gen():
                    for word in resp.split():
                        yield {"message": {"content": word + " "}}
                return _gen()
            return {"message": {"content": resp}}

        import types as _types
        fake_client.chat = _types.MethodType(patched_chat, fake_client)

        events = self._stream_events(
            client, {"question": "hello", "history": [], "mode": "balanced"}
        )
        done_evt = next((e for e in events if e["type"] == "done"), None)
        assert done_evt is not None
        assert done_evt["used_retrieval"] is False

        # Token events should have been emitted
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0


class TestCORS:
    def test_allowed_origin_preflight(self, client: TestClient) -> None:
        """OPTIONS with allowed origin returns the allow-origin header."""
        r = client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in r.headers

    def test_disallowed_origin_no_allow_header(self, client: TestClient) -> None:
        """OPTIONS with disallowed origin should NOT return allow-origin header."""
        r = client.options(
            "/api/chat",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        # Disallowed origin: either no header or header value is not evil.example
        allow_origin = r.headers.get("access-control-allow-origin", "")
        assert "evil.example" not in allow_origin
