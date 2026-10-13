#!/usr/bin/env python3
"""
mock_server.py — Minimal dev-only mock backend for Chat With Your Docs.
Standard library only. Implements all 7 endpoints with CORS and a fake NDJSON
chat stream (trace, sources, tokens, done events) with small delays.

Run on port 8000:
    python mock_server.py

The static frontend server (python -m http.server 5173) hits this for API calls.
"""

import json
import time
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# ─── In-memory "database" ────────────────────────────────────────────────────
_lock = threading.Lock()
_files: dict[str, dict] = {}   # name -> {name, size, chunks, pages, type}
_has_samples = True


def _make_state() -> dict:
    with _lock:
        files = list(_files.values())
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
        "has_samples": _has_samples,
    }


# ─── CORS helper ─────────────────────────────────────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept",
}


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")

    # ── Send helpers ──────────────────────────────────────────────────────────
    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_ndjson_stream(self, events):
        """Send a list of (delay_s, dict) tuples as NDJSON stream."""
        self.send_response(200)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        for delay, event in events:
            time.sleep(delay)
            line = (json.dumps(event) + "\n").encode()
            try:
                # chunked encoding
                self.wfile.write(f"{len(line):x}\r\n".encode())
                self.wfile.write(line)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except BrokenPipeError:
                break

        # Terminator
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
        if path == "/api/state":
            self._send_json(200, _make_state())
        else:
            self._send_json(404, {"detail": "Not found"})

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/samples":
            with _lock:
                _files["lecture_notes.md"] = {
                    "name": "lecture_notes.md", "size": 4096,
                    "chunks": 12, "pages": None, "type": "md"
                }
                _files["research_paper.pdf"] = {
                    "name": "research_paper.pdf", "size": 204800,
                    "chunks": 28, "pages": 9, "type": "pdf"
                }
                _files["summary.txt"] = {
                    "name": "summary.txt", "size": 1024,
                    "chunks": 3, "pages": None, "type": "txt"
                }
            self._send_json(200, _make_state())

        elif path == "/api/upload":
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            # Parse multipart minimally to extract filenames
            import re
            names = re.findall(rb'filename="([^"]+)"', body)
            results = []
            with _lock:
                for name in names:
                    fname = name.decode(errors="replace")
                    _files[fname] = {
                        "name": fname, "size": 50000,
                        "chunks": 8, "pages": 3 if fname.endswith(".pdf") else None,
                        "type": fname.rsplit(".", 1)[-1] if "." in fname else "txt",
                    }
                    results.append({"name": fname, "ok": True, "chunks": 8, "error": None})
            state = _make_state()
            state["results"] = results
            self._send_json(200, state)

        elif path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "test question")
            mode     = body.get("mode", "balanced")
            self._stream_chat(question, mode)

        else:
            self._send_json(404, {"detail": "Not found"})

    # ── DELETE ───────────────────────────────────────────────────────────────
    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/files":
            with _lock:
                _files.clear()
            self._send_json(200, _make_state())
        elif path.startswith("/api/files/"):
            name = urllib.parse.unquote(path[len("/api/files/"):])
            with _lock:
                _files.pop(name, None)
            self._send_json(200, _make_state())
        else:
            self._send_json(404, {"detail": "Not found"})

    # ── Fake NDJSON chat stream ───────────────────────────────────────────────
    def _stream_chat(self, question: str, mode: str):
        src_id_a = "research_paper.pdf-chunk-2"
        src_id_b = "lecture_notes.md-chunk-5"

        FAKE_SOURCES = [
            {
                "id": src_id_a,
                "source": "research_paper.pdf",
                "section": "Introduction",
                "page": 2,
                "text": (
                    "Neural networks have revolutionized machine learning by enabling "
                    "automatic feature extraction from raw data. Deep architectures allow "
                    "models to learn hierarchical representations, making them especially "
                    "powerful for tasks like image recognition and natural language processing."
                ),
                "score": 0.912,
                "dense_rank": 1,
                "sparse_rank": 3,
                "used": True,
            },
            {
                "id": src_id_b,
                "source": "lecture_notes.md",
                "section": "Chapter 3: Transformers",
                "page": None,
                "text": (
                    "The Transformer architecture introduced the attention mechanism, which "
                    "allows models to weigh the importance of different tokens dynamically. "
                    "Self-attention computes queries, keys, and values to produce a "
                    "weighted sum of value vectors."
                ),
                "score": 0.867,
                "dense_rank": 2,
                "sparse_rank": 1,
                "used": True,
            },
            {
                "id": "summary.txt-chunk-1",
                "source": "summary.txt",
                "section": "",
                "page": None,
                "text": "A brief overview of deep learning concepts and applications.",
                "score": 0.531,
                "dense_rank": 5,
                "sparse_rank": 7,
                "used": False,
            },
        ]

        FAKE_TERMS = ["neural networks", "attention", "transformer", "deep learning"]

        TRACE_STAGE1 = {
            "original": question,
            "query": question if mode == "fast" else f"Explain {question.lower().rstrip('?')}",
            "queries": [question, f"What is {question.lower().rstrip('?')}"],
            "hyde": None,
            "retrieval_mode": "hybrid",
            "reranker": "bge-reranker-v2-m3" if mode != "fast" else None,
        }

        TRACE_STAGE2 = {
            **TRACE_STAGE1,
            "search": {
                "dense_k": 30, "sparse_k": 30, "rrf_k": 60,
                "max_candidates": 50,
                "dense_hits": 6, "sparse_hits": 5,
                "both": 3, "fused": 8, "duplicates_removed": 2,
                "to_reranker": 8 if mode != "fast" else 5,
                "dense_index": "hnsw", "stemming": True,
            },
            "candidates": 8,
            "retrieved": [src_id_a, src_id_b],
            "paths": {
                src_id_a: {"dense": 1, "sparse": 3},
                src_id_b: {"dense": 2, "sparse": 1},
            },
            "rerank_scores": {src_id_a: 0.912, src_id_b: 0.867},
            "decision": {"enabled": True, "retrieve": True,
                         "reason": "The question requires factual lookup."},
            "grading": "reranker" if mode == "balanced" else ("llm" if mode == "thorough" else None),
            "graded_out": [],
            "groundedness": {
                "grounded": True,
                "unsupported": [],
            } if mode == "thorough" else None,
            "regenerated": False,
            "used": [src_id_a, src_id_b],
        }

        ANSWER_TOKENS = [
            "Based on your documents, ",
            "**neural networks** and the **Transformer** architecture ",
            "are central topics. ",
            "\n\n",
            "The attention mechanism allows models to dynamically weight tokens — ",
            "enabling powerful representations for NLP tasks. ",
            "[research_paper.pdf#2.0] ",
            "\n\n",
            "Self-attention computes *queries*, *keys*, and *values* to produce ",
            "a weighted sum, as described in the lecture notes. ",
            "[lecture_notes.md#5.0]",
        ]

        FINAL_ANSWER = "".join(ANSWER_TOKENS)

        TIMINGS = {
            "decide":   45,
            "rewrite":  120,
            "expand":   90 if mode != "fast" else None,
            "search":   310,
            "rerank":   280 if mode != "fast" else None,
            "grade":    400 if mode == "thorough" else None,
            "generate": 820,
            "verify":   350 if mode == "thorough" else None,
            "total":    1625,
        }
        # Remove None timings
        TIMINGS = {k: v for k, v in TIMINGS.items() if v is not None}

        events = [
            # trace: rewrite stage
            (0.05, {"type": "trace", "status": "Rewriting your question…",
                    "trace": {"original": question,
                              "query": TRACE_STAGE1["query"],
                              "queries": TRACE_STAGE1["queries"]}}),
            # trace: retrieval started
            (0.15, {"type": "trace", "status": "Searching…",
                    "trace": {"retrieval_mode": "hybrid",
                              "reranker": TRACE_STAGE1["reranker"]}}),
            # trace: search results
            (0.25, {"type": "trace", "status": "Reranking…",
                    "trace": {"search": TRACE_STAGE2["search"],
                              "candidates": TRACE_STAGE2["candidates"]}}),
            # sources arrive BEFORE first token
            (0.10, {"type": "sources",
                    "sources": FAKE_SOURCES,
                    "terms": FAKE_TERMS}),
            # trace: decision
            (0.10, {"type": "trace", "status": "Generating answer…",
                    "trace": {"decision": TRACE_STAGE2["decision"],
                              "retrieved": TRACE_STAGE2["retrieved"],
                              "paths": TRACE_STAGE2["paths"],
                              "rerank_scores": TRACE_STAGE2["rerank_scores"]}}),
        ]

        # Stream tokens
        for tok in ANSWER_TOKENS:
            events.append((0.06, {"type": "token", "text": tok}))

        # done
        TRACE_STAGE2["timings_ms"] = TIMINGS
        events.append((0.05, {
            "type": "done",
            "answer": FINAL_ANSWER,
            "used_retrieval": True,
            "sources": FAKE_SOURCES,
            "terms": FAKE_TERMS,
            "trace": TRACE_STAGE2,
        }))

        self._send_ndjson_stream(events)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8000), MockHandler)
    print("Mock backend running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
