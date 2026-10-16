"""Phase 5: FastAPI REST API for Chat With Your Docs.

Run with:
    python server.py [--host 127.0.0.1] [--port 8000]

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import shutil
import threading
import unicodedata
import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import auth
from rag import config, llm
from rag.config import PRESETS, PipelineConfig
from rag.ingest import SUPPORTED, Chunk, load_document
from rag.pipeline import RAGPipeline, Result
from rag.retrieve import Reranker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODES: list[str] = ["fast", "balanced", "thorough"]
_MAX_FILE_BYTES: int = int(os.getenv("MAX_FILE_BYTES", str(50 * 1024 * 1024)))  # 50 MB
_ACCEPT_SUFFIXES: set[str] = {".pdf", ".docx", ".md", ".txt"}

_DEFAULT_CORS_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------

def _sanitise(name: str) -> str:
    """Return a safe filename: basename only, unicode letters/digits/space/._- kept.

    Rejects path separators and '..' sequences before sanitising character set.
    """
    # Take basename to strip any injected directory parts
    name = Path(name).name
    # Normalise unicode to NFC
    name = unicodedata.normalize("NFC", name)
    # Keep only safe characters: word chars, spaces, dots, hyphens
    name = re.sub(r"[^\w\s.\-]", "_", name, flags=re.UNICODE)
    name = name.strip()
    if not name or name in (".", ".."):
        name = "upload"
    return name


# ---------------------------------------------------------------------------
# Document Store
# ---------------------------------------------------------------------------

class DocumentStore:
    """Thread-safe in-memory document registry backed by LIBRARY_DIR on disk.

    Attributes:
        ready: True if a pipeline has been successfully built.
        error: Last build error message, or None.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: dict[str, list[Chunk]] = {}   # filename -> chunks
        self._pipeline: RAGPipeline | None = None
        self._reranker: Reranker = Reranker(config.RERANK_MODEL)
        self.ready: bool = False
        self.error: str | None = None

    # ------------------------------------------------------------------
    # Disk scanning
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Load every supported file in LIBRARY_DIR into the in-memory cache."""
        lib = Path(config.LIBRARY_DIR)
        lib.mkdir(parents=True, exist_ok=True)
        for path in sorted(lib.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED:
                continue
            if path.name in self._chunks:
                continue
            try:
                self._chunks[path.name] = load_document(path)
                logger.info("Scanned %s (%d chunks)", path.name, len(self._chunks[path.name]))
            except Exception as exc:
                logger.warning("Skipping %s during scan: %s", path.name, exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, filename: str, data: bytes) -> list[Chunk]:
        """Write data to LIBRARY_DIR and parse into chunks. Returns the chunks."""
        lib = Path(config.LIBRARY_DIR)
        lib.mkdir(parents=True, exist_ok=True)
        dest = lib / filename
        dest.write_bytes(data)
        chunks = load_document(dest)
        with self._lock:
            self._chunks[filename] = chunks
        return chunks

    def remove(self, name: str) -> None:
        """Remove a file from disk and the chunk cache. No-op if not present."""
        with self._lock:
            self._chunks.pop(name, None)
        path = Path(config.LIBRARY_DIR) / name
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Remove every uploaded file from disk and the chunk cache."""
        names = list(self._chunks.keys())
        for name in names:
            self.remove(name)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _all_chunks(self) -> list[Chunk]:
        return [c for chunks in self._chunks.values() for c in chunks]

    def build(self) -> None:
        """(Re)build the RAGPipeline over all current chunks, sharing the Reranker.

        Raises on errors (Ollama down, etc.). Use try_build() for safe variant.
        """
        with self._lock:
            chunks = self._all_chunks()
            if not chunks:
                self._pipeline = None
                self.ready = False
                self.error = None
                return
            pipeline = RAGPipeline(
                cfg=PipelineConfig(),
                chunks=chunks,
                reranker=self._reranker,
            )
            self._pipeline = pipeline
            self.ready = True
            self.error = None
            logger.info(
                "Pipeline rebuilt: %d chunks across %d files",
                len(chunks),
                len(self._chunks),
            )

    def try_build(self) -> None:
        """Build the pipeline, catching all exceptions into self.error."""
        try:
            self.build()
        except Exception as exc:
            msg = str(exc)
            # Surface Ollama connectivity problems clearly
            if "connect" in msg.lower() or "connection" in msg.lower():
                msg = (
                    f"Ollama is not reachable at {config.OLLAMA_HOST}. "
                    "Start Ollama and retry."
                )
            self.error = msg
            self.ready = False
            logger.error("Pipeline build failed: %s", msg)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def pipeline_for(self, mode: str) -> RAGPipeline:
        """Return a config-cloned pipeline for the given preset mode.

        Raises:
            RuntimeError: If no pipeline is ready.
        """
        if not self.ready or self._pipeline is None:
            raise RuntimeError("No pipeline is ready.")
        return self._pipeline.with_config(PRESETS[mode])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def file_infos(self) -> list[dict[str, Any]]:
        """Return per-file metadata for the API state response."""
        lib = Path(config.LIBRARY_DIR)
        infos = []
        for name, chunks in self._chunks.items():
            path = lib / name
            size = path.stat().st_size if path.exists() else 0
            # Infer pages: PDFs have int page numbers
            pages_set = {c.page for c in chunks if c.page is not None}
            pages = max(pages_set) if pages_set else None
            suffix = Path(name).suffix.lower().lstrip(".")
            infos.append({
                "name": name,
                "size": size,
                "chunks": len(chunks),
                "pages": pages,
                "type": suffix,
            })
        return infos


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
store = DocumentStore()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    question: str = Field(..., min_length=1, max_length=2000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)
    mode: str = Field("balanced")

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        return v


# ---------------------------------------------------------------------------
# Lifespan (startup) — must be defined before app creation
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Scan library dir, try to build the pipeline, and warm up models."""
    auth.init_auth_db()
    store.scan()
    store.try_build()

    # Pre-load reranker weights in background after first successful build
    if store.ready:
        threading.Thread(
            target=store._reranker.load,
            daemon=True,
            name="reranker-preload",
        ).start()

    # Warm up LLM in background
    threading.Thread(
        target=llm.warm_up,
        daemon=True,
        name="llm-warmup",
    ).start()

    yield  # application runs


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chat With Your Docs API",
    description=(
        "Local RAG backend – upload documents and chat with them. "
        "All inference runs locally via Ollama."
    ),
    version="0.5.0",
    lifespan=_lifespan,
)

# CORS
_raw_origins = os.getenv("CORS_ORIGINS", "")
_cors_origins: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins.strip()
    else _DEFAULT_CORS_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------

def _state_dict() -> dict[str, Any]:
    """Build the full API state payload."""
    cfg = PipelineConfig()
    has_samples = False
    notes_dir = Path(config.NOTES_DIR)
    if notes_dir.is_dir():
        has_samples = any(
            p.is_file() and p.suffix.lower() in SUPPORTED
            for p in notes_dir.iterdir()
        )
    return {
        "files": store.file_infos(),
        "ready": store.ready,
        "error": store.error,
        "pipeline": cfg.label(),
        "modes": _MODES,
        "models": {
            "llm": config.CHAT_MODEL,
            "embedding": config.EMBED_MODEL,
            "reranker": config.RERANK_MODEL,
        },
        "has_samples": has_samples,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="API root")
def root() -> dict[str, str]:
    """Return basic API info and links."""
    return {
        "name": "Chat With Your Docs API",
        "docs": "/docs",
        "state": "/api/state",
    }


@app.get("/api/state", summary="Library & pipeline state")
def api_state() -> dict[str, Any]:
    """Return current library files, pipeline readiness, and model info."""
    return _state_dict()


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.post("/api/auth/login", summary="Log in with email and password")
def api_login(req: auth.LoginRequest) -> dict[str, Any]:
    user = auth.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    token = auth.create_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": user}


@app.post("/api/auth/register", summary="Register a new student account")
def api_register(req: auth.RegisterRequest) -> dict[str, Any]:
    conn = auth._get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email = ?", (req.email.strip().lower(),))
    if cur.fetchone():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )
    user = auth.create_user(req.name, req.email, req.password, req.role)
    token = auth.create_token(user["id"], user["email"], user["role"])
    return {"token": token, "user": user}


@app.get("/api/auth/me", summary="Get profile of current authenticated user")
async def api_me(user: dict[str, Any] = Depends(auth.get_current_user)) -> dict[str, Any]:
    return {"user": user}


@app.get("/api/auth/demo-users", summary="Get demo user credentials for instant login")
def api_demo_users() -> dict[str, Any]:
    return {
        "users": [
            {
                "name": "Prof. Sarah (Admin)",
                "email": "admin@doctalk.com",
                "password": "admin123",
                "role": "admin",
                "description": "Full access to upload, remove documents, and view deep RAG traces.",
            },
            {
                "name": "Alex Student",
                "email": "student@doctalk.com",
                "password": "student123",
                "role": "student",
                "description": "Distraction-free study portal: ask questions with clean source citations.",
            },
        ]
    }


@app.post("/api/upload", summary="Upload one or more documents")
async def api_upload(
    files: list[UploadFile] = File(...),
    _admin: dict[str, Any] = Depends(auth.require_admin),
) -> dict[str, Any]:
    """Accept .pdf .docx .md .txt files (max 50 MB each).

    De-duplicates by filename (same name replaces old file). Rejects unsupported
    types and scanned (image-only) PDFs. Returns per-file results plus full state.
    """
    results: list[dict[str, Any]] = []
    any_success = False

    for uf in files:
        raw_name = uf.filename or "upload"
        safe_name = _sanitise(raw_name)

        # Check extension before reading bytes
        suffix = Path(safe_name).suffix.lower()
        if suffix not in _ACCEPT_SUFFIXES:
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": (
                    f"Unsupported file type '{suffix}'. "
                    f"Accepted: {sorted(_ACCEPT_SUFFIXES)}"
                ),
            })
            continue

        # Read and size-check
        data = await uf.read()
        if len(data) > _MAX_FILE_BYTES:
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": (
                    f"File exceeds the {_MAX_FILE_BYTES // (1024 * 1024)} MB limit."
                ),
            })
            continue

        # Parse (may raise ValueError for scanned PDFs)
        try:
            chunks = store.add(safe_name, data)
            results.append({
                "name": safe_name,
                "ok": True,
                "chunks": len(chunks),
                "error": None,
            })
            any_success = True
            logger.info("Uploaded %s – %d chunks", safe_name, len(chunks))
        except ValueError as exc:
            # Scanned / empty PDF or unsupported format from ingest
            store.remove(safe_name)  # clean up any partial disk write
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": str(exc),
            })
        except Exception:
            store.remove(safe_name)
            logger.exception("Unexpected error ingesting %s", safe_name)
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": "Internal error during ingestion.",
            })

    # Re-index once after the whole batch
    if any_success:
        store.try_build()

    return {"results": results, **_state_dict()}


@app.delete("/api/files/{name}", summary="Delete a single uploaded file")
def api_delete_file(
    name: str,
    _admin: dict[str, Any] = Depends(auth.require_admin),
) -> dict[str, Any]:
    """Remove one file by name. Returns 404 for unknown names or path tricks."""
    # Reject any path tricks
    known_names = {f["name"] for f in store.file_infos()}
    if name != _sanitise(name) or name not in known_names:
        raise HTTPException(status_code=404, detail=f"File '{name}' not found.")
    store.remove(name)
    store.try_build()
    return _state_dict()


@app.delete("/api/files", summary="Delete all uploaded files")
def api_delete_all(
    _admin: dict[str, Any] = Depends(auth.require_admin),
) -> dict[str, Any]:
    """Remove every uploaded file and reset the pipeline."""
    store.clear()
    store.try_build()
    return _state_dict()


@app.post("/api/samples", summary="Load built-in sample lecture notes")
def api_samples(
    _admin: dict[str, Any] = Depends(auth.require_admin),
) -> dict[str, Any]:
    """Copy data/notes/*.md into the library and index them."""
    notes_dir = Path(config.NOTES_DIR)
    if not notes_dir.is_dir():
        raise HTTPException(
            status_code=404, detail="Sample notes directory not found."
        )

    lib = Path(config.LIBRARY_DIR)
    lib.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(notes_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED:
            continue
        dest = lib / src.name
        shutil.copy2(src, dest)
        try:
            store._chunks[src.name] = load_document(dest)
            copied += 1
        except Exception as exc:
            logger.warning("Skipping sample %s: %s", src.name, exc)

    if copied:
        store.try_build()
    else:
        logger.warning("No sample notes were loaded from %s", notes_dir)

    return _state_dict()


# ---------------------------------------------------------------------------
# NDJSON streaming chat
# ---------------------------------------------------------------------------

def _build_source(chunk: Chunk, trace: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chunk + trace info into the wire source object."""
    rerank_scores: dict[str, float] = trace.get("rerank_scores", {})
    paths: dict[str, dict] = trace.get("paths", {})
    used_ids: list[str] = trace.get("used", [])
    path = paths.get(chunk.id, {})
    return {
        "id": chunk.id,
        "source": chunk.source,
        "section": chunk.section,
        "page": chunk.page,
        "text": chunk.body,
        "score": rerank_scores.get(chunk.id),
        "dense_rank": path.get("dense"),
        "sparse_rank": path.get("sparse"),
        "used": chunk.id in used_ids,
    }


def _tokenize_terms(query: str) -> list[str]:
    """Return unique lowercase tokens from the query string for UI highlighting."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


_SENTINEL = object()  # signals worker finished


def _run_pipeline_worker(
    pipeline: RAGPipeline,
    question: str,
    history: list[dict[str, str]],
    evt_queue: "queue.Queue[Any]",
    stop_event: threading.Event,
) -> None:
    """Run the RAG pipeline in a worker thread, pushing events onto evt_queue."""

    def on_event(status: str, trace: dict[str, Any]) -> None:
        if stop_event.is_set():
            return
        evt_queue.put({"type": "trace", "status": status, "trace": trace})

    def on_token(piece: str = "", reset: bool = False) -> None:
        if stop_event.is_set():
            return
        if reset:
            evt_queue.put({"type": "reset"})
        else:
            evt_queue.put({"type": "token", "text": piece})

    try:
        result: Result = pipeline.answer(
            question=question,
            history=history,
            on_event=on_event,
            on_token=on_token,
        )
        evt_queue.put({"type": "_result", "result": result})
    except Exception as exc:
        evt_queue.put({"type": "error", "message": str(exc)})
    finally:
        evt_queue.put(_SENTINEL)


async def _chat_stream(
    pipeline: RAGPipeline,
    question: str,
    history: list[dict[str, str]],
) -> AsyncGenerator[bytes, None]:
    """Async NDJSON generator bridging the blocking pipeline worker thread.

    Guarantees: sources event is emitted BEFORE any token event.
    Token events are buffered until the sources event can be sent.
    """

    evt_queue: "queue.Queue[Any]" = queue.Queue()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=_run_pipeline_worker,
        args=(pipeline, question, history, evt_queue, stop_event),
        daemon=True,
    )
    worker.start()

    sources_sent = False
    result: Result | None = None
    trace_snap: dict[str, Any] = {}
    # Buffer token events until sources have been emitted
    pending_tokens: list[bytes] = []

    def _emit_sources(tr: dict[str, Any]) -> bytes:
        """Build and return the sources NDJSON line."""
        by_id = {c.id: c for c in pipeline.chunks}
        retrieved_ids: list[str] = tr.get("retrieved", [])
        sources_objs = [
            _build_source(by_id[cid], tr)
            for cid in retrieved_ids
            if cid in by_id
        ]
        terms = _tokenize_terms(tr.get("query", question))
        return (
            json.dumps({"type": "sources", "sources": sources_objs, "terms": terms})
            + "\n"
        ).encode()

    try:
        while True:
            # Drain queue with a small timeout so we stay responsive
            try:
                item = evt_queue.get(timeout=0.05)
            except queue.Empty:
                await asyncio.sleep(0)
                continue

            if item is _SENTINEL:
                break

            if isinstance(item, dict):
                kind = item.get("type")

                if kind == "trace":
                    trace_snap = item.get("trace", {})
                    # Emit sources once 'used' list appears in the trace
                    # (set by pipeline just before it calls on_token for generation)
                    if not sources_sent and "used" in trace_snap:
                        sources_sent = True
                        yield _emit_sources(trace_snap)
                        # Flush any buffered tokens now
                        for tok_bytes in pending_tokens:
                            yield tok_bytes
                        pending_tokens.clear()
                    yield (
                        json.dumps({
                            "type": "trace",
                            "status": item["status"],
                            "trace": trace_snap,
                        }) + "\n"
                    ).encode()

                elif kind == "token":
                    tok_bytes = (
                        json.dumps({"type": "token", "text": item["text"]}) + "\n"
                    ).encode()
                    if sources_sent:
                        yield tok_bytes
                    else:
                        # Buffer until sources are sent
                        pending_tokens.append(tok_bytes)

                elif kind == "reset":
                    pending_tokens.clear()  # discard buffered draft tokens too
                    yield (json.dumps({"type": "reset"}) + "\n").encode()

                elif kind == "_result":
                    result = item["result"]

                elif kind == "error":
                    msg = item.get("message", "Unknown error")
                    if "connect" in msg.lower() or "connection" in msg.lower():
                        msg = (
                            f"Ollama is not reachable at {config.OLLAMA_HOST}. "
                            "Start Ollama and retry."
                        )
                    yield (
                        json.dumps({"type": "error", "message": msg}) + "\n"
                    ).encode()
                    return

            await asyncio.sleep(0)  # yield control between events

        # Emit sources if they weren't emitted yet (e.g. no-retrieval path)
        if not sources_sent and result is not None:
            sources_sent = True
            yield _emit_sources(result.trace)
            # Flush buffered tokens that arrived before sources
            for tok_bytes in pending_tokens:
                yield tok_bytes
            pending_tokens.clear()
        elif pending_tokens:
            # sources already sent but tokens were somehow still buffered: flush them
            for tok_bytes in pending_tokens:
                yield tok_bytes
            pending_tokens.clear()

        if result is not None:
            tr = result.trace
            by_id = {c.id: c for c in pipeline.chunks}
            retrieved_ids = tr.get("retrieved", [])
            final_sources = [
                _build_source(by_id[cid], tr)
                for cid in retrieved_ids
                if cid in by_id
            ]
            terms = _tokenize_terms(tr.get("query", question))
            done_evt = {
                "type": "done",
                "answer": result.answer,
                "used_retrieval": result.used_retrieval,
                "sources": final_sources,
                "terms": terms,
                "trace": tr,
            }
            yield (json.dumps(done_evt) + "\n").encode()

    except asyncio.CancelledError:
        # Client disconnected – stop the worker cleanly
        stop_event.set()
        raise
    finally:
        stop_event.set()


@app.post("/api/chat", summary="Stream a RAG answer as NDJSON")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    """Stream NDJSON events: trace → sources → token… → done.

    Returns 409 if no pipeline is ready, 422 for invalid mode.
    """
    if not store.ready:
        error_msg = store.error or "No documents have been uploaded yet."
        raise HTTPException(status_code=409, detail=error_msg)

    try:
        pipeline = store.pipeline_for(req.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    history_dicts = [item.model_dump() for item in req.history]

    return StreamingResponse(
        _chat_stream(pipeline, req.question, history_dicts),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat With Your Docs – API server"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Bind port (default: 8000)"
    )
    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=False,
    )
