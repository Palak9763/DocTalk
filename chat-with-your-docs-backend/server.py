"""Phase 5 + Auth: FastAPI REST API for Chat With Your Docs.

Run with:
    python server.py [--host 127.0.0.1] [--port 8000]

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import threading
from typing import Any, AsyncGenerator, Literal
import unicodedata

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

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
MAX_ACTIVE_PIPELINES: int = 5

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
    name = Path(name).name
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[^\w\s.\-]", "_", name, flags=re.UNICODE)
    name = name.strip()
    if not name or name in (".", ".."):
        name = "upload"
    return name


# ---------------------------------------------------------------------------
# Per-User Document Library & Registry
# ---------------------------------------------------------------------------

class UserLibrary:
    """Per-user document storage and isolated RAG pipeline.

    Operates in LIBRARY_DIR/<user_id>/. Uses an independent lock so uploads
    and queries for this user never block other users.
    """

    def __init__(self, user_id: str, reranker: Reranker) -> None:
        self.user_id = user_id
        self._reranker = reranker
        self._lock = threading.Lock()
        self._chunks: dict[str, list[Chunk]] = {}  # filename -> chunks
        self._pipeline: RAGPipeline | None = None
        self.ready: bool = False
        self.error: str | None = None
        self._scanned: bool = False

    @property
    def lib_dir(self) -> Path:
        return Path(config.LIBRARY_DIR) / self.user_id

    def scan(self) -> None:
        """Scan this user's directory and load all supported documents."""
        with self._lock:
            self.lib_dir.mkdir(parents=True, exist_ok=True)
            self._chunks.clear()
            for path in sorted(self.lib_dir.iterdir()):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED:
                    continue
                try:
                    self._chunks[path.name] = load_document(path)
                    logger.info("[%s] Scanned %s (%d chunks)", self.user_id, path.name, len(self._chunks[path.name]))
                except Exception as exc:
                    logger.warning("[%s] Skipping %s during scan: %s", self.user_id, path.name, exc)
            self._scanned = True

    def add(self, filename: str, data: bytes) -> list[Chunk]:
        """Write file into user's private library, parse, and update chunk cache."""
        self.lib_dir.mkdir(parents=True, exist_ok=True)
        dest = self.lib_dir / filename
        dest.write_bytes(data)
        try:
            chunks = load_document(dest)
        except Exception:
            if dest.exists():
                dest.unlink()
            raise
        with self._lock:
            self._chunks[filename] = chunks
        return chunks

    def remove(self, name: str) -> None:
        """Remove a file from disk and chunk cache for this user."""
        with self._lock:
            self._chunks.pop(name, None)
        path = self.lib_dir / name
        if path.exists():
            path.unlink()

    def clear(self) -> None:
        """Remove every file and chunk for this user."""
        with self._lock:
            self._chunks.clear()
        if self.lib_dir.exists():
            for p in self.lib_dir.iterdir():
                if p.is_file():
                    try:
                        p.unlink()
                    except Exception as exc:
                        logger.warning("[%s] Failed to remove %s: %s", self.user_id, p, exc)

    def _all_chunks(self) -> list[Chunk]:
        return [c for chunks in self._chunks.values() for c in chunks]

    def build(self) -> None:
        """(Re)build the RAGPipeline for this user using the shared Reranker."""
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
                "[%s] Pipeline rebuilt: %d chunks across %d files",
                self.user_id,
                len(chunks),
                len(self._chunks),
            )

    def try_build(self) -> None:
        """Safe build wrapper that records errors."""
        try:
            self.build()
        except Exception as exc:
            msg = str(exc)
            if "connect" in msg.lower() or "connection" in msg.lower():
                msg = f"Ollama is not reachable at {config.OLLAMA_HOST}. Start Ollama and retry."
            self.error = msg
            self.ready = False
            logger.error("[%s] Pipeline build failed: %s", self.user_id, msg)

    def pipeline_for(self, mode: str) -> RAGPipeline:
        """Return a config-cloned pipeline for the given preset mode."""
        with self._lock:
            if not self.ready or self._pipeline is None:
                # If pipeline was evicted or not yet built but files exist, rebuild lazily
                if self._chunks or any(self.lib_dir.glob("*")):
                    if not self._scanned:
                        self.scan()
                    self.build()
            if not self.ready or self._pipeline is None:
                raise RuntimeError("No pipeline is ready.")
            return self._pipeline.with_config(PRESETS[mode])

    def file_infos(self) -> list[dict[str, Any]]:
        """Return file metadata list for this user."""
        self.lib_dir.mkdir(parents=True, exist_ok=True)
        infos = []
        with self._lock:
            for name, chunks in self._chunks.items():
                path = self.lib_dir / name
                size = path.stat().st_size if path.exists() else 0
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

    def evict_pipeline(self) -> None:
        """Drop in-memory pipeline to enforce LRU cache budget."""
        with self._lock:
            self._pipeline = None
            self.ready = False
            logger.info("[%s] Pipeline evicted from memory (LRU cap)", self.user_id)


class LibraryRegistry:
    """Registry managing lazy UserLibrary instances and at most 5 active pipelines."""

    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker
        self._lock = threading.Lock()
        self._libraries: dict[str, UserLibrary] = {}
        self._active_lru: OrderedDict[str, None] = OrderedDict()

    def get_library(self, user_id: str) -> UserLibrary:
        with self._lock:
            if user_id not in self._libraries:
                lib = UserLibrary(user_id, self._reranker)
                lib.scan()
                lib.try_build()
                self._libraries[user_id] = lib
            lib = self._libraries[user_id]

            if not lib._scanned:
                lib.scan()
                lib.try_build()

            if lib.ready and lib._pipeline is not None:
                self._touch_lru(user_id)

            return lib

    def _touch_lru(self, user_id: str) -> None:
        if user_id in self._active_lru:
            self._active_lru.move_to_end(user_id)
        else:
            self._active_lru[user_id] = None

        while len(self._active_lru) > MAX_ACTIVE_PIPELINES:
            evicted_uid, _ = self._active_lru.popitem(last=False)
            if evicted_uid != user_id and evicted_uid in self._libraries:
                self._libraries[evicted_uid].evict_pipeline()

    def record_active(self, user_id: str) -> None:
        with self._lock:
            self._touch_lru(user_id)

    def reset(self) -> None:
        with self._lock:
            for lib in self._libraries.values():
                lib.evict_pipeline()
            self._libraries.clear()
            self._active_lru.clear()


# Module-level shared reranker and registry
_shared_reranker: Reranker = Reranker(config.RERANK_MODEL)
registry: LibraryRegistry = LibraryRegistry(_shared_reranker)

def get_library(user_id: str) -> UserLibrary:
    return registry.get_library(user_id)


_default_user_id: str = "default_test_user"

class _LegacyStoreProxy:
    """Backward compatibility shim for tests/code accessing server.store."""

    @property
    def _default_lib(self) -> UserLibrary:
        return registry.get_library(_default_user_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._default_lib, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_default_lib":
            super().__setattr__(name, value)
        else:
            setattr(self._default_lib, name, value)


store = _LegacyStoreProxy()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    mode: str = Field(
        default="balanced",
        description="Retrieval mode: fast, balanced, thorough",
    )
    history: list[HistoryItem] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn coreference resolution",
    )

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        if v not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        return v


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Initialize DB and warm up models. Does not scan all user libraries at startup."""
    auth.init_auth_db()

    # Pre-load shared reranker weights in background
    threading.Thread(
        target=_shared_reranker.load,
        daemon=True,
        name="reranker-preload",
    ).start()

    # Warm up LLM in background
    threading.Thread(
        target=llm.warm_up,
        daemon=True,
        name="llm-warmup",
    ).start()

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chat With Your Docs API",
    description=(
        "Local RAG backend – upload documents and chat with them. "
        "All inference runs locally via Ollama with per-user data isolation."
    ),
    version="1.0.0",
    lifespan=_lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
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

# Include Auth Router
app.include_router(auth.router)


# ---------------------------------------------------------------------------
# State helper
# ---------------------------------------------------------------------------

def _state_dict(lib: UserLibrary) -> dict[str, Any]:
    """Build the full API state payload for a specific user's library."""
    cfg = PipelineConfig()
    has_samples = False
    notes_dir = Path(config.NOTES_DIR)
    if notes_dir.is_dir():
        has_samples = any(
            p.is_file() and p.suffix.lower() in SUPPORTED
            for p in notes_dir.iterdir()
        )
    return {
        "files": lib.file_infos(),
        "ready": lib.ready,
        "error": lib.error,
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
# Public Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="API root")
def root() -> dict[str, str]:
    """Return basic API info and links."""
    return {
        "name": "Chat With Your Docs API",
        "docs": "/docs",
        "state": "/api/state",
    }


# ---------------------------------------------------------------------------
# Authenticated Routes (Per-User Data Isolation)
# ---------------------------------------------------------------------------

@app.get("/api/state", summary="Library & pipeline state")
def api_state(user: dict[str, Any] = Depends(auth.get_current_user)) -> dict[str, Any]:
    """Return user's library files, pipeline readiness, and model info."""
    lib = registry.get_library(user["id"])
    return _state_dict(lib)


@app.post("/api/upload", summary="Upload one or more documents")
async def api_upload(
    files: list[UploadFile] = File(...),
    user: dict[str, Any] = Depends(auth.get_current_user),
) -> dict[str, Any]:
    """Accept .pdf .docx .md .txt files (max 50 MB each).

    Operates strictly within the authenticated user's private library.
    """
    lib = registry.get_library(user["id"])
    results: list[dict[str, Any]] = []
    any_success = False

    for uf in files:
        raw_name = uf.filename or "upload"
        safe_name = _sanitise(raw_name)

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

        data = await uf.read()
        if len(data) > _MAX_FILE_BYTES:
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": f"File exceeds the {_MAX_FILE_BYTES // (1024 * 1024)} MB limit.",
            })
            continue

        try:
            chunks = lib.add(safe_name, data)
            results.append({
                "name": safe_name,
                "ok": True,
                "chunks": len(chunks),
                "error": None,
            })
            any_success = True
            logger.info("[%s] Uploaded %s – %d chunks", user["id"], safe_name, len(chunks))
        except ValueError as exc:
            lib.remove(safe_name)
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": str(exc),
            })
        except Exception:
            lib.remove(safe_name)
            logger.exception("[%s] Unexpected error ingesting %s", user["id"], safe_name)
            results.append({
                "name": safe_name,
                "ok": False,
                "chunks": 0,
                "error": "Internal error during ingestion.",
            })

    if any_success:
        lib.try_build()
        registry.record_active(user["id"])

    return {"results": results, **_state_dict(lib)}


@app.delete("/api/files/{name}", summary="Delete a single uploaded file")
def api_delete_file(
    name: str,
    user: dict[str, Any] = Depends(auth.get_current_user),
) -> dict[str, Any]:
    """Remove one file by name from current user's library.

    Returns 404 for unknown names or files belonging to other users.
    """
    lib = registry.get_library(user["id"])
    known_names = {f["name"] for f in lib.file_infos()}
    if name != _sanitise(name) or name not in known_names:
        raise HTTPException(status_code=404, detail=f"File '{name}' not found.")
    lib.remove(name)
    lib.try_build()
    registry.record_active(user["id"])
    return _state_dict(lib)


@app.delete("/api/files", summary="Delete all uploaded files")
def api_delete_all(
    user: dict[str, Any] = Depends(auth.get_current_user),
) -> dict[str, Any]:
    """Remove every uploaded file from current user's library and reset pipeline."""
    lib = registry.get_library(user["id"])
    lib.clear()
    lib.try_build()
    registry.record_active(user["id"])
    return _state_dict(lib)


@app.post("/api/samples", summary="Load built-in sample lecture notes")
def api_samples(
    user: dict[str, Any] = Depends(auth.get_current_user),
) -> dict[str, Any]:
    """Copy data/notes/*.md into current user's library and index them."""
    notes_dir = Path(config.NOTES_DIR)
    if not notes_dir.is_dir():
        raise HTTPException(status_code=404, detail="Sample notes directory not found.")

    lib = registry.get_library(user["id"])
    lib.lib_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in sorted(notes_dir.iterdir()):
        if not src.is_file() or src.suffix.lower() not in SUPPORTED:
            continue
        dest = lib.lib_dir / src.name
        shutil.copy2(src, dest)
        try:
            chunks = load_document(dest)
            with lib._lock:
                lib._chunks[src.name] = chunks
            copied += 1
        except Exception as exc:
            logger.warning("[%s] Skipping sample %s: %s", user["id"], src.name, exc)
            if dest.exists():
                dest.unlink()

    if copied:
        lib.try_build()
        registry.record_active(user["id"])
    else:
        logger.warning("[%s] No sample notes were loaded from %s", user["id"], notes_dir)

    return _state_dict(lib)


# ---------------------------------------------------------------------------
# Admin Routes
# ---------------------------------------------------------------------------

@app.get("/api/admin/users", summary="List all registered users (admin only)")
def api_admin_users(
    admin: dict[str, Any] = Depends(auth.require_admin),
) -> list[dict[str, Any]]:
    """Return all users with file counts. Never exposes password_hash."""
    conn = auth.get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, email, role, created_at, last_login FROM users ORDER BY created_at ASC"
        )
        users = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    lib_root = Path(config.LIBRARY_DIR)
    for u in users:
        user_dir = lib_root / u["id"]
        if user_dir.is_dir():
            u["file_count"] = sum(
                1 for p in user_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
            )
        else:
            u["file_count"] = 0

    return users


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


_SENTINEL = object()


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
    """Async NDJSON generator bridging the blocking pipeline worker thread."""
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
    pending_tokens: list[bytes] = []

    def _emit_sources(tr: dict[str, Any]) -> bytes:
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
            try:
                item = evt_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if item is _SENTINEL:
                break

            itype = item.get("type")

            if itype == "trace":
                trace_snap = item["trace"]
                yield (json.dumps(item) + "\n").encode()

                if not sources_sent and trace_snap.get("retrieved") is not None:
                    sources_sent = True
                    yield _emit_sources(trace_snap)
                    for pt in pending_tokens:
                        yield pt
                    pending_tokens.clear()

            elif itype == "token":
                line = (json.dumps(item) + "\n").encode()
                if sources_sent:
                    yield line
                else:
                    pending_tokens.append(line)

            elif itype == "reset":
                line = (json.dumps(item) + "\n").encode()
                if sources_sent:
                    yield line
                else:
                    pending_tokens.clear()

            elif itype == "error":
                yield (json.dumps(item) + "\n").encode()

            elif itype == "_result":
                result = item["result"]

        if not sources_sent:
            tr = result.trace if result else trace_snap
            yield _emit_sources(tr)
            for pt in pending_tokens:
                yield pt
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
        stop_event.set()
        raise
    finally:
        stop_event.set()


@app.post("/api/chat", summary="Stream a RAG answer as NDJSON")
async def api_chat(
    req: ChatRequest,
    user: dict[str, Any] = Depends(auth.get_current_user),
) -> StreamingResponse:
    """Stream NDJSON events for the current user's library and index."""
    lib = registry.get_library(user["id"])
    if not lib.ready:
        if lib.file_infos():
            lib.try_build()
    if not lib.ready:
        error_msg = lib.error or "No documents have been uploaded yet."
        raise HTTPException(status_code=409, detail=error_msg)

    try:
        pipeline = lib.pipeline_for(req.mode)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    registry.record_active(user["id"])
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
