"""Terminal CLI for DocTalk – interactive or one-shot Q&A against local documents."""

import argparse
import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# Ensure project root is in sys.path when run as a script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rag.config import PRESETS
from rag.ingest import SUPPORTED, load_chunks, load_document
from rag.pipeline import RAGPipeline
from rag.retrieve import Reranker
from tests.fakes import install_fake_reranker


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers (ANSI – Windows 10+ with ENABLE_VIRTUAL_TERMINAL_PROCESSING)
# ──────────────────────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_RED = "\033[91m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _RESET


def _check_ollama() -> bool:
    """Return True if Ollama is reachable."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def _load_docs(docs_path: str) -> list:
    """Load chunks from a file or directory."""
    p = Path(docs_path)
    if p.is_file():
        if p.suffix.lower() not in SUPPORTED:
            print(_c(f"Unsupported file type: {p.suffix}. Supported: {sorted(SUPPORTED)}", _RED))
            sys.exit(1)
        return load_document(p)
    elif p.is_dir():
        chunks = load_chunks(p)
        if not chunks:
            print(_c(f"No supported documents found in {p}", _YELLOW))
        return chunks
    else:
        print(_c(f"Path not found: {p}", _RED))
        sys.exit(1)


def _print_trace(trace: dict, show_trace: bool) -> None:
    """Print sources and optionally the full trace."""
    used_ids: list[str] = trace.get("used", [])
    contexts = trace.get("_contexts", [])

    # Always print sources
    if used_ids:
        print()
        print(_c("--- Sources -------------------------------------------", _DIM))
        by_id = {c.id: c for c in contexts}
        for cid in used_ids:
            c = by_id.get(cid)
            page_str = f"  page {c.page}" if c and c.page else ""
            print(f"  {_c(cid, _CYAN)}{page_str}")
    else:
        print(_c("\n(No sources used)", _DIM))

    if not show_trace:
        return

    print()
    print(_c("--- Trace -----------------------------------------------", _DIM))

    # Decision
    dec = trace.get("decision", {})
    if dec.get("enabled"):
        flag = _c("[YES] retrieve", _GREEN) if dec.get("retrieve") else _c("[NO] direct answer", _YELLOW)
        print(f"  Decision : {flag} -- {dec.get('reason', '')}")

    # Query rewriting / expansion
    orig = trace.get("original", "")
    rewritten = trace.get("query", orig)
    if orig and orig != rewritten:
        print(f"  Original : {orig}")
        print(f"  Rewritten: {_c(rewritten, _CYAN)}")
    queries = trace.get("queries", [])
    if len(queries) > 1:
        print(f"  Queries  : {', '.join(repr(q) for q in queries)}")
    if "hyde" in trace:
        print(f"  HyDE     : {trace['hyde'][:80]}...")

    # Search counts
    s = trace.get("search", {})
    if s:
        print(
            f"  Search   : dense={s.get('dense_hits',0)}  sparse={s.get('sparse_hits',0)}"
            f"  both={s.get('both',0)}  fused={s.get('fused',0)}"
            f"  dupes_removed={s.get('duplicates_removed',0)}"
            f"  ->reranker={s.get('to_reranker',0)}"
        )

    # Rerank scores
    rscores = trace.get("rerank_scores", {})
    if rscores:
        top5 = sorted(rscores.items(), key=lambda kv: -kv[1])[:5]
        score_str = "  ".join(f"{_c(k, _CYAN)}={v:.3f}" for k, v in top5)
        print(f"  Reranked : {score_str}")

    # Grading
    graded_out = trace.get("graded_out", [])
    if graded_out:
        print(f"  Graded[-] : {', '.join(graded_out)}")

    # Groundedness
    gnd = trace.get("groundedness")
    if gnd:
        gflag = _c("[OK] grounded", _GREEN) if gnd.get("grounded") else _c("[!] not grounded", _RED)
        print(f"  Grounded : {gflag}")
        if gnd.get("unsupported"):
            print(f"  Unsupported claims: {'; '.join(gnd['unsupported'][:3])}")

    if trace.get("regenerated"):
        print(_c("  [*] Answer was regenerated (groundedness fix)", _YELLOW))

    # Timings
    timings = trace.get("timings_ms", {})
    if timings:
        parts = [f"{k}={v}ms" for k, v in timings.items()]
        print(f"  Timings  : {', '.join(parts)}")


def _run_query(pipeline: RAGPipeline, question: str, history: list, show_trace: bool) -> str:
    """Ask a question and print the streamed answer; return the answer text."""
    print()
    print(_c(f"Q: {question}", _BOLD))
    print(_c("A: ", _GREEN), end="", flush=True)

    tokens: list[str] = []
    reset_received = [False]

    def on_token(piece: str, reset: bool = False) -> None:
        if reset:
            reset_received[0] = True
            print()
            print(_c("[Regenerating for groundedness…]", _YELLOW))
            print(_c("A: ", _GREEN), end="", flush=True)
            tokens.clear()
        else:
            print(piece, end="", flush=True)
            tokens.append(piece)

    def on_event(status: str, _trace: dict) -> None:
        # Print stage status inline on a separate line above the answer
        pass  # kept minimal; trace shown after answer

    result = pipeline.answer(
        question=question,
        history=history,
        on_event=on_event,
        on_token=on_token,
    )

    print()  # newline after streamed answer

    # Stash contexts for _print_trace
    result.trace["_contexts"] = result.contexts
    _print_trace(result.trace, show_trace)

    return result.answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocTalk – Chat with your documents (local RAG)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--docs", default="data/notes", help="File or folder of documents to chat with")
    parser.add_argument("--mode", choices=["fast", "balanced", "thorough"], default="balanced")
    parser.add_argument("--show-trace", action="store_true", help="Print retrieval trace after each answer")
    parser.add_argument("--ask", default=None, help="One-shot question (omit for interactive mode)")
    args = parser.parse_args()

    # Ollama check
    if not _check_ollama():
        print(_c(
            "⚠  Ollama is not reachable at http://localhost:11434.\n"
            "   Start it with: ollama serve\n"
            "   Then pull models: ollama pull qwen2.5:3b-instruct && ollama pull nomic-embed-text",
            _RED
        ))
        sys.exit(1)

    print(_c(f"DocTalk  [{args.mode}]  docs={args.docs}", _BOLD + _MAGENTA))

    # Load documents
    chunks = _load_docs(args.docs)
    if not chunks:
        print(_c("No documents loaded. Exiting.", _RED))
        sys.exit(1)
    print(_c(f"Loaded {len(chunks)} chunk(s) from {args.docs}", _DIM))

    # Build pipeline
    cfg = PRESETS[args.mode]
    pipeline = RAGPipeline(cfg=cfg, chunks=chunks)

    history: list = []

    if args.ask:
        # One-shot mode
        answer = _run_query(pipeline, args.ask, history, args.show_trace)
        history.append(("user", args.ask))
        history.append(("assistant", answer))
    else:
        # Interactive loop
        print(_c("Interactive mode — type your question (or 'exit' to quit)\n", _DIM))
        while True:
            try:
                question = input(_c("You: ", _BOLD + _CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                break
            answer = _run_query(pipeline, question, history, args.show_trace)
            history.append(("user", question))
            history.append(("assistant", answer))


if __name__ == "__main__":
    main()
