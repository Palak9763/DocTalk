"""Test doubles and fake client implementations for unit tests."""

import hashlib
from pathlib import Path
from typing import Any, Generator

import numpy as np
from reportlab.pdfgen import canvas

from rag import config, llm


class FakeEmbedResponse:
    """Container for mock embeddings matching Ollama response schema."""

    def __init__(self, embeddings: list[list[float]]):
        self.embeddings = embeddings

    def __getitem__(self, item: str) -> Any:
        if item == "embeddings":
            return self.embeddings
        raise KeyError(item)


class FakeClient:
    """Mock Ollama client for deterministic offline testing."""

    def __init__(self, canned_chat: str = "Canned response for testing."):
        self.canned_chat = canned_chat
        self.calls: list[dict[str, Any]] = []

    def embed(
        self,
        model: str,
        input: list[str] | str,
        keep_alive: str | None = None,
    ) -> FakeEmbedResponse:
        inputs = [input] if isinstance(input, str) else list(input)
        self.calls.append({
            "method": "embed",
            "model": model,
            "input": inputs,
            "keep_alive": keep_alive,
        })
        embeddings: list[list[float]] = []
        for text in inputs:
            vec = np.zeros(128, dtype=np.float32)
            words = text.split()
            if not words:
                vec[0] = 1.0
            else:
                for w in words:
                    h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % 128
                    vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())

        return FakeEmbedResponse(embeddings)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> Any:
        self.calls.append({
            "method": "chat",
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            "keep_alive": keep_alive,
        })
        if stream:
            def _gen() -> Generator[dict[str, Any], None, None]:
                for word in self.canned_chat.split():
                    yield {"message": {"content": word + " "}}
            return _gen()
        return {"message": {"content": self.canned_chat}}


def install(index_dir: Path | str) -> FakeClient:
    """Install FakeClient into llm module and point INDEX_DIR to index_dir."""
    fake = FakeClient()
    llm._client = fake
    config.INDEX_DIR = Path(index_dir)
    llm.reset_embedding_cache()
    return fake


def make_pdf(path: Path | str, pages: list[str]) -> None:
    """Generate a valid multi-page PDF document with specified page texts."""
    c = canvas.Canvas(str(path))
    for page_text in pages:
        if page_text:
            text_obj = c.beginText(72, 720)
            for line in page_text.splitlines():
                text_obj.textLine(line)
            c.drawText(text_obj)
        c.showPage()
    c.save()
