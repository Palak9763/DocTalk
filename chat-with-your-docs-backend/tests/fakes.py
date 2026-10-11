"""Test doubles and fake client implementations for unit tests."""

import hashlib
import json
from pathlib import Path
import re
import types
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
        self.log: list[str] = []

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

        system_prompt = ""
        user_prompt = ""
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content", "")
            elif m.get("role") == "user":
                user_prompt = m.get("content", "")

        sys_lower = system_prompt.lower()
        content = self.canned_chat

        if self.canned_chat != "Canned response for testing.":
            if "rewrite" in sys_lower or "reformulation" in sys_lower:
                self.log.append("rewrite")
            elif "expansion" in sys_lower or "expand" in sys_lower:
                self.log.append("expand")
            elif "hypothetical" in sys_lower or "lecture note" in sys_lower or "hyde" in sys_lower:
                self.log.append("hyde")
            else:
                self.log.append("chat")
            content = self.canned_chat
        elif "rewrite" in sys_lower or "reformulation" in sys_lower:
            self.log.append("rewrite")
            # Extract question and history
            q_match = re.search(r"Follow-up question:\s*(.*?)(?:\nStandalone question:|\Z)", user_prompt, re.DOTALL)
            hist_match = re.search(r"Conversation history:\s*(.*?)(?:\n\nFollow-up question:|\Z)", user_prompt, re.DOTALL)
            question = q_match.group(1).strip() if q_match else user_prompt.strip()
            history_text = hist_match.group(1).strip() if hist_match else ""

            # Find subject from history (e.g. from "What is backpropagation?" -> "backpropagation")
            topic = "the subject"
            if "backpropagation" in history_text.lower():
                topic = "backpropagation"
            elif "relu" in history_text.lower():
                topic = "ReLU"
            elif "adam" in history_text.lower():
                topic = "Adam optimizer"
            else:
                m_hist = re.findall(r"(?:what is|about|explains?)\s+([a-zA-Z0-9_\- ]+?)(?:\?|\.|\n|$)", history_text, re.IGNORECASE)
                if m_hist:
                    topic = m_hist[-1].strip()

            # Replace pronoun in question
            rewritten = re.sub(r"\b(it|this|that)\b", topic, question, count=1, flags=re.IGNORECASE)
            rewritten = re.sub(r"\bits\b", f"{topic}'s", rewritten, count=1, flags=re.IGNORECASE)
            # Capitalize first letter
            if rewritten:
                rewritten = rewritten[0].upper() + rewritten[1:]
            content = rewritten

        elif "expansion" in sys_lower or "expand" in sys_lower:
            self.log.append("expand")
            q_match = re.search(r'for:\s*"(.*?)"', user_prompt)
            query = q_match.group(1).strip() if q_match else user_prompt.strip()
            content = json.dumps({"queries": [f"{query} mechanism", f"{query} derivation"]})

        elif "hypothetical" in sys_lower or "lecture note" in sys_lower or "hyde" in sys_lower:
            self.log.append("hyde")
            q_match = re.search(r"Query:\s*(.*?)(?:\nLecture passage:|\Z)", user_prompt, re.DOTALL)
            query = q_match.group(1).strip() if q_match else user_prompt.strip()
            content = f"In machine learning, {query} describes a core mathematical technique used throughout lecture notes."

        else:
            self.log.append("chat")

        if stream:
            def _gen() -> Generator[dict[str, Any], None, None]:
                for word in content.split():
                    yield {"message": {"content": word + " "}}
            return _gen()
        return {"message": {"content": content}}


_ORIGINAL_RERANKER_SCORE: Any = None


def install_fake_reranker(target: Any = None) -> None:
    """Install a word-overlap fake scorer (values in [0, 1]) on Reranker without loading any model."""
    global _ORIGINAL_RERANKER_SCORE
    from rag.retrieve import Reranker

    if _ORIGINAL_RERANKER_SCORE is None:
        _ORIGINAL_RERANKER_SCORE = Reranker.score

    def fake_score(self: Any, query: str, chunks: Any, idxs: list[int]) -> list[float]:
        if not idxs:
            return []
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scores: list[float] = []
        for i in idxs:
            c_text = chunks[i].text.lower()
            c_tokens = set(re.findall(r"\w+", c_text))
            if not q_tokens or not c_tokens:
                scores.append(0.0)
            else:
                overlap = len(q_tokens & c_tokens)
                scores.append(min(1.0, overlap / len(q_tokens)))
        return scores

    if target is not None:
        target.score = types.MethodType(fake_score, target)
    else:
        Reranker.score = fake_score


def restore_reranker() -> None:
    """Restore original Reranker.score method if monkeypatched."""
    global _ORIGINAL_RERANKER_SCORE
    from rag.retrieve import Reranker

    if _ORIGINAL_RERANKER_SCORE is not None:
        Reranker.score = _ORIGINAL_RERANKER_SCORE



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
