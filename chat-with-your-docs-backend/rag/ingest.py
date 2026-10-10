"""Document ingestion, section parsing, and sliding-window chunking."""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Literal

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import pypdf

logger = logging.getLogger(__name__)

SUPPORTED: set[str] = {".md", ".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    """A chunk of ingested document text with provenance metadata."""

    id: str
    source: str
    section: str
    text: str
    page: int | None
    prefix_len: int = 0

    @property
    def body(self) -> str:
        """Return the chunk text without the provenance context header."""
        return self.text[self.prefix_len :]


def chunk_text(text: str, words: int = 180, overlap: int = 40) -> list[str]:
    """Split text into word-level windows with fixed word overlap.

    Consecutive chunks share exactly `overlap` words. The final chunk
    extends to the end of the text. Never returns empty chunks.

    Raises:
        ValueError: If overlap is greater than or equal to words.
    """
    if overlap >= words:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than words ({words})"
        )
    if words <= 0 or overlap < 0:
        raise ValueError("words must be positive and overlap non-negative")

    word_tokens = text.split()
    if not word_tokens:
        return []

    step = words - overlap
    chunks: list[str] = []
    for start in range(0, len(word_tokens), step):
        chunk_words = word_tokens[start : start + words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + words >= len(word_tokens):
            break

    return chunks


def _load_markdown(path: Path) -> list[tuple[str, str, int | None]]:
    """Parse a markdown file into sections based on headings."""
    content = path.read_text(encoding="utf-8", errors="replace")
    sections: list[tuple[str, str, int | None]] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for line in content.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if match:
            text_so_far = "\n".join(current_lines).strip()
            if text_so_far:
                sections.append((current_section, text_so_far, None))
            current_section = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    text_so_far = "\n".join(current_lines).strip()
    if text_so_far:
        sections.append((current_section, text_so_far, None))

    return sections


def _load_txt(path: Path) -> list[tuple[str, str, int | None]]:
    """Parse a text file grouped into a section named after the file stem."""
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    if not blocks:
        return []
    return [(path.stem, "\n\n".join(blocks), None)]


def _load_pdf(path: Path) -> list[tuple[str, str, int | None]]:
    """Parse a PDF document with one section per page."""
    reader = pypdf.PdfReader(str(path))
    sections: list[tuple[str, str, int | None]] = []
    has_text = False

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = (page.extract_text() or "").strip()
        if page_text:
            has_text = True
            sections.append((f"Page {page_num}", page_text, page_num))

    if not has_text:
        raise ValueError(
            "No text layer found: scanned PDFs are not supported (no OCR)."
        )

    return sections


def _load_docx(path: Path) -> list[tuple[str, str, int | None]]:
    """Parse a DOCX document splitting sections by headings and preserving tables."""
    doc = Document(str(path))
    sections: list[tuple[str, str, int | None]] = []
    current_section = "Introduction"
    current_lines: list[str] = []

    for element in doc.element.body:
        if element.tag.endswith("p"):
            p = Paragraph(element, doc)
            style_name = p.style.name if p.style else ""
            text = p.text.strip()
            if style_name.startswith("Heading") and text:
                text_so_far = "\n".join(current_lines).strip()
                if text_so_far:
                    sections.append((current_section, text_so_far, None))
                current_section = text
                current_lines = []
            elif text:
                current_lines.append(text)
        elif element.tag.endswith("tbl"):
            tbl = Table(element, doc)
            for row in tbl.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_str = " | ".join(cells).strip()
                if row_str:
                    current_lines.append(row_str)

    text_so_far = "\n".join(current_lines).strip()
    if text_so_far:
        sections.append((current_section, text_so_far, None))

    return sections


def load_document(
    path: Path | str, words: int = 180, overlap: int = 40
) -> list[Chunk]:
    """Load a supported document and split its sections into overlapping chunks.

    Args:
        path: Path to the target document.
        words: Maximum words per chunk window.
        overlap: Words shared between consecutive chunk windows.

    Returns:
        List of Chunk objects with provenance headers.

    Raises:
        ValueError: If file suffix is unsupported or PDF lacks extractable text.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {sorted(SUPPORTED)}"
        )

    if suffix == ".md":
        sections = _load_markdown(file_path)
    elif suffix == ".txt":
        sections = _load_txt(file_path)
    elif suffix == ".pdf":
        sections = _load_pdf(file_path)
    elif suffix == ".docx":
        sections = _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'")

    chunks: list[Chunk] = []
    for sec_idx, (section_name, section_text, page_num) in enumerate(sections):
        raw_chunks = chunk_text(section_text, words=words, overlap=overlap)
        for j, raw_chunk in enumerate(raw_chunks):
            header = f"{file_path.name} › {section_name}\n"
            prefix_len = len(header)
            full_text = f"{header}{raw_chunk}"
            chunk_id = f"{file_path.name}#{sec_idx}.{j}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source=file_path.name,
                    section=section_name,
                    text=full_text,
                    page=page_num,
                    prefix_len=prefix_len,
                )
            )

    return chunks


def load_chunks(
    directory: Path | str, words: int = 180, overlap: int = 40
) -> list[Chunk]:
    """Load and chunk all supported documents in a directory sorted by filename."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.warning("Notes directory '%s' does not exist", dir_path)
        return []

    supported_files = [
        p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED
    ]
    supported_files.sort(key=lambda p: p.name)

    all_chunks: list[Chunk] = []
    for file_path in supported_files:
        all_chunks.extend(
            load_document(file_path, words=words, overlap=overlap)
        )

    return all_chunks
