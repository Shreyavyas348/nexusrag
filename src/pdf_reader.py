"""Download PDFs from arXiv and extract text with optional section tagging."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import requests

logger = logging.getLogger(__name__)

# Max characters to process per paper (avoid huge RAM / slow embeds)
MAX_TEXT_CHARS = 200_000

_REF_SPLIT = re.compile(r"\breferences?\s*$", re.IGNORECASE | re.MULTILINE)

# (canonical_key, regex matching a line as section start)
_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "intro",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(introduction|background|motivation|related\s*work)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "lit_review",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(literature\s+review|related\s*work|state\s+of\s+the\s+art)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "method",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(methodology|methods?|materials?\s+and\s+methods?|"
            r"proposed\s+(method|approach|model|framework)|experimental\s+(setup|design))\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "results",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(results?|experiments?|evaluation|empirical\s+results?|"
            r"performance\s+analysis|ablation\s+study)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(discussion|analysis|interpretations?)\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^\s*(?:\d+\.?\s*)?(conclusion|conclusions?|discussion\s+and\s+conclusion|"
            r"summary|future\s+work)\s*$",
            re.IGNORECASE,
        ),
    ),
]


def _abs_to_pdf_url(link: str) -> str:
    link = link.strip()
    if "/abs/" in link:
        tail = link.split("/abs/", 1)[-1].rstrip("/")
        return f"https://arxiv.org/pdf/{tail}.pdf"
    if link.endswith(".pdf"):
        return link
    return link


def read_pdf_bytes(link: str, timeout: float = 45.0) -> Optional[bytes]:
    pdf_url = _abs_to_pdf_url(link)
    logger.info("Downloading PDF: %s", pdf_url)
    try:
        r = requests.get(pdf_url, timeout=timeout)
        r.raise_for_status()
        if not r.content or len(r.content) < 100:
            logger.warning("Downloaded PDF content too small or empty: %s", pdf_url)
            return None
        logger.info("Successfully downloaded %d bytes from %s", len(r.content), pdf_url)
        return r.content
    except requests.RequestException as e:
        logger.warning("PDF download failed for %s: %s", pdf_url, e)
        return None


def extract_raw_text(pdf_bytes: bytes) -> str:
    text_parts: List[str] = []
    logger.info("Starting PDF text extraction with pdfplumber...")
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            max_pages = 20
            logger.info("PDF has %d pages. Processing first %d pages.", len(pdf.pages), min(len(pdf.pages), max_pages))
            for i, page in enumerate(pdf.pages[:max_pages]):
                t = page.extract_text()
                if t and t.strip():
                    text_parts.append(t.strip())
                else:
                    logger.debug("Page %d yielded no text.", i+1)
    except Exception as e:
        logger.error("pdfplumber failed during extraction: %s", e)
        return ""
        
    full = "\n\n".join(text_parts)
    m = _REF_SPLIT.search(full)
    if m:
        logger.debug("Found References section; trimming content.")
        full = full[: m.start()]
        
    # Preserve newlines for section headers; collapse spaces within lines only
    norm_lines = []
    for line in full.split("\n"):
        norm_lines.append(re.sub(r"[ \t]+", " ", line).strip())
    full = "\n".join(norm_lines)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    
    if len(full) > MAX_TEXT_CHARS:
        logger.warning("Extracted text exceeds max limit. Trimming to %d chars.", MAX_TEXT_CHARS)
        full = full[:MAX_TEXT_CHARS]
        
    logger.info("Extraction complete. Length: %d characters.", len(full))
    return full


def extract_structured_sections(text: str) -> Dict[str, str]:
    """
    Heuristic section split by first-line headers. Keys: intro, method, results, conclusion.
    Missing sections return empty string.
    """
    out = {"intro": "", "lit_review": "", "method": "", "results": "", "discussion": "", "conclusion": ""}
    if not text.strip():
        return out

    lines = text.split("\n")
    current: Optional[str] = None
    buffers: Dict[str, List[str]] = {k: [] for k in out}

    for line in lines:
        stripped = line.strip()
        matched_key: Optional[str] = None
        for key, pat in _SECTION_PATTERNS:
            if pat.match(stripped):
                matched_key = key
                break
        if matched_key:
            current = matched_key
            continue
        if current:
            buffers[current].append(line)

    found_sections = []
    for k in out:
        joined = "\n".join(buffers[k]).strip()
        if joined:
            out[k] = joined[:8000]
            found_sections.append(k)
        else:
            out[k] = ""

    logger.debug("Sections detected: %s", found_sections)
    return out


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_chars: int = 80,
) -> List[str]:
    """Character-based chunks with overlap; skips trivially small segments."""
    text = (text or "").strip()
    if len(text) < min_chunk_chars:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if len(piece) >= min_chunk_chars:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def load_paper_document(link: str, fallback_summary: str) -> Dict[str, Any]:
    """
    Returns dict: full_text, sections (intro/method/results/conclusion), source (pdf|abstract).
    """
    pdf_bytes = read_pdf_bytes(link)
    full_text = ""
    source = "abstract"

    if pdf_bytes:
        full_text = extract_raw_text(pdf_bytes)
        if full_text:
            source = "pdf"
            logger.info("Successfully loaded content from PDF.")

    if not full_text.strip() and (fallback_summary or "").strip():
        logger.info("Falling back to paper abstract text.")
        full_text = re.sub(r"\s+", " ", fallback_summary.strip())
        source = "abstract"

    if source == "pdf":
        sections = extract_structured_sections(full_text)
    else:
        # Abstract-only: one body; avoid duplicating the same text as labeled sections
        sections = {"intro": "", "method": "", "results": "", "conclusion": ""}

    return {
        "full_text": full_text.strip(),
        "sections": sections,
        "source": source,
    }
