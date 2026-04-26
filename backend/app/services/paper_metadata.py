"""Deterministic paper metadata extraction from parsed paper content."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_SPACE_RE = re.compile(r"[\s\u00a0\u2000-\u200f\u202f\u205f\u3000]+")
_DOI_RE = re.compile(r"\b(doi|https?://doi\.org|arxiv)\b", re.IGNORECASE)
_TITLE_MAX_CHARS = 280


def filename_to_title(filename: str) -> str:
    """Return a readable title fallback when parsed metadata is unavailable."""
    stem = Path(filename).stem or filename
    title = re.sub(r"[_\-]+", " ", stem).strip()
    return title or filename


def extract_paper_title(parsed: Any, filename: str = "", pdf_path: str | None = None) -> str:
    """Extract the paper title without requiring an LLM.

    MinerU often places the first-page journal header, DOI, title, authors, and
    abstract into one text block. This extractor therefore skips common front
    matter and stops once author or abstract-like lines begin.
    """
    title = _extract_from_content_list(getattr(parsed, "content_list", []) or [])
    if title:
        return title

    title = _extract_from_markdown(getattr(parsed, "markdown_content", "") or "")
    if title:
        return title

    title = _extract_from_pdf_metadata(pdf_path)
    if title:
        return title

    return filename_to_title(filename)


def _extract_from_content_list(content_list: list[dict[str, Any]]) -> str:
    first_page_texts: list[str] = []
    early_texts: list[str] = []

    for item in content_list:
        if item.get("type") != "text":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        page_idx = _safe_int(item.get("page_idx"), default=0)
        if page_idx == 0:
            first_page_texts.append(text)
        if page_idx <= 1:
            early_texts.append(text)

    for text in [*first_page_texts, *early_texts[:3]]:
        title = _extract_from_text(text)
        if title:
            return title
    return ""


def _extract_from_markdown(markdown: str) -> str:
    if not markdown.strip():
        return ""
    first_lines = "\n".join(markdown.splitlines()[:90])
    return _extract_from_text(first_lines)


def _extract_from_text(text: str) -> str:
    lines = _split_clean_lines(text)
    if not lines:
        return ""

    marker_index = -1
    for index, line in enumerate(lines[:50]):
        if _DOI_RE.search(line):
            marker_index = index

    if marker_index >= 0:
        title = _collect_title(lines[marker_index + 1 :])
        if title:
            return title

    return _collect_title(lines[:45])


def _collect_title(lines: list[str]) -> str:
    collected: list[str] = []

    for line in lines:
        if _is_boilerplate_line(line):
            if collected:
                break
            continue

        if not collected and _is_short_lowercase_header(line):
            continue

        if collected and (_is_stop_line(line) or _looks_like_author_line(line)):
            break

        if not collected and (_is_stop_line(line) or _looks_like_author_line(line)):
            continue

        if len(line) > 180 and not collected:
            continue

        collected.append(line)
        joined = _join_title_lines(collected)
        if len(joined) > _TITLE_MAX_CHARS or len(collected) >= 5:
            break

    title = _join_title_lines(collected)
    return title if _is_valid_title(title) else ""


def _split_clean_lines(text: str) -> list[str]:
    cleaned = text.replace("\ufeff", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = _clean_line(raw_line)
        if line:
            lines.append(line)
    return lines


def _clean_line(line: str) -> str:
    normalized = _SPACE_RE.sub(" ", line)
    normalized = re.sub(r"^[#\s]+", "", normalized)
    return normalized.strip(" \t")


def _join_title_lines(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+([,:;.!?])", r"\1", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip(" .")


def _is_boilerplate_line(line: str) -> bool:
    lower = line.lower().strip()
    if not lower:
        return True
    if lower in {
        "article",
        "research article",
        "review",
        "review article",
        "communication",
        "communications",
        "letter",
        "letters",
        "open access",
        "check for updates",
    }:
        return True
    if re.fullmatch(r"(page\s*)?\d{1,4}", lower):
        return True
    if lower.startswith(("page ", "copyright", "©", "submitted", "revised")):
        return True
    if _DOI_RE.search(line):
        return True
    if " e-mail:" in f" {lower}" or lower.startswith("e-mail:"):
        return True
    if "|" in line and re.search(r"\b(volume|vol\.|issue|pages?|pp\.)\b", lower):
        return True
    if re.search(r"\b(volume|vol\.)\s+\d+", lower) and re.search(r"\d", lower):
        return True
    return False


def _is_short_lowercase_header(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    return bool(words) and len(words) <= 5 and line == line.lower()


def _is_stop_line(line: str) -> bool:
    lower = line.lower().strip()
    return lower.startswith(
        (
            "abstract",
            "keywords",
            "received:",
            "accepted:",
            "published:",
            "published online:",
            "correspondence",
            "affiliations",
            "introduction",
        )
    )


def _looks_like_author_line(line: str) -> bool:
    lower = line.lower()
    if "@" in line or "author" in lower and "contribution" not in lower:
        return True
    if re.search(r"\b[A-Z][A-Za-z'’.-]+\s+[A-Z][A-Za-z'’.-]+[0-9*,†‡]*", line):
        if "," in line or " & " in line or " and " in lower:
            return True
    if re.search(r"[A-Za-z][A-Za-z'’.-]+[0-9*,†‡]{1,}", line) and (
        "," in line or " & " in line
    ):
        return True
    return False


def _is_valid_title(title: str) -> bool:
    if not (8 <= len(title) <= _TITLE_MAX_CHARS):
        return False
    lower = title.lower()
    if _DOI_RE.search(title) or lower.startswith(("abstract", "keywords", "article")):
        return False
    if "@" in title:
        return False
    alpha_count = sum(char.isalpha() for char in title)
    if alpha_count < 6:
        return False
    words = re.findall(r"[\w\-]+", title, re.UNICODE)
    return 2 <= len(words) <= 45


def _extract_from_pdf_metadata(pdf_path: str | None) -> str:
    if not pdf_path:
        return ""
    try:
        import fitz  # type: ignore

        with fitz.open(pdf_path) as doc:
            title = _clean_line(str((doc.metadata or {}).get("title") or ""))
    except Exception:
        return ""

    if not title or title.lower() in {"untitled", "unknown"}:
        return ""
    return title if _is_valid_title(title) else ""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default
