"""PDF utilities: normalization and extraction helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


def normalize_pdf_text(s: str) -> str:
    if not s:
        return ""
    # Remove zero-width and control junk
    s = s.replace("\u200b", "").replace("\ufeff", "")
    # Collapse mid-dots
    s = re.sub(r"(?<=\w)[·∙•](?=\w)", "", s)
    s = s.replace("∙", "").replace("•", "").replace("·", "")
    s = s.replace("\x03", "·")
    # Normalize whitespace/hyphenation across line breaks
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def extract_pdf_text(path: Path, max_pages: int = 40) -> Tuple[str, int]:
    try:
        # Try pdfminer.six first
        try:
            from pdfminer.high_level import extract_text as _pdfminer_extract

            txt = _pdfminer_extract(str(path))
            if txt:
                pages = txt.split("\f")
                pages = pages[:max_pages]
                out = "\n".join(pages)
                return (normalize_pdf_text(out), len(pages))
        except Exception:
            pass

        # pypdf / PyPDF2 fallback
        try:
            from pypdf import PdfReader as _PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader as _PdfReader
            except Exception:
                _PdfReader = None

        if _PdfReader is not None:
            reader = _PdfReader(str(path))
            pages = getattr(reader, "pages", [])
            out = []
            for i, page in enumerate(pages, 1):
                if i > max_pages:
                    break
                try:
                    out.append(page.extract_text() or "")
                except Exception:
                    out.append("")
            return (normalize_pdf_text("\n".join(out)), len(pages))
        else:
            raise ImportError("No PDF backend available")
    except Exception as e:
        try:
            import logging

            logging.getLogger(__name__).warning(f"[extract_pdf_text] {path.name}: {e}")
        except Exception:
            pass
        return ("", 0)


def clean_attachment_text(s: str) -> str:
    # remove our chunk headers like: [A1.3] attachment:XYZ
    s = re.sub(r"^\[A\d+\.\d+\]\s*attachment:[^\n]+\n", "", s, flags=re.M)
    s = s.replace("\x03", "·")
    return s
