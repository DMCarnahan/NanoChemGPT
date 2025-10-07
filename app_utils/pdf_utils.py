"""PDF utilities: normalization and extraction helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Dict, Any
import time
import json
import logging


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
    t0 = time.time()
    log: Dict[str, Any] = {"event": "pdf_extract", "file": path.name, "max_pages": max_pages}
    backend = None
    try:
        # Try pdfminer.six first
        try:
            from pdfminer.high_level import extract_text as _pdfminer_extract  # type: ignore

            txt = _pdfminer_extract(str(path))
            if txt:
                backend = "pdfminer"
                pages = txt.split("\f")
                pages = pages[:max_pages]
                out = "\n".join(pages)
                log.update({"ok": True, "backend": backend, "pages": len(pages)})
                return (normalize_pdf_text(out), len(pages))
        except Exception as e_pdfminer:
            log["pdfminer_error"] = str(e_pdfminer)

        # pypdf / PyPDF2 fallback
        _PdfReader = None
        try:
            from pypdf import PdfReader as _PdfReader  # type: ignore
            backend = "pypdf"
        except Exception as e_pypdf:
            log["pypdf_error"] = str(e_pypdf)
            try:
                from PyPDF2 import PdfReader as _PdfReader  # type: ignore
                backend = "PyPDF2"
            except Exception as e_pypdf2:
                log["pypdf2_error"] = str(e_pypdf2)
                _PdfReader = None

        if _PdfReader is not None:
            try:
                reader = _PdfReader(str(path))
                pages = getattr(reader, "pages", [])
                out = []
                for i, page in enumerate(pages, 1):
                    if i > max_pages:
                        break
                    try:
                        out.append(page.extract_text() or "")
                    except Exception as pe:
                        out.append("")
                        log.setdefault("page_errors", 0)
                        log["page_errors"] += 1
                log.update({"ok": True, "backend": backend, "pages": min(len(pages), max_pages)})
                return (normalize_pdf_text("\n".join(out)), len(pages))
            except Exception as e_reader:
                log["reader_error"] = str(e_reader)

        raise ImportError("No PDF backend available")
    except Exception as e:
        log.update({"ok": False, "error": str(e)})
        logging.getLogger(__name__).warning(json.dumps(log))
        return ("", 0)
    finally:
        if "ok" not in log:
            log["ok"] = True
        log["elapsed_ms"] = int((time.time() - t0) * 1000)
        logging.getLogger(__name__).info(json.dumps(log))


def clean_attachment_text(s: str) -> str:
    # remove our chunk headers like: [A1.3] attachment:XYZ
    s = re.sub(r"^\[A\d+\.\d+\]\s*attachment:[^\n]+\n", "", s, flags=re.M)
    s = s.replace("\x03", "·")
    return s
