"""PDF utilities: normalization and extraction helpers.

Normalization heuristics address noisy PDF text where every character may
be preceded by an interpunct/mid-dot (·, •, ∙, ⋅, ●). The goals:
1. Collapse artificial separator dots inserted between letters ("m·a·t·e·r·i·a·l").
2. Preserve chemically meaningful middle dots in hydrate / adduct formulas
     (e.g., "CuSO4·5H2O", "Na2SO4·10H2O").
3. Avoid over-cleaning that would join words incorrectly or strip legitimate
     Unicode characters.

Approach:
* Protect hydrate/adduct patterns by temporarily replacing the dot with a
    placeholder token before bulk dot stripping.
* Remove residual separator characters when they appear at high frequency or
    between single letters.
* Apply conservative whitespace and hyphenation normalization last.

If future legitimate mid-dot use cases surface (e.g., co-crystals like
"A·B"), extend the protection regex list below.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Dict, Any
import time
import json
import logging


def normalize_pdf_text(s: str) -> str:
    """Normalize noisy PDF extracted text.

    Removes pervasive mid-dots inserted between letters while preserving
    mid-dots in common hydrate/adduct chemical formulas (e.g., CuSO4·5H2O).
    """
    if not s:
        return ""

    original = s
    # Remove zero-width / BOM & control junk
    s = s.replace("\u200b", "").replace("\ufeff", "")

    # Characters considered spurious separators.
    sep_chars = "·∙•⋅●"  # (Common mid-dot variants)

    # Protect hydrate/adduct patterns by replacing the separator with a placeholder.
    # Patterns covered (case-sensitive, typical chemistry):
    #   CuSO4·5H2O, Na2SO4·10H2O, CaCl2·2H2O, CoCl2·6H2O, etc.
    # Also handle single hydrate ("CuSO4·H2O").
    HY_PLACEHOLDER = "<HYDOT>"

    # Accept optional internal separators inside the left part (e.g., C·u·S·O·4· ) by stripping them before reinsertion.
    hydrate_pattern = re.compile(
        rf"(([A-Z](?:[{sep_chars}]?[A-Za-z0-9]){{0,20}}))([{sep_chars}])((?:\d+)?H2O)"
    )
    # Replace all hydrate occurrences with placeholder for the dot
    def _protect_hydrate(m: re.Match) -> str:
        left_full, left_core, _dot, right = m.groups()
        # Remove any separator chars inside left_full to reconstruct contiguous formula
        left_clean = re.sub(rf"[{sep_chars}]", "", left_full)
        return f"{left_clean}{HY_PLACEHOLDER}{right}"

    s = hydrate_pattern.sub(_protect_hydrate, s)

    # Additional pattern: numeric group after dot but not necessarily H2O, e.g. 'CoSO4·7NH3'
    # Limit to capital letter + optional digits + capital letter start to reduce false positives.
    adduct_pattern = re.compile(
        rf"([A-Z][A-Za-z0-9]{{0,12}})([{sep_chars}])(\d+[A-Z][A-Za-z0-9]*)"
    )
    def _protect_adduct(m: re.Match) -> str:
        return f"{m.group(1)}{HY_PLACEHOLDER}{m.group(3)}"

    s = adduct_pattern.sub(_protect_adduct, s)

    # Remove separators that appear between single letters (the noisy case: m·a·t·e·r·i·a·l)
    # Strategy: if a separator is surrounded by letters (both sides alpha) drop it.
    s = re.sub(rf"(?<=[A-Za-z])[{sep_chars}](?=[A-Za-z])", "", s)

    # Strip any remaining separator characters unless they were part of a protected placeholder.
    s = re.sub(rf"[{sep_chars}]", "", s)

    # High-density safety: If we somehow still have many mid-dots (unlikely now), purge them.
    interpunct_ratio = sum(original.count(c) for c in sep_chars) / max(1, len(original))
    if interpunct_ratio > 0.05:
        # Already stripped; nothing extra to do – retained for logic symmetry.
        pass

    # Restore protected hydrate/adduct dots
    s = s.replace(HY_PLACEHOLDER, "·")

    # Post-pass: If a formula ending with digits is followed by a space then hydrate (e.g. "CuSO4 5H2O"),
    # reintroduce a mid-dot. This captures cases where noisy separators were stripped before protection.
    s = re.sub(r"\b([A-Z][A-Za-z0-9]{1,10})(?:\s+)(\d+H2O)\b", r"\1·\2", s)

    # Normalize stray control replacement (if any were in source)
    s = s.replace("\x03", "·")

    # Normalize hyphenation across line breaks (remove hyphen + newline splits)
    # Replace hyphen + newline (broken word) with a single empty join but keep a space if the next segment starts lowercase to avoid accidental concatenation of separate words.
    def _fix_hyphen(m: re.Match) -> str:
        prev = m.group(1)
        nxt = m.group(2)
        combined = prev + nxt
        # Join if prev or next is short (<=4) OR combined appears morphological (endswith common suffix)
        suffixes = ("tion", "sion", "ment", "ing", "able", "ible", "ally", "ness", "iate")
        if nxt.islower() and (len(prev) <= 4 or any(combined.endswith(s) for s in suffixes)):
            return combined
        if prev.islower() and nxt.islower():
            return combined
        return prev + " " + nxt

    s = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", _fix_hyphen, s)
    # Clean excessive whitespace around newlines
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
