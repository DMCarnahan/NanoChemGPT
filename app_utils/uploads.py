"""Uploads and attachments helper utilities.

This module centralizes saving attachments and builtin files and ensures
directory creation and optional PDF text extraction. It imports canonical
path constants from `app_utils.constants` so callers (like `app.py`) do not
need to reference legacy module-level names directly.
"""
from pathlib import Path
import os
from typing import List

from werkzeug.datastructures import FileStorage

from app_utils.constants import ATTACH_DIR, BUILTIN_DIR


def save_attachment(file: FileStorage, max_pages: int | None = None) -> dict:
    """Save uploaded attachment file and return metadata.

    Returns a dict with keys: id, filename, kind, n_pages (if pdf), n_chars (if pdf)
    """
    import uuid
    from werkzeug.utils import secure_filename

    fname = secure_filename(file.filename or "file")
    aid = uuid.uuid4().hex[:12]
    dest = ATTACH_DIR / f"{aid}__{fname}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    file.save(dest)

    meta = {"id": aid, "filename": fname, "kind": "pdf" if fname.lower().endswith(".pdf") else "file"}

    if meta["kind"] == "pdf":
        # Defer heavy import until needed
        try:
            from app_utils.paths import ensure_pdf_text_extracted, extract_pdf_text as _extract_pdf_text

            # Let local helper extract text and return (text, n_pages)
            txt, n_pages = _extract_pdf_text(dest, max_pages=max_pages or int(os.environ.get("ATTACH_MAX_PAGES", "40")))
            (ATTACH_DIR / f"{aid}.txt").write_text(txt, encoding="utf-8")
            meta.update({"n_pages": n_pages, "n_chars": len(txt)})
        except Exception:
            # Best-effort: don't fail the entire request on extraction errors
            pass

    return meta


def save_builtin_files(files: List[FileStorage]) -> List[str]:
    """Save builtin files into BUILTIN_DIR and return list of saved filenames."""
    from werkzeug.utils import secure_filename

    saved = []
    for f in files:
        if not getattr(f, "filename", None):
            continue
        fname = secure_filename(f.filename)
        dest = BUILTIN_DIR / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved.append(fname)
    return saved


def read_attachment_text(aid: str, max_pages: int | None = None) -> str:
    """Return extracted text for an attachment id (best-effort).

    Looks for a pre-extracted .txt file first; if not found, tries to find
    an attached file with a matching prefix (aid__*) and extract text from
    PDF or read plain text files. Returns empty string on failure.
    """
    try:
        txt_path = ATTACH_DIR / f"{aid}.txt"
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="ignore")

        # Fallback: look for an uploaded file with prefix 'aid__'
        for pth in ATTACH_DIR.glob(f"{aid}__*"):
            if pth.suffix.lower() == ".pdf":
                try:
                    from app_utils.paths import extract_pdf_text as _extract_pdf_text

                    txt, _ = _extract_pdf_text(pth, max_pages=max_pages or int(os.environ.get("ATTACH_MAX_PAGES", "40")))
                    return txt
                except Exception:
                    continue
            else:
                try:
                    return pth.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
    except Exception:
        pass
    return ""


def latest_attachment_id() -> str | None:
    """Return the stem (id without extension) of the most recent .txt attachment.

    Returns None if none found.
    """
    try:
        latest_txt = max(ATTACH_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, default=None)
        if latest_txt:
            return latest_txt.stem
    except Exception:
        return None
    return None
