"""Simple in-memory job tracking used by the upload endpoints for quick status checks."""

from __future__ import annotations

from typing import Dict

JOBS: Dict[str, Dict] = {}


def set_job(jid: str, **kw):
    JOBS.setdefault(jid, {}).update(kw)


def get_job(jid: str):
    return JOBS.get(jid)


def mark_done(jid: str):
    set_job(jid, status="done", progress=100)
