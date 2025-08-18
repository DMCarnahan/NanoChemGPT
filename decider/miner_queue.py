
"""
Stub for your text-miner queue. Replace with your Celery/RQ/Arq/whatever.
Provide a single enqueue function the /ask route can call.
"""
from __future__ import annotations
from typing import Optional, Dict

def enqueue_text_mining_job(query: str, *, user_id: Optional[str] = None, intent: Optional[str] = None, reason: Optional[str] = None, features: Optional[Dict]=None) -> str:
    """
    Enqueue a mining job and return a job_id.
    Implement this using your queue backend.
    """
    # TODO: Implement. For now, return a fake id.
    return "job_fake_000000"
