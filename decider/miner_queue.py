"""
Text-miner enqueue shim.

Usage (env):
  MINER_MODE=disabled   -> no-op; returns a fake id (default)
  MINER_MODE=log        -> just logs payload locally, returns log-based id
  MINER_MODE=redis      -> enqueue to RQ (Redis Queue)
  
  REDIS_URL=x
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import os, time, json, logging

logging.basicConfig(level=logging.INFO)
MINER_MODE = os.getenv("MINER_MODE", "disabled").strip().lower()

def _now_ms() -> int:
    return int(time.time() * 1000)

def _mk_payload(query: str, user_id: Optional[str], intent: Optional[str],
                reason: Optional[str], features: Optional[Dict]) -> Dict[str, Any]:
    return {
        "query": query,
        "user_id": user_id,
        "intent": intent,
        "reason": reason,
        "features": features or {},
        "ts_ms": _now_ms(),
    }

def enqueue_text_mining_job(
    query: str, *,
    user_id: Optional[str] = None,
    intent: Optional[str] = None,
    reason: Optional[str] = None,
    features: Optional[Dict] = None
) -> str:
    """
    Enqueue a mining job and return a job_id, or a sentinel if disabled.
    This function is safe to call unconditionally from /ask.
    """
    payload = _mk_payload(query, user_id, intent, reason, features)

    if MINER_MODE in ("", "disabled", "off", "0", "false", "no"):
        # No-op path — behaves as “successfully enqueued” but nothing runs
        logging.info("[miner] disabled; skipping enqueue. query=%r", query[:120])
        return "job_disabled_" + str(payload["ts_ms"])

    if MINER_MODE == "log":
        # Minimal filesystem “queue”: append to a log file for later manual runs
        path = os.getenv("MINER_LOG_PATH", "/tmp/miner_jobs.log")
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            job_id = f"log_{payload['ts_ms']}"
            logging.info("[miner] logged job -> %s (%s)", job_id, path)
            return job_id
        except Exception as e:
            logging.exception("[miner] log mode failed: %s", e)
            return "job_log_error_" + str(payload["ts_ms"])

    if MINER_MODE == "redis":
        # RQ (Redis Queue)
        try:
            from redis import Redis
            from rq import Queue
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            conn = Redis.from_url(redis_url)
            q = Queue("miner", connection=conn, default_timeout=600)
            # Expect a worker process importing workers.miner:run_miner
            job = q.enqueue("workers.miner.run_miner", payload, job_timeout=600)
            logging.info("[miner] RQ enqueued -> %s", job.id)
            return str(job.id)
        except Exception as e:
            logging.exception("[miner] redis/RQ enqueue failed: %s", e)
            return "job_redis_error_" + str(payload["ts_ms"])

    logging.warning("[miner] unknown MINER_MODE=%r; acting disabled", MINER_MODE)
    return "job_unknown_mode_" + str(payload["ts_ms"])
