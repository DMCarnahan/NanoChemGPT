"""Small dependency-free per-process rate limiter for paid API routes."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after: int


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window limiter.

    Railway may run more than one worker, so this is a defensive cost guard,
    not a distributed quota system. A Redis-backed limiter can replace it
    later without changing route behavior.
    """

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int = 60,
        now: float | None = None,
    ) -> RateLimitDecision:
        if limit <= 0:
            return RateLimitDecision(True, 0)
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - max(1, window_seconds)
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(events[0] + window_seconds - timestamp) + 1)
                return RateLimitDecision(False, retry_after)
            events.append(timestamp)
            return RateLimitDecision(True, 0)
