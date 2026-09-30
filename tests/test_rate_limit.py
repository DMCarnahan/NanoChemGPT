from app_utils.rate_limit import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_recovers_after_window():
    limiter = SlidingWindowRateLimiter()

    assert limiter.check("client", limit=2, window_seconds=60, now=0).allowed
    assert limiter.check("client", limit=2, window_seconds=60, now=1).allowed
    denied = limiter.check("client", limit=2, window_seconds=60, now=2)
    assert denied.allowed is False
    assert denied.retry_after == 59
    assert limiter.check("client", limit=2, window_seconds=60, now=61).allowed


def test_disabled_rate_limit_allows_requests():
    limiter = SlidingWindowRateLimiter()
    assert limiter.check("client", limit=0).allowed
