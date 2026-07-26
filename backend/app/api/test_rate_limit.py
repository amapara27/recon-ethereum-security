# Minimal check for the analyzer sliding-window limiter. Run: python test_rate_limit.py
import asyncio
import types
from fastapi import HTTPException

import api

def test_limit_then_reset():
    api._analyze_calls.clear()
    clock = [1000.0]
    api.time = types.SimpleNamespace(monotonic=lambda: clock[0])  # freeze/advance clock
    run = asyncio.new_event_loop().run_until_complete

    for _ in range(api.ANALYZE_MAX):          # first N allowed
        run(api.rate_limit_analyzer())

    try:                                       # N+1 within window -> 429
        run(api.rate_limit_analyzer())
        assert False, "expected 429 after limit hit"
    except HTTPException as e:
        assert e.status_code == 429

    clock[0] += api.ANALYZE_WINDOW + 1         # advance past window -> frees up
    run(api.rate_limit_analyzer())
    print("ok")

if __name__ == "__main__":
    test_limit_then_reset()
