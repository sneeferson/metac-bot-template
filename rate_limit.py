"""
Request throttling for models that have a requests-per-minute cap.

Why this exists: OpenRouter limits new accounts to 20 requests per minute on
some models (seen on anthropic/claude-sonnet-5.5, 10 Nov 2026). The bot makes
5 forecasts per question and works on several questions at once, so without
throttling it bursts past that cap and questions fail with HTTP 429.

ThrottledLlm behaves exactly like GeneralLlm, except that every network call
(including automatic retries) first waits for a free slot in a sliding
60-second window.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)


class SlidingWindowThrottle:
    """Allows at most `max_calls` call starts in any `period_seconds` window.

    Uses a plain threading lock plus asyncio.sleep, rather than asyncio locks,
    so one instance keeps working across the several asyncio.run() calls that
    main.py makes.
    """

    def __init__(self, max_calls: int, period_seconds: float = 60.0) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._call_starts: deque[float] = deque()
        self._lock = threading.Lock()

    def _seconds_until_slot(self) -> float:
        """Reserves a slot and returns 0, or returns how long to wait."""
        with self._lock:
            now = time.monotonic()
            while (
                self._call_starts
                and now - self._call_starts[0] >= self.period_seconds
            ):
                self._call_starts.popleft()
            if len(self._call_starts) < self.max_calls:
                self._call_starts.append(now)
                return 0.0
            return self.period_seconds - (now - self._call_starts[0])

    async def wait_for_slot(self) -> None:
        while True:
            wait = self._seconds_until_slot()
            if wait <= 0:
                return
            # Small random extra so waiting calls don't all wake together.
            await asyncio.sleep(wait + random.uniform(0.05, 0.5))


class ThrottledLlm(GeneralLlm):
    """GeneralLlm that never exceeds `requests_per_minute` for its model.

    All ThrottledLlm objects that use the same model name share one throttle.
    """

    _throttles: dict[str, SlidingWindowThrottle] = {}
    _throttles_lock = threading.Lock()

    def __init__(self, *args, requests_per_minute: int = 15, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        with ThrottledLlm._throttles_lock:
            if self.model not in ThrottledLlm._throttles:
                ThrottledLlm._throttles[self.model] = SlidingWindowThrottle(
                    max_calls=requests_per_minute
                )
        self._throttle = ThrottledLlm._throttles[self.model]

    async def _call_litellm(self, prompt):  # type: ignore[override]
        await self._throttle.wait_for_slot()
        return await super()._call_litellm(prompt)
