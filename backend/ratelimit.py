"""A small thread-safe sliding-window limiter for paid API calls (Gemini, ElevenLabs).

One limiter per provider, shared by every request: varying dates, alert ids or reports can't get
around it. Only actual external calls take a slot; cached answers don't.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Tuple


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_calls, self.per_seconds, self._clock = max_calls, per_seconds, clock
        self._calls: deque = deque()
        self._lock = threading.Lock()

    def acquire(self) -> Tuple[bool, float]:
        """Take a slot if one is free. Returns (allowed, seconds until a slot frees up)."""
        with self._lock:
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.per_seconds:
                self._calls.popleft()
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return True, 0.0
            return False, max(0.0, self.per_seconds - (now - self._calls[0]))

    def used(self) -> int:
        with self._lock:
            now = self._clock()
            return sum(1 for t in self._calls if now - t < self.per_seconds)
