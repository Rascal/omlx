# SPDX-License-Identifier: Apache-2.0
"""Rolling-window decode rate for the dashboard's per-request rows."""

from collections import deque
from collections.abc import Iterable

WINDOW_SECONDS = 3.0
# Below this span two polls are too close for a meaningful rate.
MIN_SPAN_SECONDS = 1.0
MAX_SAMPLES = 64


class RecentRateTracker:
    """Tokens per second over the last few seconds, sampled at poll time.

    The lifetime average the dashboard used to show hides how a row is doing
    right now: it keeps the history of a request that ran alone before its
    neighbours arrived. This tracks (time, generated tokens) per request and
    reports the rate between the oldest sample still inside the window and
    the newest one.
    """

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        min_span_seconds: float = MIN_SPAN_SECONDS,
    ) -> None:
        self._window = window_seconds
        self._min_span = min_span_seconds
        self._samples: dict[str, deque[tuple[float, int]]] = {}

    def observe(self, request_id: str, now: float, generated_tokens: int) -> float | None:
        """Record a sample; return the windowed rate, or None until the window is warm."""
        history = self._samples.get(request_id)
        if history is None:
            history = self._samples[request_id] = deque(maxlen=MAX_SAMPLES)
        history.append((now, generated_tokens))
        cutoff = now - self._window
        while len(history) > 1 and history[0][0] < cutoff:
            history.popleft()
        oldest_at, oldest_tokens = history[0]
        span = now - oldest_at
        if span < self._min_span:
            return None
        return max(0.0, (generated_tokens - oldest_tokens) / span)

    def prune(self, active_ids: Iterable[str]) -> None:
        """Drop history for requests that are no longer generating."""
        keep = set(active_ids)
        for request_id in list(self._samples):
            if request_id not in keep:
                del self._samples[request_id]
