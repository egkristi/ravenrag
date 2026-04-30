"""
Timing: Performance measurement utilities for RavenRAG.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Global timing registry — stores cumulative stats
_timings: dict[str, dict[str, float | int]] = {}


def timed(name: str | None = None) -> Callable:
    """Decorator that logs execution time and records stats.

    Args:
        name: Operation name for logging. Defaults to function name.

    Example::

        @timed("embed")
        def encode(texts):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        op_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                logger.debug("%s took %.3fs", op_name, elapsed)
                if op_name not in _timings:
                    _timings[op_name] = {"total_seconds": 0.0, "calls": 0}
                _timings[op_name]["total_seconds"] += elapsed
                _timings[op_name]["calls"] += 1

        return wrapper

    return decorator


def get_timings() -> dict[str, dict[str, float | int]]:
    """Return collected timing statistics.

    Returns:
        Dict mapping operation name to {total_seconds, calls}.
    """
    return dict(_timings)


def reset_timings() -> None:
    """Clear all collected timing statistics."""
    _timings.clear()
