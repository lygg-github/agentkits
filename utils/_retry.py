from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .._logging import logger

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: float = 0.25
    _custom: dict = field(default_factory=dict, repr=False)


async def retry_async(
    op: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[BaseException], bool],
    op_name: str = "operation",
) -> T:
    def _before_sleep(retry_state):
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        sleep = getattr(retry_state.next_action, "sleep", None)
        logger.warning(
            "%s failed (attempt %d/%d): %s; retrying in %.2fs",
            op_name,
            retry_state.attempt_number,
            policy.max_retries + 1,
            exc,
            sleep if sleep is not None else 0.0,
        )

    retryer = AsyncRetrying(
        stop=stop_after_attempt(policy.max_retries + 1),
        wait=wait_random_exponential(
            multiplier=policy.initial_delay,
            max=policy.max_delay,
        ),
        retry=retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=_before_sleep,
    )

    try:
        async for attempt in retryer:
            with attempt:
                return await op()
    except RetryError as e:
        last = e.last_attempt.exception() if e.last_attempt else e
        raise last from e
    raise RuntimeError("unreachable")
