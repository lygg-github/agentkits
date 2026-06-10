from __future__ import annotations

import inspect
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Generator


async def maybe_await(cb: Callable[..., Any] | None, *args: Any, **kwargs: Any) -> Any:
    if cb is None:
        return None
    res = cb(*args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


async def iter_to_async_stream(value: Any) -> AsyncGenerator[Any, None]:
    if isinstance(value, AsyncIterator):
        async for item in value:
            yield item
        return
    if isinstance(value, Generator):
        for item in value:
            yield item
        return
    yield value
