from __future__ import annotations

from typing import Any


class AgentkitsError(Exception):
    pass


class ModelError(AgentkitsError):
    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.__cause__ = cause


class ModelBehaviorError(ModelError):
    pass


class ToolError(AgentkitsError):
    pass


class ToolNotFoundError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Tool '{name}' is not registered.")
        self.name = name


class ToolExecutionError(ToolError):
    def __init__(self, name: str, cause: BaseException) -> None:
        super().__init__(f"Tool '{name}' raised: {cause}")
        self.name = name
        self.__cause__ = cause


class AgentError(AgentkitsError):
    pass


class MaxIterationsReached(AgentError):
    def __init__(self, max_iterations: int, *, last_state: Any = None) -> None:
        super().__init__(
            f"Agent did not converge within {max_iterations} iterations.",
        )
        self.max_iterations = max_iterations
        self.last_state = last_state
