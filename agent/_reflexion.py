from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List

from ..message import ChatMessageBase
from ..model import ChatModelBase
from ..tool import Toolkit
from ._base import AgentBase, AgentResult
from ._plan import _extract_user_text
from ._react import ReActAgent, ReActResult


EvaluateFn = Callable[[str, str], bool | Awaitable[bool]]


_DEFAULT_REFLECTION_PROMPT = """\
You are a self-reflection module. The actor attempted the task below
and its output was judged incorrect. Write a short (<=3 sentences)
reflection that the actor can use on the next attempt to avoid the
same mistake. Do NOT restate the task; focus on what went wrong and
what to do differently.

Task:
{task}

Attempted answer:
{answer}"""


_DEFAULT_EVAL_PROMPT = """\
Judge whether the actor's final answer correctly solves the task.
Reply with a single token: "yes" or "no".

Task:
{task}

Answer:
{answer}"""


@dataclass
class ReflexionResult(AgentResult):
    reflections: List[str] = field(default_factory=list)
    trials: int = 0
    success: bool = False


class ReflexionAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "reflexion",
        description: str = "",
        model: ChatModelBase,
        toolkit: Toolkit | None = None,
        system_prompt: str | None = None,
        max_trials: int = 3,
        max_iterations: int = 8,
        evaluator: EvaluateFn | None = None,
        evaluator_model: ChatModelBase | None = None,
        reflection_model: ChatModelBase | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"A Reflexion agent named {name}."
        self.model = model
        self.toolkit = toolkit or Toolkit()
        self.system_prompt = system_prompt
        self.max_trials = max_trials
        self.max_iterations = max_iterations
        self.evaluator = evaluator
        self.evaluator_model = evaluator_model or model
        self.reflection_model = reflection_model or model

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        *,
        max_trials: int | None = None,
        output_type: type | None = None,
        **_: Any,
    ) -> ReflexionResult:
        task = _extract_user_text(user_input)
        limit = max_trials or self.max_trials
        reflections: list[str] = []
        last_result: ReActResult | None = None
        success = False

        for trial in range(1, limit + 1):
            actor = ReActAgent(
                name=f"{self.name}.actor#{trial}",
                model=self.model,
                toolkit=self.toolkit,
                system_prompt=self._actor_prompt(reflections),
                max_iterations=self.max_iterations,
            )
            last_result = await actor.run(task)

            if await self._evaluate(task, last_result.text()):
                success = True
                break

            if trial >= limit:
                break
            reflections.append(
                await self._reflect(task, last_result.text()),
            )

        assert last_result is not None

        parsed_obj = None
        total_usage = last_result.usage
        if success and output_type is not None:
            from ._base import finalize_structured_output

            parsed_obj, extra = await finalize_structured_output(
                self.model, last_result.messages, output_type,
            )
            if extra is not None:
                total_usage = extra if total_usage is None else total_usage + extra

        result = ReflexionResult(
            messages=last_result.messages,
            final_message=last_result.final_message,
            iterations=last_result.iterations,
            tool_calls=last_result.tool_calls,
            usage=total_usage,
            parsed=parsed_obj,
            reflections=reflections,
            trials=trial,
            success=success,
        )
        return result

    def _actor_prompt(self, reflections: list[str]) -> str:
        base = self.system_prompt or ""
        if not reflections:
            return base or "Solve the user's task step by step."
        lessons = "\n".join(f"- {r}" for r in reflections)
        extra = (
            "Previous attempts failed. Apply these reflections before "
            f"answering this time:\n{lessons}"
        )
        return f"{base}\n\n{extra}".strip()

    async def _evaluate(
        self,
        task: str,
        answer: str,
    ) -> bool:
        if self.evaluator is not None:
            out = self.evaluator(task, answer)
            if hasattr(out, "__await__"):
                return bool(await out)
            return bool(out)

        prev = self.evaluator_model.stream
        self.evaluator_model.stream = False
        try:
            final = await self.evaluator_model.chat_cb(
                [
                    ChatMessageBase.system(
                        _DEFAULT_EVAL_PROMPT.format(task=task, answer=answer),
                    ),
                ],
                on_message=lambda m: None,
            )
        finally:
            self.evaluator_model.stream = prev
        return final.text.strip().lower().startswith("y")

    async def _reflect(
        self,
        task: str,
        answer: str,
    ) -> str:
        prev = self.reflection_model.stream
        self.reflection_model.stream = False
        try:
            final = await self.reflection_model.chat_cb(
                [
                    ChatMessageBase.system(
                        _DEFAULT_REFLECTION_PROMPT.format(task=task, answer=answer),
                    ),
                ],
                on_message=lambda m: None,
            )
        finally:
            self.reflection_model.stream = prev
        return final.text.strip() or "(no reflection produced)"
