from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from ..message import ChatMessageBase
from ..model import ChatModelBase
from ..tool import Toolkit
from ._base import AgentBase, AgentResult
from ._plan import _extract_user_text
from ._react import ReActAgent, ReActResult


_DEFAULT_FEEDBACK_PROMPT = """\
You are a strict reviewer. Given the task and a draft answer, list the
concrete issues that must be fixed. Be specific. If the answer is
already correct, reply exactly with: "No further improvements needed."

Task:
{task}

Draft answer:
{answer}"""


_DEFAULT_REFINE_PROMPT = """\
You previously answered the task. Revise your answer using the reviewer
feedback. Produce only the revised final answer.

Task:
{task}

Previous answer:
{answer}

Reviewer feedback:
{feedback}"""


@dataclass
class SelfRefineResult(AgentResult):
    feedbacks: List[str] = field(default_factory=list)
    drafts: List[str] = field(default_factory=list)
    rounds: int = 0


class SelfRefineAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "self_refine",
        description: str = "",
        model: ChatModelBase,
        toolkit: Toolkit | None = None,
        system_prompt: str | None = None,
        max_rounds: int = 3,
        max_iterations: int = 6,
        stop_marker: str = "no further",
    ) -> None:
        self.name = name
        self.description = description or f"A Self-Refine agent named {name}."
        self.model = model
        self.toolkit = toolkit or Toolkit()
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.max_iterations = max_iterations
        self.stop_marker = stop_marker.lower()

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        *,
        max_rounds: int | None = None,
        output_type: type | None = None,
        **_: Any,
    ) -> SelfRefineResult:
        task = _extract_user_text(user_input)
        rounds_limit = max_rounds or self.max_rounds

        draft_result = await self._produce(task)
        drafts: list[str] = [draft_result.text()]
        feedbacks: list[str] = []
        final_answer = drafts[0]
        last_result: ReActResult = draft_result

        for rnd in range(1, rounds_limit + 1):
            feedback = await self._feedback(task, final_answer)
            feedbacks.append(feedback)
            if self.stop_marker in feedback.lower():
                break
            refined = await self._refine(task, final_answer, feedback)
            final_answer = refined.text()
            drafts.append(final_answer)
            last_result = refined

        parsed_obj = None
        total_usage = last_result.usage
        if output_type is not None:
            from ._base import finalize_structured_output

            parsed_obj, extra = await finalize_structured_output(
                self.model, last_result.messages, output_type,
            )
            if extra is not None:
                total_usage = extra if total_usage is None else total_usage + extra

        result = SelfRefineResult(
            messages=last_result.messages,
            final_message=last_result.final_message,
            iterations=last_result.iterations,
            tool_calls=last_result.tool_calls,
            usage=total_usage,
            parsed=parsed_obj,
            drafts=drafts,
            feedbacks=feedbacks,
            rounds=len(drafts) - 1,
        )
        return result

    async def _produce(self, task: str) -> ReActResult:
        actor = ReActAgent(
            name=f"{self.name}.actor",
            model=self.model,
            toolkit=self.toolkit,
            system_prompt=self.system_prompt or "Solve the user's task.",
            max_iterations=self.max_iterations,
        )
        return await actor.run(task)

    async def _refine(
        self,
        task: str,
        answer: str,
        feedback: str,
    ) -> ReActResult:
        actor = ReActAgent(
            name=f"{self.name}.refiner",
            model=self.model,
            toolkit=self.toolkit,
            max_iterations=self.max_iterations,
        )
        prompt = _DEFAULT_REFINE_PROMPT.format(
            task=task, answer=answer, feedback=feedback,
        )
        return await actor.run(prompt)

    async def _feedback(self, task: str, answer: str) -> str:
        prev = self.model.stream
        self.model.stream = False
        try:
            final = await self.model.chat_cb(
                [
                    ChatMessageBase.system(
                        _DEFAULT_FEEDBACK_PROMPT.format(task=task, answer=answer),
                    ),
                ],
                on_message=lambda m: None,
            )
        finally:
            self.model.stream = prev
        return final.text.strip()
