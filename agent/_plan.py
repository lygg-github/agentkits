from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from ..message import ChatMessageBase
from ..model import ChatModelBase
from ..model._base import OnChunkCb, OnMessageCb
from ..tool import Toolkit
from ._base import AgentBase, AgentResult
from ._react import ReActAgent, ReActResult


_DEFAULT_PLANNER_PROMPT = """\
You are the planner. Decompose the user's request into a minimal numbered
list of short imperative steps (no more than {max_steps}). Use only tools
that the executor has access to - the tool names are listed below. If a
tool is not needed, you may write a reasoning step instead.

Available tools:
{tools}

Respond with ONLY a numbered list, one step per line, like:

1. <step>
2. <step>

Keep each step tied to one tool category or one final reasoning step.
Repeated calls to the same tool for similar items may share one step. Do
not combine unrelated tool categories in the same numbered step.
Do not add any prose before or after the list."""


_DEFAULT_EXECUTOR_PROMPT = """\
You are the executor for a plan. Work through the plan step by step,
calling tools as needed. When the final step is complete, produce the
final answer.

The plan is:
{plan}

The user's original request was:
{user_input}"""


@dataclass
class PlanResult(AgentResult):
    plan: List[str] = field(default_factory=list)


class PlanAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "planner",
        description: str = "",
        model: ChatModelBase | None = None,
        planner_model: ChatModelBase | None = None,
        executor_model: ChatModelBase | None = None,
        toolkit: Toolkit | None = None,
        system_prompt: str | None = None,
        max_steps: int = 6,
        max_iterations: int = 12,
        stream: bool | None = None,
    ) -> None:
        if planner_model is None and executor_model is None and model is None:
            raise ValueError(
                "PlanAgent requires at least one of model / planner_model / executor_model.",
            )
        shared_model = model or planner_model or executor_model
        self.name = name
        self.description = description or f"A plan-and-execute agent named {name}."
        self.planner_model = planner_model or shared_model
        self.executor_model = executor_model or shared_model
        self.toolkit = toolkit or Toolkit()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.default_stream = stream

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        *,
        stream: bool | None = None,
        on_chunk: OnChunkCb | None = None,
        on_message: OnMessageCb | None = None,
        max_iterations: int | None = None,
        output_type: type | None = None,
        **_: Any,
    ) -> PlanResult:
        user_text = _extract_user_text(user_input)

        plan = await self._plan(user_text)
        executor = ReActAgent(
            name=f"{self.name}.executor",
            model=self.executor_model,
            toolkit=self.toolkit,
            system_prompt=self._executor_prompt(plan, user_text),
            max_iterations=max_iterations or self.max_iterations,
            stream=stream if stream is not None else self.default_stream,
        )
        react_result: ReActResult = await executor.run(
            user_text,
            on_chunk=on_chunk,
            on_message=on_message,
            output_type=output_type,
        )

        result = PlanResult(
            messages=react_result.messages,
            final_message=react_result.final_message,
            iterations=react_result.iterations + 1,
            tool_calls=react_result.tool_calls,
            usage=react_result.usage,
            parsed=react_result.parsed,
            plan=plan,
        )
        return result

    async def _plan(
        self,
        user_text: str,
    ) -> list[str]:
        tools_list = "\n".join(
            f"- {s['function']['name']}: {s['function'].get('description') or ''}".rstrip()
            for s in self.toolkit.get_json_schemas()
        ) or "(no tools available)"

        planner_prompt = _DEFAULT_PLANNER_PROMPT.format(
            max_steps=self.max_steps, tools=tools_list,
        )
        prev_stream = self.planner_model.stream
        self.planner_model.stream = False
        try:
            final = await self.planner_model.chat_cb(
                [
                    ChatMessageBase.system(planner_prompt),
                    ChatMessageBase.user(user_text),
                ],
                on_message=lambda m: None,
            )
        finally:
            self.planner_model.stream = prev_stream

        steps = _parse_numbered_list(final.text, self.max_steps)
        return steps or [user_text]

    def _executor_prompt(self, plan: list[str], user_text: str) -> str:
        plan_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        prompt = _DEFAULT_EXECUTOR_PROMPT.format(
            plan=plan_block, user_input=user_text,
        )
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
        return prompt


_NUMBERED_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")


def _parse_numbered_list(text: str, max_steps: int) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            step = m.group(1).strip()
            if step:
                steps.append(step)
            if len(steps) >= max_steps:
                break
    return steps


def _extract_user_text(
    user_input: str | ChatMessageBase | list[ChatMessageBase],
) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, ChatMessageBase):
        return user_input.text
    for m in user_input:
        if m.role == "user":
            return m.text
    return ""
