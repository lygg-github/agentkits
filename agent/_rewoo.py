from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List

from ..message import ChatMessageBase, ChatUsage, ToolCall
from ..model import ChatModelBase
from ..tool import Toolkit, ToolResponse
from ..utils._json import safe_json_loads
from ..utils._pattern import PatternSchema
from ._base import AgentBase, AgentResult
from ._plan import _extract_user_text


@dataclass
class _PlanStep:
    key: str
    thought: str
    tool: str
    args_raw: str


@dataclass
class ReWOOResult(AgentResult):
    plan: List[_PlanStep] = field(default_factory=list)
    observations: dict[str, str] = field(default_factory=dict)


_DEFAULT_PLANNER_PROMPT = """\
You are the ReWOO planner. Break the user's task into a short DAG of
tool calls (at most {max_steps} steps) before any execution happens.

Strict output format:

Plan: <one-line thought>
#E1 = tool_name[json_args]
Plan: <one-line thought>
#E2 = tool_name[json_args]
...

Rules:
- Tool names must come from the list below.
- Arguments must be a single JSON object per step.
- To reference a previous observation, put "#E<k>" literally inside the
  JSON string values, e.g. {{"query": "#E1"}}. Substitution happens at
  runtime.
- Do NOT produce anything else (no markdown, no prose, no trailing text).

Available tools:
{tools}

Task:
{task}"""


_DEFAULT_SOLVER_PROMPT = """\
You are the ReWOO solver. Using the plan and the observations collected
from each step, produce the final answer.

Task:
{task}

Plan:
{plan}

Observations:
{observations}

Reply with the final answer only. No explanation."""


_STEP_SCHEMA = PatternSchema.build(
    r"(?:Plan\s*:\s*(?P<thought>.+?)\s*\n\s*)?#(?P<key>E\d+)\s*=\s*(?P<tool>[A-Za-z_][\w-]*)\s*\[(?P<args>.*?)\]\s*$",
    multiline=True,
    dotall=True,
)
_PLACEHOLDER_RE = re.compile(r"#E\d+")


class ReWOOAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "rewoo",
        description: str = "",
        model: ChatModelBase | None = None,
        planner_model: ChatModelBase | None = None,
        solver_model: ChatModelBase | None = None,
        toolkit: Toolkit | None = None,
        max_steps: int = 6,
    ) -> None:
        if planner_model is None and solver_model is None and model is None:
            raise ValueError(
                "ReWOOAgent requires at least one of model / planner_model / solver_model.",
            )
        shared_model = model or planner_model or solver_model
        self.name = name
        self.description = description or f"A ReWOO agent named {name}."
        self.planner_model = planner_model or shared_model
        self.solver_model = solver_model or shared_model
        self.toolkit = toolkit or Toolkit()
        self.max_steps = max_steps

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        *,
        output_type: type | None = None,
        **_: Any,
    ) -> ReWOOResult:
        task = _extract_user_text(user_input)

        plan, usage_plan = await self._plan(task)
        observations, tool_calls = await self._work(plan)
        final_answer, usage_solve = await self._solve(task, plan, observations)

        usage: ChatUsage | None = None
        for u in (usage_plan, usage_solve):
            if u is None:
                continue
            usage = u if usage is None else usage + u

        final_message = ChatMessageBase.assistant(final_answer)
        messages = [ChatMessageBase.user(task), final_message]

        parsed_obj = None
        if output_type is not None:
            from ._base import finalize_structured_output

            parsed_obj, extra = await finalize_structured_output(
                self.solver_model, messages, output_type,
            )
            if extra is not None:
                usage = extra if usage is None else usage + extra

        result = ReWOOResult(
            messages=messages,
            final_message=final_message,
            iterations=2,
            tool_calls=tool_calls,
            usage=usage,
            parsed=parsed_obj,
            plan=plan,
            observations=observations,
        )
        return result

    async def _plan(
        self,
        task: str,
    ) -> tuple[list[_PlanStep], ChatUsage | None]:
        tools_list = "\n".join(
            f"- {s['function']['name']}: {s['function'].get('description') or ''}".rstrip()
            for s in self.toolkit.get_json_schemas()
        ) or "(no tools available)"

        prompt = _DEFAULT_PLANNER_PROMPT.format(
            max_steps=self.max_steps, tools=tools_list, task=task,
        )
        prev = self.planner_model.stream
        self.planner_model.stream = False
        try:
            response = await self.planner_model.chat(
                [ChatMessageBase.system(prompt)],
            )
        finally:
            self.planner_model.stream = prev

        from ..model._response import ChatResponse
        assert isinstance(response, ChatResponse)
        steps = _parse_plan(response.message.text, self.max_steps)
        return steps, response.usage

    async def _work(self, plan: list[_PlanStep]) -> tuple[dict[str, str], int]:
        observations: dict[str, str] = {}
        executed = 0
        for step in plan:
            args_str = _substitute(step.args_raw, observations)
            args = safe_json_loads(args_str, default={})
            if not isinstance(args, dict):
                observations[step.key] = f"(invalid args: {args_str!r})"
                continue
            chunks: list[ToolResponse] = []
            async for chunk in self.toolkit.call_tool(
                ToolCall(id=step.key, name=step.tool, input=args),
            ):
                chunks.append(chunk)
            text = "".join(c.text for c in chunks)
            observations[step.key] = text
            executed += 1
        return observations, executed

    async def _solve(
        self,
        task: str,
        plan: list[_PlanStep],
        observations: dict[str, str],
    ) -> tuple[str, ChatUsage | None]:
        plan_block = "\n".join(
            f"Plan: {s.thought}\n#{s.key} = {s.tool}[{s.args_raw}]"
            if s.thought
            else f"#{s.key} = {s.tool}[{s.args_raw}]"
            for s in plan
        )
        obs_block = "\n".join(
            f"#{k} = {v}" for k, v in observations.items()
        ) or "(none)"

        prompt = _DEFAULT_SOLVER_PROMPT.format(
            task=task, plan=plan_block, observations=obs_block,
        )
        prev = self.solver_model.stream
        self.solver_model.stream = False
        try:
            response = await self.solver_model.chat(
                [ChatMessageBase.system(prompt)],
            )
        finally:
            self.solver_model.stream = prev

        from ..model._response import ChatResponse
        assert isinstance(response, ChatResponse)
        return response.message.text.strip(), response.usage


def _parse_plan(text: str, max_steps: int) -> list[_PlanStep]:
    steps: list[_PlanStep] = []
    for hit in _STEP_SCHEMA.match_all(text):
        steps.append(
            _PlanStep(
                key=hit.get("key", ""),
                thought=hit.get("thought", ""),
                tool=hit.get("tool", ""),
                args_raw=hit.get("args", ""),
            ),
        )
        if len(steps) >= max_steps:
            break
    return steps


def _substitute(text: str, observations: dict[str, str]) -> str:
    if not text:
        return text

    def _sub(match: re.Match) -> str:
        key = match.group(0)[1:]
        return _escape_for_json(observations.get(key, match.group(0)))

    return _PLACEHOLDER_RE.sub(_sub, text)


def _escape_for_json(value: str) -> str:
    return json.dumps(value)[1:-1]
