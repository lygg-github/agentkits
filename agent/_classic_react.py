from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple

from .._logging import logger
from ..exceptions import MaxIterationsReached
from ..message import ChatMessageBase, ChatUsage, ToolCall
from ..model import ChatModelBase
from ..tool import Toolkit, ToolResponse
from ..utils._pattern import PatternSchema
from ._base import AgentBase, AgentResult


_DEFAULT_SYSTEM = """\
Solve the user's task by interleaving Thought, Action, and Observation
steps. Each step has the following format:

Thought N: <short reasoning about what to do next>
Action  N: <tool_name>[<argument>]

After you emit Action N, STOP. The system will execute the tool and
reply with:

Observation N: <tool result>

Then continue with Thought N+1 and the next Action.

Tools you may call:
{tool_lines}
- finish[<final answer>]: produce the final answer and end the task.

Rules:
- Emit exactly one Thought and one Action per step.
- Numbering starts at 1 and increments by 1 each step.
- <argument> is either a plain string (for a single-argument tool) or a
  JSON object on one line (for a multi-argument tool, e.g.
  ``search[{{"query": "x", "topk": 3}}]``).
- Never invent an Observation; the system fills it in.
- When you are confident, call ``finish[<answer>]`` as the final Action."""


@dataclass
class ClassicReActResult(AgentResult):
    scratchpad: str = ""
    final_answer: str = ""
    steps: List["ReActStep"] = field(default_factory=list)


@dataclass
class ReActStep:
    index: int
    thought: str
    action_name: str
    action_arg: str
    observation: str


_STEP_SCHEMA = PatternSchema.build(
    r"""
    Thought\s*(?P<ti>\d+)?\s*:\s*(?P<thought>.*?)
    (?=\n\s*Action\s*\d*\s*:)
    \n\s*Action\s*(?P<ai>\d+)?\s*:\s*(?P<name>[A-Za-z_][\w-]*)\s*\[(?P<arg>.*?)\]
    """,
    verbose=True,
    dotall=True,
)

_OBSERVATION_STOP_RE = re.compile(r"\n\s*Observation\s*\d*\s*:", re.IGNORECASE)

_FINISH_NAMES = {"finish", "final_answer", "answer"}


class ClassicReActAgent(AgentBase):
    def __init__(
        self,
        *,
        name: str = "classic_react",
        description: str = "",
        model: ChatModelBase,
        toolkit: Toolkit | None = None,
        extra_instructions: str | None = None,
        max_steps: int = 8,
    ) -> None:
        self.name = name
        self.description = description or f"A classic ReAct agent named {name}."
        self.model = model
        self.toolkit = toolkit or Toolkit()
        self.extra_instructions = extra_instructions
        self.max_steps = max_steps

    async def run(
        self,
        user_input: str | ChatMessageBase | list[ChatMessageBase],
        *,
        output_type: type | None = None,
        **_: Any,
    ) -> ClassicReActResult:
        question = _extract_question(user_input)
        system_text = self._build_system_prompt()

        history: list[ChatMessageBase] = [
            ChatMessageBase.system(system_text),
            ChatMessageBase.user(f"Question: {question}"),
        ]
        scratchpad = ""
        steps: list[ReActStep] = []
        tool_calls_made = 0
        usage_total: ChatUsage | None = None

        from ..model._response import ChatResponse

        prev_stream = self.model.stream
        self.model.stream = False
        try:
            for idx in range(1, self.max_steps + 1):
                response = await self.model.chat(history)
                if isinstance(response, ChatResponse):
                    raw = response.message.text
                    if response.usage is not None:
                        usage_total = (
                            response.usage
                            if usage_total is None
                            else usage_total + response.usage
                        )
                else:
                    raw = ""

                truncated = _truncate_at_observation(raw)
                if not truncated.strip():
                    logger.warning(
                        "classic_react[%s] empty step %d; nudging",
                        self.name, idx,
                    )
                    history.append(ChatMessageBase.assistant(raw or ""))
                    history.append(
                        ChatMessageBase.user(
                            "Empty response. Reply strictly in the required "
                            "format: one Thought line followed by one Action "
                            "line. If you already have the answer, call "
                            "finish[<answer>].",
                        ),
                    )
                    continue

                parsed = _parse_step(truncated)
                if parsed is None:
                    logger.warning(
                        "classic_react[%s] unparsable step %d: %r",
                        self.name, idx, truncated[:200],
                    )
                    nudge = (
                        "Your last step was not a valid Thought/Action pair. "
                        "Reply strictly in the required format."
                    )
                    history.append(ChatMessageBase.assistant(raw))
                    history.append(ChatMessageBase.user(nudge))
                    continue

                thought, action_name, action_arg = parsed

                if action_name.lower() in _FINISH_NAMES:
                    steps.append(
                        ReActStep(
                            index=idx,
                            thought=thought,
                            action_name=action_name,
                            action_arg=action_arg,
                            observation="",
                        ),
                    )
                    scratchpad += _format_step(
                        idx, thought, action_name, action_arg, observation=None,
                    )
                    final_msg = ChatMessageBase.assistant(
                        f"Thought {idx}: {thought}\n"
                        f"Action {idx}: {action_name}[{action_arg}]",
                    )
                    history.append(final_msg)

                    parsed_obj = None
                    if output_type is not None:
                        from ._base import finalize_structured_output

                        parsed_obj, extra = await finalize_structured_output(
                            self.model, history, output_type,
                        )
                        if extra is not None:
                            usage_total = (
                                extra if usage_total is None else usage_total + extra
                            )

                    result = ClassicReActResult(
                        messages=history,
                        final_message=final_msg,
                        iterations=idx,
                        tool_calls=tool_calls_made,
                        usage=usage_total,
                        parsed=parsed_obj,
                        scratchpad=scratchpad,
                        final_answer=action_arg,
                        steps=steps,
                    )
                    return result

                observation = await self._invoke_tool(action_name, action_arg)
                tool_calls_made += 1

                steps.append(
                    ReActStep(
                        index=idx,
                        thought=thought,
                        action_name=action_name,
                        action_arg=action_arg,
                        observation=observation,
                    ),
                )
                scratchpad += _format_step(
                    idx, thought, action_name, action_arg, observation=observation,
                )

                history.append(
                    ChatMessageBase.assistant(
                        f"Thought {idx}: {thought}\n"
                        f"Action {idx}: {action_name}[{action_arg}]",
                    ),
                )
                history.append(
                    ChatMessageBase.user(f"Observation {idx}: {observation}"),
                )
        finally:
            self.model.stream = prev_stream

        last = history[-1] if history else None
        raise MaxIterationsReached(
            self.max_steps,
            last_state=ClassicReActResult(
                messages=history,
                final_message=last,
                iterations=self.max_steps,
                tool_calls=tool_calls_made,
                usage=usage_total,
                scratchpad=scratchpad,
                steps=steps,
            ),
        )

    def _build_system_prompt(self) -> str:
        lines: list[str] = []
        for schema in self.toolkit.get_json_schemas():
            fn = schema.get("function", {})
            lines.append(
                f"- {fn.get('name')}[<arg>]: "
                f"{(fn.get('description') or '').splitlines()[0]}".rstrip(),
            )
        tool_block = "\n".join(lines) if lines else "(no tools available)"
        prompt = _DEFAULT_SYSTEM.format(tool_lines=tool_block)
        if self.extra_instructions:
            prompt = f"{prompt}\n\n{self.extra_instructions}"
        return prompt

    async def _invoke_tool(
        self,
        name: str,
        arg: str,
    ) -> str:
        entry = self.toolkit.tools.get(name)
        if entry is None:
            return f"(error) unknown tool {name!r}"

        kwargs = _bind_action_arg(arg, entry.json_schema)
        chunks: list[ToolResponse] = []
        async for chunk in self.toolkit.call_tool(
            ToolCall(id=f"classic_react_{name}", name=name, input=kwargs),
        ):
            chunks.append(chunk)
        if not chunks:
            return ""
        return "".join(c.text for c in chunks).strip()


def _bind_action_arg(arg: str, json_schema: dict) -> dict[str, Any]:
    arg = arg.strip()
    if arg.startswith("{"):
        try:
            parsed = json.loads(arg)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    params = json_schema.get("function", {}).get("parameters", {})
    props = params.get("properties", {}) or {}
    required = params.get("required", []) or []
    if not props:
        return {}
    first = required[0] if required else next(iter(props))
    return {first: arg}


def _truncate_at_observation(text: str) -> str:
    m = _OBSERVATION_STOP_RE.search(text)
    if m is None:
        return text
    return text[: m.start()]


def _parse_step(text: str) -> Tuple[str, str, str] | None:
    hit = _STEP_SCHEMA.match_one(text)
    if hit is None:
        return None
    return hit.get("thought", ""), hit.get("name", ""), hit.get("arg", "")


def _format_step(
    idx: int,
    thought: str,
    name: str,
    arg: str,
    observation: str | None,
) -> str:
    buf = f"Thought {idx}: {thought}\nAction {idx}: {name}[{arg}]\n"
    if observation is not None:
        buf += f"Observation {idx}: {observation}\n"
    return buf


def _extract_question(
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
