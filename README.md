# agentkits

`agentkits` is a lightweight async agent SDK for building tool-using agents on top of OpenAI-compatible Chat Completions, OpenAI Responses, and Claude.

The package keeps the runtime small and explicit: messages, models, tools, and agents are separate layers.

## Core Ideas

| Layer     | Responsibility                                               |
| --------- | ------------------------------------------------------------ |
| `message` | Normalizes system, user, assistant, and tool messages across providers. |
| `model`   | Wraps OpenAI-compatible Chat Completions, OpenAI Responses, and Claude behind one async interface. |
| `tool`    | Registers Python functions as JSON-schema tools and dispatches tool calls. |
| `agent`   | Provides ReAct, planning, handoff, agent-as-tool, and reflection-style runtimes. |

`ReActAgent` is the core execution loop. `PlanAgent` plans first, then runs an internal ReAct executor. `handoff(...)` lets a router agent transfer a conversation to another agent through a tool marker.

## Quick Start

```python
import asyncio

from agentkits import OpenAIChatModel, ReActAgent, Toolkit, ToolResponse

toolkit = Toolkit()


@toolkit.tool()
def add(a: int, b: int) -> ToolResponse:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return ToolResponse.from_value(str(a + b))


async def main() -> None:
    async with OpenAIChatModel.from_ali_env() as model:
        agent = ReActAgent(model=model, toolkit=toolkit)
        result = await agent.run("Use the tool to calculate 21 + 21.")
        print(result.text())


asyncio.run(main())
```

The larger demo in `examples/08_reproducible_demo.py` shows a full order-recovery workflow with a router agent, handoff, plan generation, tool calls, and final output.

## Agents

| Agent               | Purpose                                            |
| ------------------- | -------------------------------------------------- |
| `ReActAgent`        | Tool-calling loop.                                 |
| `ClassicReActAgent` | Text-based Thought / Action / Observation loop.    |
| `PlanAgent`         | Plan first, then execute.                          |
| `ReWOOAgent`        | Plan a tool DAG, execute observations, then solve. |
| `ReflexionAgent`    | Act, evaluate, reflect, and retry.                 |
| `SelfRefineAgent`   | Draft, review, and revise.                         |

## License

[Apache-2.0](./LICENSE)
