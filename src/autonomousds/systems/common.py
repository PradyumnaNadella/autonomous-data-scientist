"""Helpers shared by all agent systems (single_agent, pipeline, debate)."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from autonomousds.benchmark.tasks import ROOT, Task
from autonomousds.llm.models import LOG_PATH

RESULTS_DIR = ROOT / "benchmark" / "results"


def message_text(msg) -> str:
    """Return the plain text of a chat message.

    Gemini can return content as a list of blocks instead of a string;
    this joins the text parts and ignores the rest.

    Args:
        msg: A LangChain message (AIMessage, ToolMessage, ...).

    Returns:
        The message text, or an empty string if it has none.
    """
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))


def print_step(msg, agent: str = "agent") -> None:
    """Print a one-line summary of an agent step to the terminal.

    Args:
        msg: The newest message in the agent's conversation.
        agent: Name shown in front of the agent's text replies.
    """
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for call in msg.tool_calls:
            first = str(next(iter(call["args"].values()), "")).strip().splitlines()
            print(f"  -> {call['name']}: {first[0][:80] if first else ''}")
    elif isinstance(msg, ToolMessage):
        lines = message_text(msg).splitlines() or [""]
        print(f"     {lines[0][:80]}")
    elif isinstance(msg, AIMessage):
        print(f"  {agent}: {message_text(msg)[:200]}")


def run_usage(run_id: str) -> dict:
    """Total the LLM usage of one run from logs/llm_calls.jsonl.

    Args:
        run_id: The run whose calls should be counted.

    Returns:
        Dict with llm_calls, input_tokens, output_tokens and est_cost_usd.
    """
    lines = LOG_PATH.read_text().splitlines() if LOG_PATH.exists() else []
    calls = [c for c in map(json.loads, lines) if c["run_id"] == run_id]
    return {"llm_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "est_cost_usd": round(sum(c["est_cost_usd"] for c in calls), 4)}


def append_result(task: Task, system: str, record: dict) -> Path:
    """Append one run's result to benchmark/results/<task>/<system>.json.

    Agent runs vary, so each run is added to a list instead of overwriting.

    Args:
        task: The task that was solved.
        system: System name, e.g. "single_agent".
        record: The run's result record.

    Returns:
        Path of the results file.
    """
    out_dir = RESULTS_DIR / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{system}.json"
    data = (json.loads(path.read_text()) if path.exists() else
            {"task": task.name, "system": system, "metric": task.metric, "runs": []})
    data["runs"].append(record)
    path.write_text(json.dumps(data, indent=2))
    return path