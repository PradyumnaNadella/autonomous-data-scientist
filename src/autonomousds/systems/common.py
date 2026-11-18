"""Helpers shared by all agent systems (single_agent, pipeline, debate)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from autonomousds.benchmark.tasks import ROOT, Task
from autonomousds.llm.models import LOG_PATH, get_chat_model

RESULTS_DIR = ROOT / "benchmark" / "results"
BUDGET_MARGIN = 3        # extra graph room so a finishing agent is never cut off


def new_run_id(system: str, task: Task) -> str:
    """Create a run id that sorts by time, e.g. pipeline-spaceship-titanic-20261116-165149.

    Args:
        system: System name, e.g. "pipeline".
        task: The task being solved.

    Returns:
        The run id.
    """
    return f"{system}-{task.name}-{time.strftime('%Y%m%d-%H%M%S')}"


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


def describe_call(call: dict) -> str:
    """Make a short, readable summary of a tool call for the terminal.

    Args:
        call: One tool call from an AIMessage (has "name" and "args").

    Returns:
        A one-line description, e.g. "Split Cabin into deck/side (24 lines)".
    """
    args = call["args"]
    if call["name"] == "run_python":
        lines = [l.strip() for l in args.get("code", "").splitlines() if l.strip()]
        comment = next((l.lstrip("# ") for l in lines if l.startswith("#")), None)
        first_code = next((l for l in lines
                           if not l.startswith(("import ", "from ", "#"))), "")
        return f"{comment or first_code} ({len(lines)} lines)"
    if call["name"] == "log_experiment":
        return f"{args.get('model', '')} = {args.get('cv_score', '')}"
    return ", ".join(str(v) for v in args.values())


def print_step(msg, agent: str = "agent") -> None:
    """Print a one-line summary of an agent step to the terminal.

    Args:
        msg: The newest message in the agent's conversation.
        agent: Name shown in front of the agent's lines.
    """
    if isinstance(msg, AIMessage) and msg.tool_calls:
        for call in msg.tool_calls:
            print(f"  [{agent}] -> {call['name']}: {describe_call(call)[:90]}")
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


def count_failed_executions(workdir: Path) -> int:
    """Count sandbox executions that ended in an error or timeout.

    Args:
        workdir: The run folder containing sandbox_log.jsonl.

    Returns:
        Number of failed code executions in the run.
    """
    log = workdir / "sandbox_log.jsonl"
    if not log.exists():
        return 0
    return sum(1 for l in log.read_text().splitlines()
               if json.loads(l)["exit_code"] != 0)


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


def with_budget(tools: list, max_steps: int) -> list:
    """Wrap tools so the agent always sees, and cannot exceed, its budget.

    Every result ends with "[tool calls used: k/N]". Two calls before the end it
    adds an instruction to finish. Past the budget, tools are not executed and
    only tell the agent to give its final answer.

    Args:
        tools: The tools to wrap.
        max_steps: Tool-call budget for this agent.

    Returns:
        Wrapped tools with the same names, descriptions and arguments.
    """
    used = {"n": 0}

    def wrap(t):
        def run(**kwargs):
            used["n"] += 1
            n = used["n"]
            if n > max_steps:
                return ("Tool budget exhausted: this call was NOT executed. "
                        "Do not call any more tools. Give your final answer now.")
            note = f"\n[tool calls used: {n}/{max_steps}]"
            if max_steps - n <= 2:
                note += " Almost out of budget: finish and give your final answer."
            return f"{t.invoke(kwargs)}{note}"

        return StructuredTool.from_function(func=run, name=t.name,
                                            description=t.description,
                                            args_schema=t.args_schema)

    return [wrap(t) for t in tools]


def run_agent(name: str, system_prompt: str, tools: list, user_message: str,
              run_id: str, max_steps: int, transcript: list) -> str:
    """Run one agent to completion and return its final text reply.

    Args:
        name: Role name, used in logs and printouts (e.g. "critic").
        system_prompt: The role's instructions.
        tools: Tools this role may use (empty list for text-only roles).
        user_message: The input for this turn (reports, plan, critique...).
        run_id: Run id, so LLM costs are logged against this run.
        max_steps: Tool-call budget for this agent.
        transcript: Shared list; this agent's messages are appended to it.

    Returns:
        The agent's final reply, or a note saying why it stopped.
    """
    agent = create_agent(model=get_chat_model(agent=name, run_id=run_id),
                         tools=with_budget(tools, max_steps),
                         system_prompt=system_prompt)
    limit = 2 * (max_steps + BUDGET_MARGIN) + 3       # safety net only
    messages, error = [], None
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"recursion_limit": limit},
            stream_mode="values",
        ):
            messages = chunk["messages"]
            print_step(messages[-1], agent=name)
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  [{name}] STOPPED: {error}")

    transcript.extend({"agent": name, "type": m.type, "text": message_text(m),
                       "tool_calls": getattr(m, "tool_calls", None)} for m in messages)
    final = next((message_text(m) for m in reversed(messages)
                  if isinstance(m, AIMessage) and not m.tool_calls), "")

    # Backup: stopped without a final answer -> ask once more, no tools allowed
    if not final and error and messages:
        history = messages
        if isinstance(history[-1], AIMessage) and history[-1].tool_calls:
            history = history[:-1]
        try:
            reply = get_chat_model(agent=name, run_id=run_id).invoke(
                [SystemMessage(system_prompt), *history,
                 HumanMessage("You have used all your tool calls. Reply now with "
                              "your final answer, in the required format.")])
            final = message_text(reply)
            print(f"  [{name}] (final answer after stop) {final[:150]}")
            transcript.append({"agent": name, "type": "ai", "text": final,
                               "tool_calls": None})
        except Exception as e:
            error += f" | final-answer request failed: {type(e).__name__}"

    if final:
        return final
    return f"(no final answer; stopped: {error})" if error else "(no final answer)"