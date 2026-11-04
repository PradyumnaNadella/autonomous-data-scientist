"""single_agent: one LLM agent with sandbox tools solves the whole task."""
from __future__ import annotations

import json
import time
import uuid

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from autonomousds.benchmark.tasks import Task, score
from autonomousds.llm.models import get_chat_model
from autonomousds.systems.prompts import single_agent_prompt
from autonomousds.systems.common import (append_result, message_text,
                                         print_step, run_usage)
from autonomousds.tools.agent_tools import make_tools, setup_run

MAX_STEPS = 15

def run_single_agent(task: Task, max_steps: int = MAX_STEPS) -> dict:
    """Run one LLM agent end to end on a task and score its predictions.

    The agent gets a fresh sandbox with train.csv and test.csv and the four
    agent tools. Its predictions are scored against the hidden labels, the
    transcript is saved to runs/<run_id>/transcript.json, and a result record
    is appended to benchmark/results/<task>/single_agent.json.

    Args:
        task: The benchmark task to solve.
        max_steps: Maximum number of tool calls before the run is stopped.

    Returns:
        The result record: score, steps, time, tokens, cost and any error.
    """
    run_id = f"single-{task.name}-{uuid.uuid4().hex[:6]}"
    sb = setup_run(task, run_id=run_id)
    tools, state = make_tools(sb, task)
    agent = create_agent(
        model=get_chat_model(agent="single_agent", run_id=run_id),
        tools=tools,
        system_prompt=single_agent_prompt(task, max_steps),
    )
    print(f"Run {run_id} | task: {task.name} | max steps: {max_steps}")

    messages, error, start = [], None, time.time()
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": "Solve the task. Start now."}]},
            config={"recursion_limit": 2 * max_steps + 1},
            stream_mode="values",
        ):
            messages = chunk["messages"]
            print_step(messages[-1], agent="single_agent")
    except Exception as e:                      # step limit, API error, etc.
        error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  STOPPED: {error}")
    secs = time.time() - start

    holdout = None
    if state["submitted"]:
        preds = pd.read_csv(sb.workdir / state["submitted"])
        holdout = round(score(task, preds), 4)

    # Full transcript, for debugging and the run viewer later
    transcript = [{"type": m.type, "text": message_text(m),
                   "tool_calls": getattr(m, "tool_calls", None)} for m in messages]
    (sb.workdir / "transcript.json").write_text(json.dumps(transcript, indent=2))

    record = {
        "run_id": run_id,
        "experiments": state["experiments"],
        "submitted": state["submitted"] is not None,
        "holdout_score": holdout,
        "tool_calls": sum(len(m.tool_calls) for m in messages
                          if isinstance(m, AIMessage)),
        "seconds": round(secs, 1),
        "error": error,
        "final_summary": next((message_text(m) for m in reversed(messages)
                               if isinstance(m, AIMessage) and not m.tool_calls), ""),
        **run_usage(run_id),
    }

    # Agents vary between runs, so results are appended, not overwritten
    append_result(task, "single_agent", record)

    print(f"Done in {secs:.0f}s | submitted: {record['submitted']} | "
          f"hidden_test: {holdout} | tool calls: {record['tool_calls']} | "
          f"LLM calls: {record['llm_calls']} | est. cost if paid: ${record['est_cost_usd']}")
    return record