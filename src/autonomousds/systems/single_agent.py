"""single_agent: one LLM agent with sandbox tools solves the whole task."""
from __future__ import annotations

import json
import time

import pandas as pd

from autonomousds.benchmark.tasks import Task, score
from autonomousds.systems.common import (append_result, count_failed_executions,
                                         new_run_id, run_agent, run_usage)
from autonomousds.systems.prompts import single_agent_prompt
from autonomousds.tools.agent_tools import make_tools, setup_run

MAX_STEPS = 15


def run_single_agent(task: Task, max_steps: int = MAX_STEPS) -> dict:
    """Run one LLM agent end to end on a task and score its predictions.

    The agent gets a fresh sandbox with train.csv and test.csv and the agent
    tools. Its predictions are scored against the hidden labels, the transcript
    is saved to runs/<run_id>/transcript.json, and a result record is appended
    to benchmark/results/<task>/single_agent.json.

    Args:
        task: The benchmark task to solve.
        max_steps: Tool-call budget for the agent.

    Returns:
        The result record: score, steps, time, tokens and cost.
    """
    run_id = new_run_id("single", task)
    sb = setup_run(task, run_id=run_id)
    tools, state = make_tools(sb, task)
    transcript: list = []
    print(f"Run {run_id} | task: {task.name} | max steps: {max_steps}")

    start = time.time()
    summary = run_agent("single_agent", single_agent_prompt(task, max_steps), tools,
                        "Solve the task. Start now.", run_id, max_steps, transcript)
    secs = time.time() - start

    holdout = None
    if state["submitted"]:
        holdout = round(score(task, pd.read_csv(sb.workdir / state["submitted"])), 4)

    (sb.workdir / "transcript.json").write_text(json.dumps(transcript, indent=2))
    record = {
        "run_id": run_id,
        "max_steps": max_steps,
        "submitted": state["submitted"] is not None,
        "holdout_score": holdout,
        "experiments": state["experiments"],
        "tool_calls": sum(len(e["tool_calls"] or []) for e in transcript
                          if e["type"] == "ai"),
        "failed_executions": count_failed_executions(sb.workdir),
        "seconds": round(secs, 1),
        "final_summary": summary,
        **run_usage(run_id),
    }
    append_result(task, "single_agent", record)

    print(f"Done in {secs:.0f}s | submitted: {record['submitted']} | "
          f"hidden_test: {holdout} | tool calls: {record['tool_calls']} | "
          f"failed code runs: {record['failed_executions']} | "
          f"est. cost if paid: ${record['est_cost_usd']}")
    return record