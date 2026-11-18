"""pipeline: a team of agents splits the work, with a critic who can send it back."""
from __future__ import annotations

import json
import time
from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from autonomousds.benchmark.tasks import Task, score
from autonomousds.systems.common import (append_result, count_failed_executions,
                                         new_run_id, run_agent, run_usage)
from autonomousds.systems.prompts import role_prompt
from autonomousds.tools.agent_tools import make_tools, setup_run

STEPS = {"eda": 6, "feature_engineer": 8, "modeler": 12, "critic": 10}
MAX_REVISIONS = 2


class PipelineState(TypedDict, total=False):
    eda_report: str
    plan: str
    features_report: str
    modeling_report: str
    critique: str
    verdict: str          # APPROVE | REVISE features | REVISE modeling | NO VERDICT | SUBMITTED
    revisions: int
    verdicts: list
    final_report: str


def parse_verdict(text: str) -> str:
    """Read the critic's decision from the first line of its reply.

    Args:
        text: The critic's final reply.

    Returns:
        "APPROVE", "REVISE features", "REVISE modeling", or "NO VERDICT" if the
        reply has no readable verdict line.
    """
    first = text.strip().splitlines()[0].upper() if text.strip() else ""
    if "VERDICT" not in first:
        return "NO VERDICT"
    if "REVISE" in first:
        return "REVISE features" if "FEATURE" in first else "REVISE modeling"
    return "APPROVE"


def run_pipeline(task: Task, max_revisions: int = MAX_REVISIONS) -> dict:
    """Run the pipeline team end to end on a task and score its predictions.

    Args:
        task: The benchmark task to solve.
        max_revisions: How many times the critic may send work back.

    Returns:
        The result record: score, verdicts, revisions, time, tokens and cost.
    """
    run_id = new_run_id("pipeline", task)
    sb = setup_run(task, run_id=run_id)
    tools, tool_state = make_tools(sb, task)
    T = {t.name: t for t in tools}
    explore = [T["run_python"], T["list_files"], T["read_file"]]
    transcript: list = []
    print(f"Run {run_id} | task: {task.name} | max revisions: {max_revisions}")

    def agent(role: str, toolset: list, message: str) -> str:
        steps = STEPS.get(role, 1)
        return run_agent(role, role_prompt(role, task, steps), toolset, message,
                         run_id, steps, transcript)

    def read(name: str, limit: int = 4000) -> str:
        path = sb.workdir / name
        return path.read_text()[:limit] if path.is_file() else "(missing)"

    def sync_experiment_log() -> None:
        """Rewrite experiments.jsonl from the tool's record, discarding manual edits."""
        (sb.workdir / "experiments.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in tool_state["experiments"]))

    # ---- nodes ----
    def eda(state):
        return {"eda_report": agent("eda", explore, "Explore the data and report.")}

    def planner(state):
        return {"plan": agent("planner", [], f"EDA report:\n{state['eda_report']}")}

    def features(state):
        msg = f"Plan:\n{state['plan']}\n\nEDA report:\n{state['eda_report']}"
        if state.get("verdict") == "REVISE features":
            msg += ("\n\nThe critic asked for changes. Fix every numbered issue and "
                    f"say what you changed for each:\n{state['critique']}")
        report = agent("feature_engineer", explore, msg)
        stray = sb.workdir / "predictions.csv"
        if stray.exists():                       # not this role's job
            stray.unlink()
            print("  [check] removed predictions.csv made by the feature engineer")
        return {"features_report": report}

    def modeling(state):
        msg = f"Plan:\n{state['plan']}\n\nFeature report:\n{state['features_report']}"
        if state.get("verdict", "").startswith("REVISE"):
            msg += ("\n\nThe critic asked for changes. Fix every numbered issue and "
                    f"say what you changed for each:\n{state['critique']}")
        toolset = explore + [T["log_experiment"]]
        logged_before = len(tool_state["experiments"])
        report = agent("modeler", toolset, msg)
        problems = []
        if not (sb.workdir / "train_model.py").is_file():
            problems.append("save your final training script as train_model.py and "
                            "run it to produce predictions.csv")
        if len({e["model"] for e in tool_state["experiments"]}) < 2:
            problems.append("record each model's CV score with the log_experiment "
                            "TOOL (a tool call, not Python code), for at least 2 "
                            "different models")
        if (state.get("verdict", "").startswith("REVISE")
                and len(tool_state["experiments"]) == logged_before):
            problems.append("log the CV score of your revised model with the "
                            "log_experiment TOOL (a tool call, not Python code)")
        if problems:
            print(f"  [check] modeler incomplete: {'; '.join(problems)}")
            report = agent("modeler", toolset, msg +
                           "\n\nYour work is incomplete. You must:\n- " +
                           "\n- ".join(problems) +
                           f"\n\nYour previous report:\n{report}")
        return {"modeling_report": report}

    def critic(state):
        sync_experiment_log()
        msg = (f"Plan:\n{state['plan']}\n\nFeature report:\n{state['features_report']}"
               f"\n\nModeling report:\n{state['modeling_report']}")
        text = agent("critic", explore, msg)
        verdict = parse_verdict(text)
        print(f"  [critic] {verdict}")
        return {"critique": text, "verdict": verdict,
                "verdicts": state.get("verdicts", []) + [verdict],
                "revisions": state.get("revisions", 0) + verdict.startswith("REVISE")}

    def submit(state):
        result = T["submit_predictions"].invoke({"filename": "predictions.csv"})
        print(f"  [submit] {result}")
        transcript.append({"agent": "submit", "type": "tool", "text": result,
                           "tool_calls": None})
        if result.startswith("Accepted"):
            return {"verdict": "SUBMITTED"}
        return {"verdict": "REVISE modeling", "critique": f"Submission rejected: {result}",
                "revisions": state.get("revisions", 0) + 1}

    def reporter(state):
        sync_experiment_log()
        msg = "\n\n".join(f"{k}:\n{state.get(k, '')}" for k in
                          ("eda_report", "plan", "features_report",
                           "modeling_report", "critique"))
        msg += ("\n\nFINAL CODE (train_model.py) - this produced the predictions "
                f"and overrides the reports:\n{read('train_model.py')}"
                f"\n\nEXPERIMENT LOG (experiments.jsonl):\n{read('experiments.jsonl')}")
        return {"final_report": agent("reporter", [], msg)}

    # ---- routing ----
    def after_critic(state):
        if (state["verdict"] in ("APPROVE", "NO VERDICT")
                or state["revisions"] > max_revisions):
            return "submit"
        return "features" if state["verdict"] == "REVISE features" else "modeling"

    def after_submit(state):
        if state["verdict"] == "SUBMITTED" or state["revisions"] > max_revisions:
            return "reporter"
        return "modeling"

    # ---- graph ----
    g = StateGraph(PipelineState)
    for name, fn in [("eda", eda), ("planner", planner), ("features", features),
                     ("modeling", modeling), ("critic", critic),
                     ("submit", submit), ("reporter", reporter)]:
        g.add_node(name, fn)
    g.add_edge(START, "eda")
    g.add_edge("eda", "planner")
    g.add_edge("planner", "features")
    g.add_edge("features", "modeling")
    g.add_edge("modeling", "critic")
    g.add_conditional_edges("critic", after_critic, ["submit", "features", "modeling"])
    g.add_conditional_edges("submit", after_submit, ["reporter", "modeling"])
    g.add_edge("reporter", END)
    graph = g.compile()

    (sb.workdir / "graph.mmd").write_text(graph.get_graph().draw_mermaid())
    try:
        (sb.workdir / "graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
    except Exception as e:
        print(f"  (graph PNG not saved: {type(e).__name__})")

    start, error, final = time.time(), None, {}
    try:
        final = graph.invoke({"revisions": 0, "verdicts": []},
                             config={"recursion_limit": 40})
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  STOPPED: {error}")
    secs = time.time() - start

    holdout = None
    if tool_state["submitted"]:
        holdout = round(score(task, pd.read_csv(sb.workdir / tool_state["submitted"])), 4)

    (sb.workdir / "transcript.json").write_text(json.dumps(transcript, indent=2))
    verdicts = final.get("verdicts", [])
    record = {
        "run_id": run_id,
        "submitted": tool_state["submitted"] is not None,
        "holdout_score": holdout,
        "experiments": tool_state["experiments"],
        "critic_verdicts": verdicts,
        "approved": bool(verdicts) and verdicts[-1] == "APPROVE",
        "revisions": final.get("revisions", 0),
        "failed_executions": count_failed_executions(sb.workdir),
        "seconds": round(secs, 1),
        "error": error,
        "final_summary": final.get("final_report", ""),
        **run_usage(run_id),
    }
    append_result(task, "pipeline", record)

    print(f"Done in {secs:.0f}s | submitted: {record['submitted']} | "
          f"hidden_test: {holdout} | verdicts: {verdicts} | "
          f"failed code runs: {record['failed_executions']} | "
          f"est. cost if paid: ${record['est_cost_usd']}")
    return record