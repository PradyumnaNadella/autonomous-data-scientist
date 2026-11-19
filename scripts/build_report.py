"""Build result summaries for every task and write them into the README.

Reads benchmark/results/<task>/*.json, adds full metrics, and writes:
  benchmark/summary/<task>.json   full summary per system
  benchmark/summary/RESULTS.md    tables for all tasks
  README.md                       the same tables, between the RESULTS markers
"""
from __future__ import annotations

import json
import time

from autonomousds.benchmark.metrics import enrich_baseline, enrich_run, summarize_system
from autonomousds.benchmark.tasks import ROOT, Task, load_task

RESULTS_DIR = ROOT / "benchmark" / "results"
SUMMARY_DIR = ROOT / "benchmark" / "summary"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
BASELINES = ("direct", "automl")
ORDER = ["direct", "automl", "single_agent", "pipeline", "debate"]


def fmt(v, digits: int = 4) -> str:
    """Format a value for a Markdown table ("-" when missing)."""
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def fmt_time(seconds) -> str:
    """Format seconds as "45 s" or "9.1 min"."""
    if seconds is None:
        return "-"
    return f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def system_groups(system: str, runs: list[dict]) -> dict[str, list[dict]]:
    """Split runs into separately reported groups when step budgets differ.

    Args:
        system: System name.
        runs: That system's run records.

    Returns:
        {display name: runs}, e.g. "single_agent (45 steps)".
    """
    steps = sorted({r.get("max_steps") for r in runs if r.get("max_steps")})
    if len(steps) <= 1:
        return {system: runs}
    return {f"{system} ({s} steps)": [r for r in runs if r.get("max_steps") == s]
            for s in steps}


def summarize_task(task: Task) -> list[dict]:
    """Summarize every system that has results for a task, in a fixed order.

    Args:
        task: The benchmark task.

    Returns:
        One summary dict per system (or step-budget group).
    """
    rows = []
    files = sorted((RESULTS_DIR / task.name).glob("*.json"),
                   key=lambda p: (ORDER.index(p.stem) if p.stem in ORDER else 99, p.stem))
    for path in files:
        data = json.loads(path.read_text())
        if path.stem in BASELINES:
            rows.append(summarize_system(task, path.stem, [enrich_baseline(task, path.stem, data)]))
            continue
        runs = [enrich_run(task, r) for r in data.get("runs", [])]
        for name, group in system_groups(path.stem, runs).items():
            rows.append(summarize_system(task, name, group))
    return rows


def task_markdown(task: Task, rows: list[dict]) -> str:
    """Render one task's summary as Markdown tables.

    Args:
        task: The benchmark task.
        rows: Output of summarize_task.

    Returns:
        Markdown text.
    """
    m = task.metric
    direction = "higher" if task.higher_is_better else "lower"
    extra = next((list(r["model_metrics_mean"]) for r in rows if r["model_metrics_mean"]), [])
    extra = [k for k in extra if k != m]

    out = [f"### {task.name}", "",
           f"Main metric: **{m}** ({direction} is better), measured on a hidden "
           "test set no system ever sees.", "",
           f"| System | Runs | Success | {m} (mean ± std) | Worst | Best* | "
           + " | ".join(extra) + " |",
           "|" + "---|" * (6 + len(extra))]
    for r in rows:
        mm = r["model_metrics_mean"]
        out.append(
            f"| {r['system']} | {r['runs']} | {fmt(r['success_rate'], 2)} | "
            f"{fmt(r['score_mean'])} ± {fmt(r['score_std'])} | {fmt(r['score_worst'])} | "
            f"{fmt(r['score_best'])} | " + " | ".join(fmt(mm.get(k)) for k in extra) + " |")

    agents = [r for r in rows if r["llm_calls_mean"] is not None]
    if agents:
        out += ["", "| System | Median time | Cost / run | Cost / success | LLM calls | "
                "Failed code runs | Recovered | Timeouts | CV optimism | Approved | "
                "Revisions | Checks fired |",
                "|" + "---|" * 12]
        for r in agents:
            out.append(
                f"| {r['system']} | {fmt_time(r['seconds_median'])} | "
                f"${fmt(r['cost_mean_usd'], 3)} | ${fmt(r['cost_per_success_usd'], 3)} | "
                f"{fmt(r['llm_calls_mean'], 1)} | {fmt(r['failed_executions_mean'], 1)} | "
                f"{fmt(r['recovery_rate'], 2)} | {fmt(r['timeouts_total'])} | "
                f"{fmt(r['cv_optimism_mean'])} | {fmt(r['approval_rate'], 2)} | "
                f"{fmt(r['revisions_mean'], 1)} | {fmt(r['checks_fired_total'])} |")

    out += ["", "*Best: the single best run, picked with knowledge of the test set. "
            "Shows the upside of running a system several times, not a score you "
            "could choose in practice.", ""]
    return "\n".join(out)


def update_readme(markdown: str) -> None:
    """Replace the text between the RESULTS markers in README.md (append if missing).

    Args:
        markdown: The results section to insert.
    """
    block = f"{START}\n{markdown}\n{END}"
    text = README.read_text() if README.exists() else ""
    if START in text and END in text:
        text = text.split(START)[0] + block + text.split(END, 1)[1]
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    README.write_text(text)


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    sections = []
    for task_dir in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        task = load_task(task_dir.name)
        rows = summarize_task(task)
        (SUMMARY_DIR / f"{task.name}.json").write_text(json.dumps(rows, indent=2))
        sections.append(task_markdown(task, rows))
        print(f"Summarized {task.name}: {len(rows)} systems")

    markdown = (f"_Last updated: {time.strftime('%Y-%m-%d')}. "
                "Generated by `scripts/build_report.py`; do not edit by hand._\n\n"
                + "\n".join(sections))
    (SUMMARY_DIR / "RESULTS.md").write_text(markdown)
    update_readme(markdown)
    print(f"Wrote {SUMMARY_DIR / 'RESULTS.md'} and updated README.md")


if __name__ == "__main__":
    main()