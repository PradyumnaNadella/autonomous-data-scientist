"""Run one agent system on a task in the foreground, with a time limit.

Does what POST /runs does (failure records, container cleanup, report refresh),
but waits for the result. Used by the GitHub Actions "run" workflow.

Examples:
    python scripts/run_job.py --system single_agent --task titanic
    python scripts/run_job.py --system pipeline --name tips \
        --url https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv \
        --target tip --task-type regression --metric mae
"""
import argparse
import json
import sys

from autonomousds.api.runner import LocalRunner, create_job, read_job
from autonomousds.benchmark.tasks import create_csv_task, load_task
from autonomousds.sandbox.docker_runner import check_docker
from autonomousds.systems.single_agent import MAX_STEPS


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Run one agent system on a task.")
    p.add_argument("--system", required=True, choices=["single_agent", "pipeline"])
    p.add_argument("--task", help="Existing task name")
    p.add_argument("--name", help="Name for a new CSV task")
    p.add_argument("--url", help="http(s) URL of the CSV")
    p.add_argument("--target")
    p.add_argument("--task-type", choices=["classification", "regression"])
    p.add_argument("--metric")
    p.add_argument("--id-column")
    p.add_argument("--max-steps", type=int, help="single_agent only")
    p.add_argument("--time-limit-minutes", type=int, default=30)
    return p.parse_args()


def main() -> None:
    """Resolve or create the task, run the job, and print its final status."""
    args = parse_args()
    try:
        if args.task:
            task = load_task(args.task)
        else:
            task, _, _ = create_csv_task(name=args.name, target=args.target,
                                         task_type=args.task_type, metric=args.metric,
                                         url=args.url, id_column=args.id_column)
        check_docker()
    except (OSError, ValueError, RuntimeError, TypeError) as e:
        sys.exit(f"Error: {e}")

    max_steps = (args.max_steps or MAX_STEPS) if args.system == "single_agent" else None
    job = create_job(task, args.system, max_steps, args.time_limit_minutes * 60)
    print(f"Started {job['run_id']}")
    LocalRunner().run_now(job)
    print(json.dumps(read_job(job["run_id"]), indent=2))


if __name__ == "__main__":
    main()