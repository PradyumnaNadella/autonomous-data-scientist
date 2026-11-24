"""Run one agent system on one task; started as a separate process by the API.

Example:
    python -m autonomousds.api.worker --task titanic --system single_agent \
        --run-id single-titanic-20261123-190000
"""
import argparse

from autonomousds.benchmark.tasks import load_task, prepare


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Run one agent system on one task.")
    p.add_argument("--task", required=True)
    p.add_argument("--system", required=True, choices=["single_agent", "pipeline"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    """Prepare the task's data if needed, then run the chosen system.

    Any exception is left uncaught on purpose: its traceback goes to the job log
    and the non-zero exit code tells the runner the run crashed.
    """
    args = parse_args()
    task = load_task(args.task)
    if not (task.agent_dir / "train.csv").exists():
        prepare(task)
    if args.system == "single_agent":
        from autonomousds.systems.single_agent import run_single_agent
        kwargs = {"max_steps": args.max_steps} if args.max_steps else {}
        run_single_agent(task, run_id=args.run_id, **kwargs)
    else:
        from autonomousds.systems.pipeline import run_pipeline
        run_pipeline(task, run_id=args.run_id)


if __name__ == "__main__":
    main()