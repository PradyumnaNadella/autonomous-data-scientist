"""Add a single-table CSV dataset as a benchmark task and prepare its data.

Example:
    python scripts/add_task.py --name titanic \
        --url https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv \
        --target Survived --task-type classification --metric accuracy --id-column PassengerId
"""
import argparse
import sys

from autonomousds.benchmark.tasks import METRICS, TASKS_DIR, create_csv_task


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Add a CSV dataset as a benchmark task.")
    p.add_argument("--name", required=True, help="Task name, e.g. house-prices")
    p.add_argument("--target", required=True, help="Column to predict")
    p.add_argument("--task-type", required=True, choices=["classification", "regression"])
    p.add_argument("--metric", required=True, choices=sorted(METRICS))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="http(s) URL of the CSV")
    source.add_argument("--path", help="Local path of the CSV")
    p.add_argument("--id-column", default=None,
                   help="Row ID column (default: create 'row_id')")
    p.add_argument("--force", action="store_true",
                   help="Replace an existing task that has no results yet")
    return p.parse_args()


def main() -> None:
    """Create the task and print a short summary, or a clean error message."""
    args = parse_args()
    try:
        task, train, hold = create_csv_task(
            name=args.name, target=args.target, task_type=args.task_type,
            metric=args.metric, url=args.url, path=args.path,
            id_column=args.id_column, force=args.force)
    except (OSError, ValueError) as e:      # OSError covers file and network errors
        sys.exit(f"Error: {e}")
    print(f"Saved {TASKS_DIR / f'{task.name}.yaml'}")
    print(f"Agent train: {len(train)} rows x {train.shape[1]} columns")
    print(f"Hidden test: {len(hold)} rows")


if __name__ == "__main__":
    main()