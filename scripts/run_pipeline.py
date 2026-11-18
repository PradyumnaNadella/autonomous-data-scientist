import argparse

from autonomousds.benchmark.tasks import load_task, prepare
from autonomousds.systems.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="spaceship-titanic")
parser.add_argument("--max-revisions", type=int, default=2)
args = parser.parse_args()

task = load_task(args.task)
if not (task.agent_dir / "train.csv").exists():
    prepare(task)
run_pipeline(task, max_revisions=args.max_revisions)