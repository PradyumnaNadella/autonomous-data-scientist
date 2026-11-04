import argparse

from autonomousds.baselines.automl import run_automl
from autonomousds.benchmark.tasks import load_task, prepare

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="spaceship-titanic")
parser.add_argument("--time-limit", type=int, default=300)
args = parser.parse_args()

task = load_task(args.task)
if not (task.agent_dir / "train.csv").exists():
    prepare(task)
run_automl(task, time_limit=args.time_limit)