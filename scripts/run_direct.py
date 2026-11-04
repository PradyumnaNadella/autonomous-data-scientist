import argparse

from autonomousds.baselines.direct import run_direct
from autonomousds.benchmark.tasks import load_task, prepare

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="spaceship-titanic")
args = parser.parse_args()

task = load_task(args.task)
if not (task.agent_dir / "train.csv").exists():
    prepare(task)
run_direct(task)