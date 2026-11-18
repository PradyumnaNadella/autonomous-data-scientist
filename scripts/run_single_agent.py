import argparse

from autonomousds.benchmark.tasks import load_task, prepare
from autonomousds.systems.single_agent import run_single_agent

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="spaceship-titanic")
parser.add_argument("--max-steps", type=int, default=15)
args = parser.parse_args()

task = load_task(args.task)
if not (task.agent_dir / "train.csv").exists():
    prepare(task)
run_single_agent(task, max_steps=args.max_steps)