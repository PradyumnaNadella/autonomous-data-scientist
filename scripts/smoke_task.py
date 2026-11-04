import pandas as pd
from autonomousds.benchmark.tasks import load_task, prepare, score

task = load_task("spaceship-titanic")
train, hold = prepare(task)
print(f"Agent train rows: {len(train)}, hidden test rows: {len(hold)}")

# Dumbest possible baseline: always predict the most common class
majority = train[task.target].mode()[0]
test = pd.read_csv(task.agent_dir / "test.csv")
preds = pd.DataFrame({task.id_column: test[task.id_column], task.target: majority})
print(f"Majority-class {task.metric}: {score(task, preds):.4f}")