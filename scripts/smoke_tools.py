from autonomousds.benchmark.tasks import load_task
from autonomousds.tools.agent_tools import make_tools, setup_run

task = load_task("spaceship-titanic")
sb = setup_run(task)
tools, state = make_tools(sb, task)
t = {x.name: x for x in tools}

print("1. list_files:\n", t["list_files"].invoke({}))

print("\n2. run_python:\n", t["run_python"].invoke({"code":
    "import pandas as pd\ndf = pd.read_csv('train.csv')\nprint(df.shape)"}))

print("\n2b. submit before logging experiments (should be rejected):")
print(t["submit_predictions"].invoke({"filename": "predictions.csv"}))

print("\n3. bad submission (1/0 instead of True/False):")
t["log_experiment"].invoke({"model": "baseline A", "cv_score": 0.5})
t["log_experiment"].invoke({"model": "baseline B", "cv_score": 0.5})
t["run_python"].invoke({"code":
    "import pandas as pd\nt = pd.read_csv('test.csv')\n"
    "pd.DataFrame({'PassengerId': t.PassengerId, 'Transported': 1})"
    ".to_csv('predictions.csv', index=False)"})
print(t["submit_predictions"].invoke({"filename": "predictions.csv"}))

print("\n4. good submission:")
t["run_python"].invoke({"code":
    "import pandas as pd\nt = pd.read_csv('test.csv')\n"
    "pd.DataFrame({'PassengerId': t.PassengerId, 'Transported': True})"
    ".to_csv('predictions.csv', index=False)"})
print(t["submit_predictions"].invoke({"filename": "predictions.csv"}))
print("state:", state)

print("\n5. read_file outside the folder (should be blocked):")
print(t["read_file"].invoke({"name": "../../.env"}))