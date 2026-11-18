# Project map: where to find what

Autonomous Data Scientist: AI agents that solve tabular ML tasks, compared
against classic baselines.

Placeholders used below:
- `<task>`: a dataset's name, from `name:` in its task file
- `<system>`: one of `direct`, `automl`, `single_agent`, `pipeline`
- `<run_id>`: one run of an agent system, e.g. `<system>-<task>-YYYYMMDD-HHMMSS`

Every folder that holds data or results is named after `<task>`, so each
dataset's files stay separate and new datasets need no code changes.

---

## 1. Source code: `src/autonomousds/`

| Area | File | What it does |
|---|---|---|
| LLM | `llm/models.py` | `get_chat_model()`: every LLM call goes through it. Applies the limits in `configs/llm.yaml` and logs tokens, cost and truncation. |
| Sandbox | `sandbox/docker_runner.py` | Runs agent-written code in an isolated Docker container (no internet, limited CPU, memory and time). Saves every script and its output. |
| Tasks | `benchmark/tasks.py` | Loads a task file, downloads its data, creates the hidden test split, scores predictions. |
| Baselines | `baselines/direct.py` | Default ML models with generic preprocessing. No LLM. |
| | `baselines/automl.py` | AutoGluon with a time limit. No LLM. |
| Agent tools | `tools/agent_tools.py` | What agents can call: run code, list and read files, log experiments, submit predictions (with format checks). |
| Agent systems | `systems/prompts.py` | All prompts. `BASE_RULES` are shared by every agent system. |
| | `systems/common.py` | Shared helpers: running an agent with a step budget, saving results, totalling cost. |
| | `systems/single_agent.py` | One agent does the whole task. |
| | `systems/pipeline.py` | A team: EDA, planner, feature engineer, modeler, critic (can send work back), submit, reporter. |

## 2. Configuration

| File | What it controls |
|---|---|
| `configs/llm.yaml` | Model, output limit, retries, timeout, paid-mode budgets, prices for cost estimates. |
| `benchmark/tasks/<task>.yaml` | One file per dataset. |
| `.env` | API keys and `ALLOW_PAID`. Never committed. |

Task file fields:

```yaml
name: <task>                  # used for every folder name
competition: <kaggle-slug>    # where the data is downloaded from
train_file: train.csv         # training file inside the download
target: <column>              # column to predict
id_column: <column>           # optional; omit it and a row_id is created
task_type: classification     # classification | regression
metric: accuracy              # accuracy | rmse | mae
holdout_fraction: 0.2         # share of rows hidden for scoring
seed: 42                      # makes the split reproducible
```

## 3. Commands: `scripts/`

| Command | What it does |
|---|---|
| `python scripts/run_direct.py --task <task>` | Run the direct baseline. |
| `python scripts/run_automl.py --task <task> --time-limit <seconds>` | Run AutoGluon. |
| `python scripts/run_single_agent.py --task <task> [--max-steps N]` | One single-agent run (added to results). |
| `python scripts/run_pipeline.py --task <task>` | One pipeline run (added to results). |
| `python scripts/smoke_*.py` | Quick checks: LLM, sandbox, task loading, tools. |

If `--task` is omitted, each script uses its built-in default task.

## 4. Where outputs go

| Location | Contents | In Git? |
|---|---|---|
| `data/<task>/raw/` | The downloaded dataset | No |
| `data/<task>/agent/` | `train.csv` and `test.csv` without labels: the only data agents see | No |
| `data/<task>/hidden/` | `test_labels.csv`: answers, used only for scoring | No |
| `benchmark/results/<task>/<system>.json` | Scores and metrics for every run of a system | **Yes** |
| `runs/<task>/<run_id>/` | Everything from one agent run (see below) | No |
| `runs/<task>/automl/` | AutoGluon's trained models | No |
| `logs/llm_calls.jsonl` | Every LLM call across all tasks: run, agent, tokens, cost, finish reason | No |
| `benchmark/notes.md` | Manual notes, e.g. runs stopped by hand | Yes |

Inside a run folder `runs/<task>/<run_id>/`:

| File | Contents |
|---|---|
| `transcript.json` | The full conversation: every agent message and tool call |
| `step_*.py`, `step_*.out.txt` | Each piece of code the agents ran, and its output |
| `sandbox_log.jsonl` | One line per code execution: exit code, time, output |
| `experiments.jsonl` | Models compared and their CV scores (official record from the tool) |
| `predictions.csv` | The submitted predictions |
| `features.py`, `train_model.py` | Pipeline only: feature code and final training script |
| `graph.mmd`, `graph.png` | Pipeline only: diagram of the agent graph |

## 5. Adding a new dataset

1. Make sure the data source allows downloading (for Kaggle, accept the rules).
2. Create `benchmark/tasks/<task>.yaml` with the fields above.
3. Run the baselines, then the agent systems, with `--task <task>`.

Supported: one CSV, classification or regression, metrics accuracy, RMSE or
MAE, up to tens of thousands of rows, not time series.

## 6. Useful commands

```bash
# Newest run folder for a task
ls -t runs/<task>/ | head -1

# Errors in a run, counted by type
d=runs/<task>/<run_id>
for f in $(grep -l "exit_code: 1" $d/*.out.txt); do grep -E "Error" "$f" | tail -1; done | sort | uniq -c

# Number of recorded runs for a system
python -c "import json;print(len(json.load(open('benchmark/results/<task>/<system>.json'))['runs']))"

# Last LLM calls
tail -3 logs/llm_calls.jsonl
```
