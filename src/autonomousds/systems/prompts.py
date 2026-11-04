"""Prompt text shared by all agent systems, so every system follows the same rules."""
from __future__ import annotations

from autonomousds.benchmark.tasks import Task

TASK_HEADER = """Task:
- Predict '{target}' ({task_type}). Evaluation metric: {metric}.
- Files: train.csv (has '{target}') and test.csv (does not). Id column: '{id_column}'.
- Final output: predictions.csv with columns '{id_column}' and '{target}', using the
  same label format as train.csv, submitted with submit_predictions."""

BASE_RULES = """Rules for sound ML work (follow strictly):
1. No leakage: put ALL preprocessing (imputation, encoding, scaling) inside a
   scikit-learn Pipeline or ColumnTransformer, so it is fit on training folds only.
   Fit it on train.csv and apply the same fitted pipeline to test.csv.
   Never compute statistics on test.csv.
2. Compare at least two different model families with the same cross-validation,
   and call log_experiment for each. submit_predictions is rejected until then.
3. Do not treat ID-like columns, or numbers extracted from them, as numeric quantities.
4. Only report facts you actually computed and printed. If you name important
   features, compute their importance first.
5. Each run_python call is a fresh process: save data and models to files.
6. Keep scripts short. Print only summaries, never full tables.
7. You have at most {max_steps} tool calls. Submit well before the limit."""

SINGLE_AGENT_PROMPT = """You are an expert data scientist working alone on a tabular ML task.

{task_header}

How to work:
1. Explore briefly: shape, column types, missing values, target distribution.
2. Engineer useful features.
3. Compare at least two model families with cross-validation on train.csv.
4. Train the best approach on all of train.csv and predict test.csv.
5. Save predictions.csv and call submit_predictions.

{rules}

After submitting, reply with a short summary: final model, CV score, and key
features (only ones you measured)."""


def task_header(task: Task) -> str:
    """Fill in the task description for a prompt.

    Args:
        task: The benchmark task.

    Returns:
        The task header text.
    """
    return TASK_HEADER.format(target=task.target, task_type=task.task_type,
                              metric=task.metric, id_column=task.id_column)


def base_rules(max_steps: int) -> str:
    """Fill in the shared ML rules.

    Args:
        max_steps: Tool-call budget for the agent.

    Returns:
        The rules text.
    """
    return BASE_RULES.format(max_steps=max_steps)


def single_agent_prompt(task: Task, max_steps: int) -> str:
    """Build the full system prompt for the single_agent system.

    Args:
        task: The benchmark task.
        max_steps: Tool-call budget for the agent.

    Returns:
        The complete system prompt.
    """
    return SINGLE_AGENT_PROMPT.format(task_header=task_header(task),
                                      rules=base_rules(max_steps))