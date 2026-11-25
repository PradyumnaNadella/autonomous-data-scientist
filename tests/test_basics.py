"""Basic checks that run in CI without Docker or an LLM key."""
import pytest
from fastapi.testclient import TestClient

from autonomousds.api.app import app
from autonomousds.benchmark.tasks import TASKS_DIR, Task, load_task

client = TestClient(app)      # without "with", the background runner is not started


def test_task_validation():
    """Invalid task settings are rejected when the task is created."""
    with pytest.raises(ValueError):    # metric does not match task type
        Task(name="t", target="y", task_type="classification", metric="rmse",
             source="csv", url="https://example.com/a.csv")
    with pytest.raises(ValueError):    # unsafe name
        Task(name="../bad", target="y", task_type="regression", metric="mae",
             source="csv", url="https://example.com/a.csv")


def test_committed_tasks_load():
    """Every task YAML in the repo is valid."""
    for path in TASKS_DIR.glob("*.yaml"):
        load_task(path.stem)


def test_read_endpoints():
    """Health and task listing work."""
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/tasks").status_code == 200


def test_run_requires_key(monkeypatch):
    """Starting a run without the right key is refused."""
    monkeypatch.setenv("RUN_API_KEY", "secret")
    r = client.post("/runs", json={"system": "single_agent", "task": "tips"},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401