"""Web API: benchmark results, run details, and starting new runs."""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from autonomousds.api import runner, store
from autonomousds.benchmark.tasks import NAME_PATTERN, create_csv_task, load_task
from autonomousds.sandbox.docker_runner import check_docker
from autonomousds.systems.single_agent import MAX_STEPS
from autonomousds.trigger import TRIGGER_ENV

load_dotenv()
os.environ.setdefault(TRIGGER_ENV, "api")      # runs started here are marked "api"
TIME_LIMIT_MINUTES = int(os.getenv("RUN_TIME_LIMIT_MINUTES", "30"))
RUNNER: runner.Runner = runner.LocalRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background runner when the API starts."""
    RUNNER.start()
    yield


app = FastAPI(title="Autonomous Data Scientist API", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class RunRequest(BaseModel):
    """Body of POST /runs: an existing task, or a new CSV dataset."""
    system: Literal["single_agent", "pipeline"]
    task: str | None = Field(None, description="Existing task name, e.g. 'titanic'")
    name: str | None = Field(None, description="Name for a new CSV task")
    url: str | None = Field(None, description="http(s) URL of the CSV")
    target: str | None = None
    task_type: str | None = Field(None, description="classification or regression")
    metric: str | None = Field(None, description="accuracy, rmse or mae")
    id_column: str | None = None
    max_steps: int | None = Field(None, ge=1, le=45, description="single_agent only")
    time_limit_minutes: int | None = Field(None, ge=1, description="Can only lower the server limit")

    @model_validator(mode="after")
    def check_dataset(self) -> RunRequest:
        """Require either an existing task or a complete new dataset, not both."""
        new = [self.name, self.url, self.target, self.task_type, self.metric]
        if self.task and any(v is not None for v in new + [self.id_column]):
            raise ValueError("Give either 'task' or a new dataset, not both")
        if not self.task and any(v is None for v in new):
            raise ValueError("A new dataset needs name, url, target, task_type and metric")
        if self.system == "pipeline" and self.max_steps is not None:
            raise ValueError("max_steps applies only to single_agent")
        return self


def require_key(x_api_key: str | None = Header(None)) -> None:
    """Allow the request only with the right X-API-Key header.

    Args:
        x_api_key: Value of the X-API-Key header.

    Raises:
        HTTPException: 503 if no key is configured, 401 if the key is wrong.
    """
    expected = os.getenv("RUN_API_KEY")
    if not expected:
        raise HTTPException(503, "Starting runs is disabled: RUN_API_KEY is not set")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "Invalid or missing X-API-Key")


def _get(fn, *args):
    """Call a store function and turn 'not found' into an HTTP 404."""
    try:
        return fn(*args)
    except store.NotFound as e:
        raise HTTPException(status_code=404, detail=f"Not found: {e}")


@app.get("/health")
def health() -> dict:
    """Liveness check for hosting platforms."""
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> list[dict]:
    """All tasks with results."""
    return store.list_tasks()


@app.get("/tasks/{task}/summary")
def summary(task: str) -> list[dict]:
    """Per-system summary metrics for one task."""
    return _get(store.task_summary, task)


@app.get("/tasks/{task}/runs")
def runs(task: str, system: str | None = None) -> list[dict]:
    """Agent runs for one task, optionally filtered by system."""
    return _get(store.list_runs, task, system)


@app.post("/runs", status_code=202, dependencies=[Depends(require_key)])
def start_run(req: RunRequest) -> dict:
    """Start an agent run in the background and return its run id immediately."""
    try:
        check_docker()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    try:
        if req.task:
            if not NAME_PATTERN.fullmatch(req.task):
                raise FileNotFoundError(f"Unknown task '{req.task}'")
            task = load_task(req.task)
        else:
            task, _, _ = create_csv_task(name=req.name, target=req.target,
                                         task_type=req.task_type, metric=req.metric,
                                         url=req.url, id_column=req.id_column)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except FileExistsError as e:
        raise HTTPException(409, f"{e}. To run on it, send {{'task': '{req.name}'}}")
    except (ValueError, OSError) as e:
        raise HTTPException(400, str(e))

    limit = min(req.time_limit_minutes or TIME_LIMIT_MINUTES, TIME_LIMIT_MINUTES)
    max_steps = (req.max_steps or MAX_STEPS) if req.system == "single_agent" else None
    job = runner.create_job(task, req.system, max_steps, limit * 60)
    RUNNER.submit(job)
    return {"run_id": job["run_id"], "task": task.name, "status": job["status"],
            "status_url": f"/runs/{task.name}/{job['run_id']}"}


@app.get("/runs/{task}/{run_id}")
def run(task: str, run_id: str) -> dict:
    """Job status (for API-started runs) plus full run detail once it exists."""
    job = runner.read_job(run_id)
    if job and job["task"] != task:
        job = None
    try:
        detail = store.run_detail(task, run_id)
    except store.NotFound:
        if job is None:
            raise HTTPException(404, f"Not found: {run_id}")
        detail = {"record": None, "transcript": [], "steps": [], "files": []}
    return {"job": job, **detail}


@app.get("/runs/{task}/{run_id}/files/{name}", response_class=PlainTextResponse)
def run_file(task: str, run_id: str, name: str) -> str:
    """One viewable file from a run, as plain text."""
    return _get(store.run_file, task, run_id, name)