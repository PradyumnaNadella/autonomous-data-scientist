"""One factory for every chat model: limits from llm.yaml + cost logging."""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs" / "llm.yaml"
LOG_PATH = ROOT / "logs" / "llm_calls.jsonl"

# Which .env variable holds the key for each provider
API_KEY_ENV = {
    "google_genai": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}


class BudgetExceeded(RuntimeError):
    pass


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def paid_mode(cfg: dict) -> bool:
    env_flag = os.getenv("ALLOW_PAID", "false").lower() == "true"
    return bool(cfg.get("allow_paid")) and env_flag


def total_paid_spend() -> float:
    if not LOG_PATH.exists():
        return 0.0
    with LOG_PATH.open() as f:
        return sum(r["est_cost_usd"] for r in map(json.loads, f) if r.get("paid"))


class CostLogger(BaseCallbackHandler):
    """Checks budgets before each call and logs tokens/cost after it."""

    raise_error = True  # let BudgetExceeded actually stop the call

    def __init__(self, cfg: dict, model: str, agent: str, run_id: str):
        self.cfg, self.model, self.agent, self.run_id = cfg, model, agent, run_id
        self.paid = paid_mode(cfg)
        self.run_cost = 0.0
        self._start = 0.0

    def on_chat_model_start(self, *args, **kwargs):
        if self.paid:
            if self.run_cost >= self.cfg["per_run_budget_usd"]:
                raise BudgetExceeded(f"Run {self.run_id} hit its per-run budget")
            if total_paid_spend() >= self.cfg["total_budget_usd"]:
                raise BudgetExceeded("Total budget reached")
        self._start = time.time()

    def on_llm_end(self, response, **kwargs):
        usage = {}
        for gens in response.generations:
            for g in gens:
                msg = getattr(g, "message", None)
                usage = getattr(msg, "usage_metadata", None) or usage
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)

        price = self.cfg.get("prices", {}).get(self.model)
        cost = ((in_tok * price["input"] + out_tok * price["output"]) / 1_000_000
                if price else 0.0)
        if self.paid:
            self.run_cost += cost

        LOG_PATH.parent.mkdir(exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": self.run_id,
                "agent": self.agent,
                "model": self.model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "est_cost_usd": round(cost, 6),
                "paid": self.paid,
                "seconds": round(time.time() - self._start, 2),
            }) + "\n")


def get_chat_model(agent: str = "unknown", run_id: str | None = None,
                   model: str | None = None):
    """Use this everywhere instead of calling init_chat_model directly."""
    cfg = load_config()
    name = model or cfg["default_model"]
    provider = name.split(":")[0]
    key_env = API_KEY_ENV.get(provider)

    return init_chat_model(
        model=name,
        api_key=os.getenv(key_env) if key_env else None,
        max_tokens=cfg["max_output_tokens"],
        max_retries=cfg["max_retries"],
        timeout=cfg["timeout_seconds"],
        callbacks=[CostLogger(cfg, name, agent, run_id or uuid.uuid4().hex[:8])],
    )