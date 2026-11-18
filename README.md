# autonomous-data-scientist
AI agents that solve ML problems end to end, comparing AutoML, a single agent, a team of agents, and agents that debate, on real Kaggle tasks.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
docker build -t autonomousds-sandbox:latest -f docker/sandbox.Dockerfile docker/
cp .env.example .env    # add your API keys

python scripts/run_direct.py --task <task>
python scripts/run_single_agent.py --task <task>
python scripts/run_pipeline.py --task <task>
```

Full guide to the code, outputs and commands: [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)