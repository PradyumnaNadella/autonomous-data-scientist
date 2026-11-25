# API image: serves results and starts agent runs.
# Generic on purpose: nothing machine-specific here (that lives in compose.yaml).
FROM python:3.12-slim

# Docker command-line tool only (no Docker engine): lets the API start sandbox
# containers on the host's Docker when the host socket is mounted in.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

# Install dependencies only. The package itself is not installed: the code is
# found via PYTHONPATH, so the project folder (ROOT) is wherever src/ lives.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && pip uninstall -y autonomousds

# Code and committed results (read-only use, e.g. a hosted demo)
COPY scripts ./scripts
COPY configs ./configs
COPY benchmark ./benchmark

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 HOME=/tmp
RUN useradd -m app
USER app

EXPOSE 8000
CMD ["uvicorn", "autonomousds.api.app:app", "--host", "0.0.0.0", "--port", "8000"]