FROM python:3.12-slim

# libgomp1 is needed by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas numpy scikit-learn lightgbm xgboost matplotlib joblib pyarrow

# Run as a normal user, never root
RUN useradd -m sandbox
USER sandbox

# Writable locations for libraries (root filesystem will be read-only)
ENV HOME=/tmp MPLCONFIGDIR=/tmp MPLBACKEND=Agg PYTHONUNBUFFERED=1
WORKDIR /workspace