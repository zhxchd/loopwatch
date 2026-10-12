FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libssl-dev libblas-dev liblapack-dev \
    libffi-dev libxml2-dev libxslt-dev && rm -rf /var/lib/apt/lists/*
COPY AlgoTune /opt/AlgoTune
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e /opt/AlgoTune
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    NUMBA_THREADING_LAYER=workqueue PYTHONPATH=/work/AlgoTune \
    ALGOTUNE_CONFIG_PATH=/work/config.yaml DATA_DIR=/work/data CODE_DIR=/work/solution \
    ALGOTUNE_HF_DISABLE=1 SKIP_DATASET_GEN=1 AGENT_MODE=1 ISOLATED_EVAL=1 \
    TOKENIZERS_PARALLELISM=false LOG_LEVEL=INFO \
    LOOPWATCH_PROXY_KEY=local-relay-placeholder SUMMARY_FILE=/work/agent_summary.json
WORKDIR /work
CMD ["sleep", "infinity"]
