# ============================================================
# deploy/api.Dockerfile — the recipe for the API image (code + packages + model, no PyTorch)
# Build context = the repo root (see deploy/docker-compose.yml); .dockerignore allows api/, the five
# required model files, and optional XAI artifacts — never the 9 GB CSV, the 16 GB database, or .env.
#
# CONCEPTS
#   layers + cache : each instruction is a layer; unchanged layers are reused. Requirements are installed
#                    BEFORE the code is copied, so editing a .py file doesn't reinstall every package.
#   multi-stage    : stage 1 installs packages into a virtualenv; stage 2 starts clean and copies only that
#                    virtualenv — no pip caches or build leftovers in the final image.
#   non-root user  : if someone broke into the app, they would not be root inside the container.
#   0.0.0.0        : inside a container, 127.0.0.1 means "this container only" — listen on all interfaces so
#                    Docker can forward port 8000 from outside.
# ============================================================

# ---- stage 1: install the 9 pinned packages into /venv
FROM python:3.11-slim AS build
RUN python -m venv /venv
COPY api/requirements.txt /tmp/requirements.txt
RUN /venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# ---- stage 2: the image that runs
FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /venv /venv
# the model bundle (training/serving/): changes rarely -> its own layer, before the code
# .dockerignore is an allow-list: this directory contains only the five production files and, after
# parity validation, the three optional XAI artifacts.
COPY training/outputs/serving_v3/ /app/model/
# the code: changes most often -> last layer
COPY api/ /app/api/
ENV PATH=/venv/bin:$PATH \
    MODEL_DIR=/app/model \
    ORT_THREADS=2 \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
# one worker: each worker holds its own copy of the model in RAM (the server has ~2 GB)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
