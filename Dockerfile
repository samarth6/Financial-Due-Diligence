# ---- Stage 1: build the web page (Node is only needed here) ----
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- Stage 2: the app ----
FROM python:3.14-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app

# 1. Install the packages first, without the "-e ." line, so changing code
#    later does not reinstall torch (the slow part).
COPY requirements.txt ./
RUN grep -v '^-e' requirements.txt > /tmp/requirements-deps.txt \
 && pip install -r /tmp/requirements-deps.txt

# 2. Download the two models into the image, so a new container starts fast.
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-en-v1.5'); \
snapshot_download('BAAI/bge-reranker-large')"

# 3. The code, the raw filing and tables, and the built web page.
COPY pyproject.toml config.yaml build_financials_db.py ./
COPY src ./src
COPY data/raw ./data/raw
COPY --from=web /web/dist ./web/dist
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["dd", "serve"]