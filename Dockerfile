FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf TORCH_HOME=/app/models/torch MOTO_STATIC_DIR=/app/static
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git curl && rm -rf /var/lib/apt/lists/*
ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}
COPY pyproject.toml ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir '.[keypoints]'
RUN python -m app.models_download
COPY --from=frontend /frontend/dist /app/static
RUN useradd --create-home --uid 10001 moto && mkdir -p /app/data/user /app/models && chown -R moto:moto /app
ENV HF_HUB_OFFLINE=1
USER moto
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl --fail http://localhost:8080/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
