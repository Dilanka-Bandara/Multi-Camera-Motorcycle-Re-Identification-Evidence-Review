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
RUN if [ "$TORCH_VARIANT" = cu124 ]; then \
      pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    else pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi
COPY pyproject.toml ./
COPY backend/ ./backend/
RUN pip install --no-cache-dir '.[keypoints]'
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
ENV MOTO_CONFIG_DIR=/app/configs
RUN python -m app.models_download
RUN python /app/scripts/make_dev_fixture.py --out /app/data/fixtures/dev
COPY --from=frontend /frontend/dist /app/static
RUN useradd --create-home --uid 10001 moto && install -d -o moto -g moto /app/data/user \
    && chmod 0644 /app/models/dinov2-small/model.safetensors
ENV HF_HUB_OFFLINE=1
USER moto
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('/app/models/dinov2-small', local_files_only=True)"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl --fail http://localhost:8080/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
