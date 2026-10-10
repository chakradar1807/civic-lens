# Works on Hugging Face Spaces (Docker), Render, Railway. Port 7860 by default.
FROM python:3.11-slim

ARG WITH_YOLO=true
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-yolo.txt ./
RUN pip install -r requirements.txt
RUN if [ "$WITH_YOLO" = "true" ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install -r requirements-yolo.txt ; \
    fi

COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
