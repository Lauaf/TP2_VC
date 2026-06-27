FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.backend.txt /app/requirements.backend.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.11.0 torchvision==0.26.0 && \
    pip install --no-cache-dir -r /app/requirements.backend.txt \
    && rm -rf /var/lib/apt/lists/*

COPY ./src /app
COPY ./scripts /scripts
COPY ./.env.docker.example /app/.env

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
