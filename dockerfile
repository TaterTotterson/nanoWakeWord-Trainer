FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REC_HOST=0.0.0.0 \
    REC_PORT=8792 \
    NWW_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-ui.txt requirements-train.txt ./
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements-ui.txt

COPY . /app

EXPOSE 8792
CMD ["./run.sh"]
