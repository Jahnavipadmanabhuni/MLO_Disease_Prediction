FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY templates ./templates
COPY static ./static
COPY models ./models
COPY params.yaml .

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]