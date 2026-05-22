# Build context is the repo root (see docker-compose.yml) so the shared
# db/schema.sql can be copied in. One image, two entrypoints: the API runs
# `uvicorn app.api:app`, the worker runs `python -m app.worker`.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY ingestion/requirements.txt ./
RUN pip install -r requirements.txt

COPY db/schema.sql ./db/schema.sql
COPY ingestion/app ./app
COPY ingestion/tests ./tests
COPY ingestion/pytest.ini ./

EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
