# Build context is the repo root (see docker-compose.yml) so we can install
# the local inferlog SDK alongside the service.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# SDK changes less often than app code — install it on its own layer.
COPY sdk /tmp/sdk
RUN pip install "/tmp/sdk[openai,anthropic]"

COPY gateway/requirements.txt ./
RUN pip install -r requirements.txt

COPY db/schema.sql ./db/schema.sql
COPY gateway/app ./app
COPY gateway/tests ./tests
COPY gateway/pytest.ini ./

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
