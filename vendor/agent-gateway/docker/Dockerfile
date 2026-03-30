# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_VIRTUALENVS_CREATE=true \
    PATH="/opt/poetry/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://install.python-poetry.org | python -

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md /app/
RUN poetry install --only main --no-root

COPY src /app/src
CMD ["poetry", "run", "uvicorn", "gateway.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
