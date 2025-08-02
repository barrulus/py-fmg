# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment variables for Poetry
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    PATH="/opt/poetry/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "py_fmg.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
