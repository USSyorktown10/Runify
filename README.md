# Runify R1 Backend

FastAPI + PostgreSQL backend for Runify, a web-based athletic analysis and social platform.

## Quick Start

### Prerequisites
- Python 3.10+ (written on 3.14)
- Docker (for PostgreSQL database)

### Setup

```bash
# Start the PostgreSQL database container
docker compose up -d

# Create virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# Install dependencies using the pyproject.toml package specification
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env

# Run Alembic migrations (if applicable)
alembic upgrade head

# Run the API server locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* **Interactive API Documentation**: Once running, open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### Run Tests

```bash
# Run all tests (unit + integration + metrics engine tests)
pytest

# Run the large multi-user integration suite in verbose mode
pytest tests/test_integration.py -v
```

### Development Utilities

This codebase uses **Ruff** for fast linting and formatting.

```bash
# Run linter checks
ruff check .

# Run linter checks and automatically apply fixes
ruff check . --fix

# Format the Python codebase according to project style
ruff format .
```

### Seed Demo Data

```bash
python scripts/seed.py
```
