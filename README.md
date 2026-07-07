# Runify R1 Backend

FastAPI + PostgreSQL backend for the release 1

## Quick start

### Prerequisites
- Python 3.12+ (project venv uses 3.14)
- Docker (for PostgreSQL)

### Setup

```bash
# Start PostgreSQL
docker compose up -d

# Create virtualenv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env

# Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Run tests

```bash
pytest tests/ -v                  # all tests (unit + integration + metrics)
pytest tests/test_integration.py -v   # large multi-user integration suite only
```

### Seed demo data

```bash
python scripts/seed.py
```

