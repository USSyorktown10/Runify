# Runify R1 Backend

FastAPI + PostgreSQL backend for the Runify running platform (Release 1).

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

## Architecture

- **FastAPI** — REST API matching `RUNIFY_R1.md`
- **PostgreSQL** — primary datastore
- **SQLAlchemy 2.0** — ORM with dynamic metrics tables
- **Metrics engine** — FIT/GPX/TCX parsing with VO2max, GAP, NP, zones, distributions

## Spec deviations (documented in plan)

- `activity_type` added to activities (run-focused values)
- `PATCH /athlete/profile` for private profile fields
- `Gear.total_mileage` computed field + `UpdateGearRequest.max_mileage`
- Session management endpoints require bearer auth
- `GetStreamRequest.start_date`/`end_date` for athlete streams

## Project structure

```
app/
  api/routers/     # Endpoint handlers per domain
  core/            # Config, security, pagination, errors
  db/              # SQLAlchemy session
  models/          # ORM models
  schemas/         # Pydantic request/response types
  services/        # Business logic + metrics engine
tests/
scripts/seed.py
```
