# Runify R1 Backend

## Quick Start

### Prerequisites
- Python 3.10+ (written on 3.14)
- Docker (for PostgreSQL database)

### Setup

### Run with Docker (Recommended)

You can spin up the entire application stack (FastAPI web server and PostgreSQL database) with a single command:

```bash
# Build and start all services (web server + database)
docker compose up --build -d
```

* **API Documentation**: Once running, open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
* **Logs**: Monitor web server activity using `docker compose logs -f web`.

### Local Development Setup (Alternative)

If you prefer running the FastAPI web server locally on your host machine while keeping only the database containerized:

```bash
# 1. Start the PostgreSQL database container only
docker compose up postgres -d

# 2. Create a virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies using the pyproject.toml package specification
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env

# 5. Run Alembic migrations locally
alembic upgrade head

# 6. Run the API server locally on the host
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


### Run Tests

```bash
# Run all tests (unit + integration + metrics engine tests)
pytest

# Run the large multi-user integration suite in verbose mode
pytest tests/test_integration.py -v
```

### Frontend

See [frontend/README.md](frontend/README.md) for the React SPA setup.

```bash
cd frontend && npm install && npm run dev
```

### Development Utilities


```bash
# Run linter checks
ruff check .

# Run linter checks and automatically apply fixes
ruff check . --fix

# Format the Python codebase according to project style
ruff format .
```

### Seed Demo Data

Populate PostgreSQL with a large realistic dataset (50 athletes, ~750 activities, clubs, segments, social graph, etc.):

```bash
# Requires postgres running (docker compose up postgres -d) and migrations applied
python scripts/seed_bulk.py --clear

# Lighter run without wiping existing data (skips if DB already populated)
python scripts/seed.py
```

Login as **demo** / **demo1234** after seeding.

Options: `--athletes 100 --activities-per-athlete 25 --seed 42`
