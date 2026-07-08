"""Shared pytest fixtures for Runify API tests."""
import os
import httpx
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import app.models.activity
import app.models.athlete
import app.models.auth
import app.models.segment
import app.models.social  # noqa: F401
from app.db.base import Base
from app.db.schema_sync import sync_schema

# Connect to the Postgres database running in Docker (exposed on localhost:5432)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://runify:runify@localhost:5432/runify")
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def ensure_schema():
    """Ensure the schema exists once for the entire test session."""
    sync_schema()


@pytest.fixture
def db():
    """
    Provide a database session that is rolled back after each test.

    Uses a transaction + SAVEPOINT so all writes are undone without touching
    real data — the outer transaction is never committed.
    """
    connection = engine.connect()
    transaction = connection.begin()

    # Bind a session to the open connection so it shares the same transaction
    session = TestingSessionLocal(bind=connection)

    # Postgres requires a SAVEPOINT for nested rollbacks when using
    # begin_nested() so the ORM can roll back to the savepoint after flush
    # errors without killing the outer transaction.
    nested = connection.begin_nested()

    # Re-open the savepoint whenever the ORM session expires it (e.g. after commit)
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client():
    # Connect directly to the FastAPI server running inside the container
    base_url = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
    with httpx.Client(base_url=base_url, follow_redirects=True) as c:
        yield c
