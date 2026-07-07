"""Shared pytest fixtures for Runify API tests."""
import os
import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models.activity
import app.models.athlete
import app.models.auth
import app.models.segment
import app.models.social  # noqa: F401
from app.db.base import Base

# Connect to the Postgres database running in Docker (exposed on localhost:5432)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://runify:runify@localhost:5432/runify")
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_db(db):
    # Ensure tables exist (no-op if already created by Alembic in the container)
    Base.metadata.create_all(bind=engine)
    
    # Delete all data from tables in reversed order of dependency to prevent foreign key issues
    for table in reversed(Base.metadata.sorted_tables):
        db.execute(table.delete())
    db.commit()
    yield


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    # Connect directly to the FastAPI server running inside the container
    base_url = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
    with httpx.Client(base_url=base_url, follow_redirects=True) as c:
        yield c

