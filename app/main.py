from contextlib import asynccontextmanager

from fastapi import FastAPI

import app.models.activity

# Import all models so metadata is registered
import app.models.athlete
import app.models.auth
import app.models.segment
import app.models.social
from app.api.routers import (
    activities,
    athletes,
    authentication,
    clubs,
    gear,
    integrations,
    moderation,
    notifications,
    routes,
    segments,
    social,
    webhooks,
)
from app.core.config import get_settings
from app.core.errors import RunifyError, runify_error_handler
from app.db.base import Base
from app.db.session import engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)
    yield


application = FastAPI(
    title="Runify API",
    description="Runify Release 1 — Strava-like running platform with advanced metrics",
    version="1.0.0",
    lifespan=lifespan,
)

application.add_exception_handler(RunifyError, runify_error_handler)

application.include_router(authentication.router)
application.include_router(athletes.router)
application.include_router(activities.router)
application.include_router(gear.router)
application.include_router(segments.router)
application.include_router(routes.router)
application.include_router(social.router)
application.include_router(clubs.router)
application.include_router(notifications.router)
application.include_router(moderation.router)
application.include_router(integrations.router)
application.include_router(webhooks.router)


@application.get("/health")
def health():
    return {"status": "ok"}

# Uvicorn entrypoint: uvicorn app.main:application
app = application
